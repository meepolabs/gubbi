"""Arq job: extract dated journal entries from a saved conversation.

See ``extract_conversation`` function docstring for the CALLER CONTRACT
(security-critical).

Connection-split design:
  Phase 1 -- conn1: idempotency check, role-filtered load, rule-based skips.
  Phase 2 -- LLM pipeline, then embeddings: NO database connection held.
  Phase 3 -- conn2: persistence under an explicit nested SAVEPOINT transaction.

This ensures the database connection is released during the LLM calls
(which can take minutes for a long conversation), so the pool is not
exhausted when max_jobs concurrent workers are all mid-LLM.
"""

from __future__ import annotations

import asyncio
import functools
import json
import time
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Any, Final, cast
from uuid import UUID, uuid4

import structlog
from gubbi_common.audit.targets import TargetKind
from gubbi_common.budget import PRE_CHARGE_CENTS, current_period_start
from gubbi_common.db.user_scoped import user_scoped_connection
from opentelemetry import trace
from opentelemetry.metrics import Counter, get_meter

from gubbi.audit import record_audit
from gubbi.config import get_settings
from gubbi.constants import EXTRACTION_JOB_DEADLINE_SECS, EXTRACTION_VERSION
from gubbi.extraction.dedupe import embed_texts
from gubbi.extraction.llm.provider import (
    LLMProviderError,
    LLMRateLimitError,
    LLMRefusalError,
    LLMSchemaError,
    LLMTruncatedError,
    LLMUsage,
)
from gubbi.extraction.persist import ExtractionToPersist, persist_extraction
from gubbi.extraction.pipeline import (
    PipelineDeadlineError,
    PipelineFailure,
    PipelineOutcome,
    run_pipeline,
)
from gubbi.extraction.skips import content_hash, evaluate_skip
from gubbi.models.conversation import keep_conversation_roles
from gubbi.storage.exceptions import ConversationNotFoundError
from gubbi.storage.repositories import conversations as conv_repo
from gubbi.storage.repositories import extraction_jobs
from gubbi.storage.repositories import topics as topic_repo
from gubbi.telemetry.attrs import (
    _NS_PER_MS,
    _TRACER_NAME,
    SpanNames,
    safe_set_attributes,
)
from gubbi.telemetry.sanitized_errors import record_exception_sanitized, safe_error_fields
from gubbi.validation import harden_llm_topic_path, local_today

if TYPE_CHECKING:
    from collections.abc import Sequence

    import asyncpg

    from gubbi.crypto.cipher import ContentCipher
    from gubbi.extraction.context import ExtractionContext
    from gubbi.models.conversation import Message
    from gubbi.storage.repositories.conversations import ExtractionState

__all__: list[str] = ["extract_conversation"]

logger = structlog.get_logger(__name__)


# Counter incremented when the worker fails BEFORE the period_start lookup
# completes. In that window we cannot guarantee the runtime period equals the
# bucket the pre-charge debited (a period boundary may have been crossed since
# ingest), so the refund is skipped. Manual reconcile signal.
@functools.lru_cache(maxsize=1)
def _get_extraction_refund_skipped_counter() -> Counter:
    """Lazily create the refund-skipped counter against the live meter.

    Creating the counter at module scope would bind it to the NoOp meter that
    exists before ``configure_otel`` runs, silently discarding every
    ``.add(...)``. Deferring creation to first call (and re-priming via
    ``rebind_metrics_after_configure``) binds it to the configured provider.
    Mirrors the canonical pattern in ``gubbi.telemetry.metrics.initialize_metrics``.
    """
    return get_meter("gubbi").create_counter(
        name="extraction.refund_skipped_total",
        description=(
            "Pre-charge refund skipped because the worker failed before the "
            "pre-charge period was known; partitioned by reason."
        ),
        unit="1",
    )


# Arq enforces its job timeout by cancelling the task, so a cancelled job is
# recorded as having run out of time.
_CANCELLED_ERROR_CODE: Final = "deadline_exceeded"
# Arq waits for a cancelled job to finish, so the cancellation cleanup is kept
# short and bounded.
_CANCEL_CLEANUP_TIMEOUT_SECS: Final = 10.0
# A cleanup still running when its job is cancelled a second time is held here
# so the event loop does not garbage-collect it before it finishes.
_DETACHED_CLEANUPS: set[asyncio.Task[None]] = set()


# ---------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Job:
    """Identity of one job attempt; ``job_uuid`` is None when there is no job row."""

    pool: asyncpg.Pool
    cipher: ContentCipher
    user_uuid: UUID
    conversation_id: int
    job_uuid: UUID | None
    extraction_attempt_id: str
    log: structlog.stdlib.AsyncBoundLogger


@dataclass(frozen=True)
class _Bucket:
    """Budget period the pre-charge was debited to; ``known`` gates every refund.

    ``pre_charged_cents`` is what is still pre-charged for this job: zero
    once an earlier run of it settled, so a revived run charges its full cost.
    """

    period_start: date
    known: bool
    pre_charged_cents: int = PRE_CHARGE_CENTS


@dataclass(frozen=True)
class _Loaded:
    """Phase 1 output: role-filtered transcript, topic catalog and extraction state."""

    messages: tuple[Message, ...]
    existing_topics: tuple[str, ...]
    state: ExtractionState


@dataclass(frozen=True)
class _Spend:
    """Token usage and estimated cost of every model call the job made."""

    usage: LLMUsage
    llm_calls: int
    chunks: int
    cents: int


_NO_SPEND = _Spend(usage=LLMUsage(), llm_calls=0, chunks=0, cents=0)


@dataclass(frozen=True)
class _Extracted:
    """Phase 2 output: the pipeline outcome, its hardened topic, and what it cost."""

    outcome: PipelineOutcome
    topic_path: str | None
    spend: _Spend


def _result(
    topic_path: str | None, entries_created: int, spend: _Spend, *, skipped: bool
) -> dict[str, Any]:
    return {
        "topic_path": topic_path,
        "entries_created": entries_created,
        "input_tokens": spend.usage.input,
        "output_tokens": spend.usage.output,
        "cents_spent": spend.cents,
        "skipped": skipped,
    }


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------


_ERROR_CODES: tuple[tuple[type[Exception], str], ...] = (
    (PipelineDeadlineError, "deadline_exceeded"),
    (LLMRefusalError, "llm_refused"),
    (LLMTruncatedError, "llm_truncated"),
    (LLMSchemaError, "llm_schema"),
    (LLMRateLimitError, "llm_rate_limited"),
    (LLMProviderError, "llm_provider_error"),
)


def _classify_error(exc: BaseException) -> str:
    """Map an exception to the short error_code stored on the failed job row.

    A :class:`PipelineFailure` is classified by its cause. Uses the
    provider-agnostic LLM* hierarchy so this layer imports no vendor SDK.
    """
    cause = exc.error if isinstance(exc, PipelineFailure) else exc
    for error_class, code in _ERROR_CODES:
        if isinstance(cause, error_class):
            return code
    return "internal_error"


# ---------------------------------------------------------------------------
# Phase 1 -- load
# ---------------------------------------------------------------------------


async def _check_idempotent(
    conn: asyncpg.Connection,
    conversation_id: int,
    user_id: str,
    log: structlog.stdlib.AsyncBoundLogger,
) -> bool:
    """Return True when the conversation was already processed (caller short-circuits)."""
    already_processed = await conv_repo.get_processed_at(conn, conversation_id)
    if already_processed is not None:
        await log.info(
            "Conversation already processed, skipping",
            user_id=user_id,
            conversation_id=conversation_id,
        )
        return True
    return False


async def _start_job(conn: asyncpg.Connection, job: _Job, fallback: _Bucket) -> _Bucket | None:
    """Claim the job row and return the bucket its pre-charge used; None when not claimed.

    An unclaimed row is owned or already finished by another execution of
    the same job, and that execution (or the orphan sweep) settles the
    pre-charge, so this one neither calls the model nor touches the budget.
    A revived failed row was settled when it failed, so nothing is still
    pre-charged for it. A job row whose period cannot be read leaves the
    bucket unknown, so a later refund is skipped rather than landing in the
    wrong period.
    """
    if job.job_uuid is None:
        return fallback
    claimed_from = await extraction_jobs.mark_running(conn, job.job_uuid)
    if claimed_from is None:
        await job.log.info(
            "extraction_job_not_claimed",
            conversation_id=job.conversation_id,
            job_id=str(job.job_uuid),
        )
        return None
    pre_charged = 0 if claimed_from == "failed" else PRE_CHARGE_CENTS
    period_start = await extraction_jobs.get_period_start(conn, job.job_uuid)
    if period_start is None:
        return _Bucket(
            period_start=fallback.period_start, known=False, pre_charged_cents=pre_charged
        )
    return _Bucket(period_start=period_start, known=True, pre_charged_cents=pre_charged)


async def _load_conversation(conn: asyncpg.Connection, job: _Job) -> _Loaded:
    """Load the role-filtered transcript, the topic catalog and the extraction state."""
    try:
        _meta, messages, _total = await conv_repo.read_conversation_by_id(
            conn, job.cipher, job.conversation_id
        )
        state = await conv_repo.get_extraction_state(conn, job.conversation_id)
        if state is None:
            raise ConversationNotFoundError(f"Conversation id {job.conversation_id} not found")
    except Exception:
        await job.log.error(
            "Failed to load conversation",
            user_id=str(job.user_uuid),
            conversation_id=job.conversation_id,
            exc_info=True,
        )
        raise
    existing_topic_metas, _ = await topic_repo.list_all(conn)
    return _Loaded(
        messages=tuple(keep_conversation_roles(messages)),
        existing_topics=tuple(t.topic for t in existing_topic_metas),
        state=state,
    )


# ---------------------------------------------------------------------------
# Phase 2 -- LLM pipeline and embeddings (no connection held)
# ---------------------------------------------------------------------------


def _spend_of(ctx: ExtractionContext, usage: LLMUsage, *, llm_calls: int, chunks: int) -> _Spend:
    cents = ctx["extraction_service"].estimate_cost_cents(usage)
    return _Spend(usage=usage, llm_calls=llm_calls, chunks=chunks, cents=cents)


async def _extract(ctx: ExtractionContext, loaded: _Loaded, started: float) -> _Extracted:
    """Run the extraction pipeline and harden the topic it chose."""
    settings = get_settings()
    outcome = await run_pipeline(
        ctx["extraction_service"],
        loaded.messages,
        existing_topics=loaded.existing_topics,
        conversation_created=loaded.state.created_at.date(),
        deadline=started + EXTRACTION_JOB_DEADLINE_SECS,
        today=date.fromisoformat(local_today(settings.timezone)),
        mode=settings.llm.extraction_mode,
        triage_enabled=settings.llm.extraction_triage,
    )
    return _Extracted(
        outcome=outcome,
        topic_path=harden_llm_topic_path(outcome.topic_path) if outcome.topic_path else None,
        spend=_spend_of(ctx, outcome.usage, llm_calls=outcome.llm_calls, chunks=outcome.chunks),
    )


def _failure_spend(ctx: ExtractionContext, exc: Exception, spent: _Spend) -> _Spend:
    if isinstance(exc, PipelineFailure):
        return _spend_of(ctx, exc.usage_so_far, llm_calls=exc.llm_calls, chunks=exc.chunks)
    return spent


# ---------------------------------------------------------------------------
# Terminal states
# ---------------------------------------------------------------------------


async def _complete_without_entries(job: _Job, transcript_hash: str, spend: _Spend) -> None:
    """Mark the conversation processed and the job completed with nothing recorded.

    Uses its own short connection. Records the transcript hash so the same
    content is not sent to the model again at this extractor version.
    """
    async with user_scoped_connection(job.pool, user_id=job.user_uuid) as conn:
        await conv_repo.mark_processed(conn, job.conversation_id)
        await conv_repo.set_extraction_state(
            conn, job.conversation_id, transcript_hash, EXTRACTION_VERSION
        )
        if job.job_uuid is None:
            return
        updated = await extraction_jobs.mark_completed(
            conn,
            job.job_uuid,
            topics_created=0,
            entries_created=0,
            cents_spent=spend.cents,
            **_usage_kwargs(spend),
        )
        if updated:
            await record_audit(
                conn,
                action="extraction_job.completed",
                actor_type="user",
                actor_id=str(job.user_uuid),
                target_kind=TargetKind.EXTRACTION_JOB,
                target_id=str(job.job_uuid),
                metadata={"conversation_id": job.conversation_id, "entries_created": 0},
            )


def _usage_kwargs(spend: _Spend) -> dict[str, int]:
    return {
        "input_tokens": spend.usage.input,
        "output_tokens": spend.usage.output,
        "cached_input_tokens": spend.usage.cached_input,
        "reasoning_tokens": spend.usage.reasoning,
        "llm_calls": spend.llm_calls,
        "chunks": spend.chunks,
    }


async def _mark_job_failed(
    pool: asyncpg.Pool,
    user_uuid: UUID,
    job_id: str,
    conversation_id: int,
    error_code: str,
    spend: _Spend = _NO_SPEND,
) -> None:
    """Open a fresh user_scoped_connection to write the failed terminal state.

    Called from the outer except block AFTER the SAVEPOINT has already been
    poisoned (rolled back). Must NOT reuse any existing connection or be
    called inside an existing transaction context. ``spend`` covers every
    model call made before the failure.

    Does NOT re-raise -- callers must re-raise the original exception themselves.
    """
    try:
        async with user_scoped_connection(pool, user_id=user_uuid) as conn:
            updated = await extraction_jobs.mark_failed(
                conn,
                UUID(job_id),
                error_code=error_code,
                cents_spent=spend.cents,
                **_usage_kwargs(spend),
            )
            if updated:
                await record_audit(
                    conn,
                    action="extraction_job.failed",
                    actor_type="user",
                    actor_id=str(user_uuid),
                    target_kind=TargetKind.EXTRACTION_JOB,
                    target_id=str(job_id),
                    metadata={"error_code": error_code, "conversation_id": conversation_id},
                )
    except Exception as exc:
        # Swallow secondary failure -- the original exception is what Arq needs.
        # The job row may remain in 'running' until the orphan sweep fails it.
        #
        # A failing ``record_audit`` above can be a PostgreSQL rejection whose
        # message and DETAIL quote the audit row's actor id, IP and User-Agent,
        # so only bounded safe fields are logged, never the exception text.
        await logger.warning(
            "mark_job_failed_secondary_error",
            user_id=str(user_uuid),
            job_id=job_id,
            conversation_id=conversation_id,
            error_code=error_code,
            **safe_error_fields(exc),
        )


async def _settle_budget(
    ctx: ExtractionContext, job: _Job, bucket: _Bucket, actual_cents: int, event: str
) -> None:
    """Reconcile the ingest pre-charge against ``actual_cents``; best-effort.

    Skipped (with a metric) when the pre-charge bucket is unknown: refunding
    into the wrong period is worse than not refunding.
    """
    helper = ctx.get("budget_helper")
    if helper is None:
        return
    if not bucket.known:
        await logger.warning(
            "extraction.refund_skipped_unknown_period",
            user_id=str(job.user_uuid),
            conversation_id=job.conversation_id,
            job_id=str(job.job_uuid),
        )
        _get_extraction_refund_skipped_counter().add(1, attributes={"reason": "unknown_period"})
        return
    try:
        await helper.record_actual_cost(
            user_id=job.user_uuid,
            period_start=bucket.period_start,
            actual_cents=actual_cents,
            estimated_cents=bucket.pre_charged_cents,
        )
    except Exception:  # broad: redis errors come in many shapes
        await job.log.warning(event, exc_info=True)


async def _settle_and_mark_failed(
    ctx: ExtractionContext, job: _Job, bucket: _Bucket, spend: _Spend, error_code: str
) -> None:
    """Settle the pre-charge against ``spend``, then write the failed job state."""
    await _settle_budget(ctx, job, bucket, spend.cents, "budget_refund_failed")
    if job.job_uuid is not None:
        await _mark_job_failed(
            job.pool,
            job.user_uuid,
            str(job.job_uuid),
            job.conversation_id,
            error_code,
            spend,
        )


async def _fail_cancelled(
    ctx: ExtractionContext, job: _Job, bucket: _Bucket, spend: _Spend
) -> None:
    """Settle and fail a cancelled job, bounded so the cancellation is not held up.

    The cleanup is shielded: a second cancellation stops the wait, not the
    cleanup. One that outlasts the bound is cancelled and left to the orphan
    sweep.
    """
    cleanup = asyncio.create_task(
        _settle_and_mark_failed(ctx, job, bucket, spend, _CANCELLED_ERROR_CODE)
    )
    try:
        await asyncio.wait_for(asyncio.shield(cleanup), _CANCEL_CLEANUP_TIMEOUT_SECS)
    except TimeoutError:
        cleanup.cancel()
        await job.log.warning(
            "extraction_cancel_cleanup_timed_out", conversation_id=job.conversation_id
        )
    except asyncio.CancelledError:
        _DETACHED_CLEANUPS.add(cleanup)
        cleanup.add_done_callback(_DETACHED_CLEANUPS.discard)


async def _publish_progress(
    redis: Any,
    user_id: str,
    conversation_id: int,
    job_id: str,
    topic_path: str | None,
    entries_created: int,
    log: structlog.stdlib.AsyncBoundLogger,
) -> None:
    """Publish the extraction completion event on the user's Redis channel; non-fatal."""
    event = {
        "topic_path": topic_path,
        "entries_created": entries_created,
        "job_id": job_id,
        "conversation_id": conversation_id,
    }
    try:
        channel = f"extraction:user:{user_id}:job:{job_id}"
        await redis.publish(channel, json.dumps(event))
    except Exception:
        await log.warning(
            "Failed to publish extraction event to Redis",
            user_id=user_id,
            conversation_id=conversation_id,
            exc_info=True,
        )


# ---------------------------------------------------------------------------
# Phase 3 -- persist
# ---------------------------------------------------------------------------


async def _persist(
    ctx: ExtractionContext,
    job: _Job,
    loaded: _Loaded,
    extracted: _Extracted,
    topic_path: str,
    embeddings: Sequence[Sequence[float]],
) -> int | None:
    """Write the run under a SAVEPOINT; None when a concurrent worker already did.

    The conversation row is locked before its processed_at is read, so of two
    workers racing here exactly one sees it unprocessed and writes.
    """
    run = ExtractionToPersist(
        user_id=job.user_uuid,
        conversation_id=job.conversation_id,
        job_id=job.job_uuid,
        extraction_attempt_id=job.extraction_attempt_id,
        client=loaded.state.platform,
        topic_path=topic_path,
        topic_title=extracted.outcome.topic_title,
        candidates=extracted.outcome.candidates,
        embeddings=embeddings,
        content_hash=content_hash(loaded.messages),
        usage=extracted.spend.usage,
        llm_calls=extracted.spend.llm_calls,
        chunks=extracted.spend.chunks,
        cents_spent=extracted.spend.cents,
    )
    async with user_scoped_connection(job.pool, user_id=job.user_uuid) as conn2:  # noqa: SIM117
        async with conn2.transaction():
            if await conv_repo.lock_processed_at(conn2, job.conversation_id) is not None:
                await job.log.info(
                    "Conversation processed by a concurrent worker, skipping",
                    user_id=str(job.user_uuid),
                    conversation_id=job.conversation_id,
                )
                return None
            persisted = await persist_extraction(conn2, job.cipher, ctx["embedding_service"], run)
    return len(persisted.entry_ids)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def extract_conversation(
    ctx: ExtractionContext,
    conversation_id: int,
    user_id: str,
    job_id: str = "unknown",
) -> dict[str, Any]:
    """Arq job: extract a conversation's topic and dated entries and persist them.

    CALLER CONTRACT (security-critical):
    The caller MUST authenticate that ``user_id`` owns ``conversation_id``
    BEFORE enqueueing this job. Today this is the ingest endpoint -- it
    derives user_id from the authenticated request and only enqueues against
    conversations that belong to that user.

    The job runs under user_scoped_connection(user_id=...), so all reads
    and writes are RLS-protected. A wrong user_id at enqueue time reads zero
    rows and the load fails. Never run it through the admin pool.

    Flow: idempotency check; role-filtered load; rule-based skips (no text,
    trivial user text, unchanged content at this extractor version); the
    extraction pipeline (chunked, optional triage, single or two-call mode);
    embeddings; persistence with dedupe. A conversation with nothing worth
    recording completes with ``entries_created=0`` and creates no topic.

    Lifecycle UPDATEs:
      mark_running   -- Phase 1, after the idempotency check passes. A row it
                        cannot claim belongs to another execution of the job,
                        so this one returns without a model call.
      mark_completed -- inside the Phase 3 SAVEPOINT, or on a short dedicated
                        connection when nothing is recorded.
      mark_failed    -- on a FRESH connection in the outer except block, with
                        the usage of every model call made before the failure;
                        on cancellation (Arq's timeout) with ``deadline_exceeded``.

    Budget: the ingest pre-charge is settled against the estimated cost of
    every model call the job made, whatever the outcome.

    Args:
        ctx: Arq worker context (pool, cipher, extraction_service,
            embedding_service, redis injected by on_startup).
        conversation_id: Database integer ID of the conversation to process.
        user_id: UUID string of the owning user (RLS scoping and pub/sub channel).
        job_id: UUID string of the extraction_jobs row created by ingest, passed
            as the 4th positional argument by the ingest enqueue. ``_job_id=`` is
            arq-internal and is NOT forwarded here. Defaults to "unknown", in
            which case no job row is transitioned.

    Returns:
        Summary dict with topic_path, entries_created, input_tokens,
        output_tokens, cents_spent, and skipped (True when nothing was
        persisted: already processed, a rule skip, or nothing to record).
    """
    user_uuid = user_id if isinstance(user_id, UUID) else UUID(user_id)
    extraction_attempt_id = str(uuid4())
    job = _Job(
        pool=ctx["pool"],
        cipher=cast("ContentCipher", ctx["cipher"]),
        user_uuid=user_uuid,
        conversation_id=conversation_id,
        job_uuid=None if job_id == "unknown" else UUID(job_id),
        extraction_attempt_id=extraction_attempt_id,
        log=cast(
            "structlog.stdlib.AsyncBoundLogger",
            logger.bind(
                component="extract_conversation", extraction_attempt_id=extraction_attempt_id
            ),
        ),
    )

    span_name = SpanNames.EXTRACTION_JOB
    span_start_ns = time.monotonic_ns()
    success = True
    failure_reason: str | None = None

    with trace.get_tracer(_TRACER_NAME).start_as_current_span(
        span_name,
        record_exception=False,
        set_status_on_exception=False,
    ) as job_span:
        safe_set_attributes(
            span_name,
            job_span,
            {
                "job_id": str(job_id),
                "user_id": str(user_id),
                "conversation_id": int(conversation_id),
            },
        )
        try:
            return await _run_job(ctx, job, str(user_id), str(job_id))
        except asyncio.CancelledError:
            success = False
            failure_reason = _CANCELLED_ERROR_CODE
            raise
        except Exception as exc:
            success = False
            failure_reason = _classify_error(exc)
            # A failing audit write inside this job propagates here, and a
            # PostgreSQL rejection's message and DETAIL quote the audit row's
            # actor id, IP and User-Agent. Sanitized for that case, SDK-default
            # for anything this codebase raises itself. The span's two
            # auto-record flags are off so the context exit cannot re-add the
            # raw event; the bare raise still ends the span and propagates.
            record_exception_sanitized(job_span, exc)
            raise
        finally:
            latency_ms = (time.monotonic_ns() - span_start_ns) / _NS_PER_MS
            final_attrs: dict[str, Any] = {
                "success": success,
                "latency_ms": round(latency_ms, 2),
            }
            if failure_reason is not None:
                final_attrs["failure_reason"] = failure_reason
            safe_set_attributes(span_name, job_span, final_attrs)


async def _run_job(ctx: ExtractionContext, job: _Job, user_id: str, job_id: str) -> dict[str, Any]:
    started = time.monotonic()
    # Until the job row's period is read, the runtime period is only a guess at
    # the bucket ingest pre-charged, so a refund is allowed only without a job row.
    bucket = _Bucket(period_start=current_period_start(), known=job.job_uuid is None)
    spent = _NO_SPEND
    try:
        async with user_scoped_connection(job.pool, user_id=job.user_uuid) as conn1:
            if await _check_idempotent(conn1, job.conversation_id, user_id, job.log):
                return _result(None, 0, _NO_SPEND, skipped=True)
            claimed = await _start_job(conn1, job, bucket)
            if claimed is None:
                return _result(None, 0, _NO_SPEND, skipped=True)
            bucket = claimed
            loaded = await _load_conversation(conn1, job)
        # conn1 released here -- pool slot returned before any LLM call.
        if await _skip_by_rule(ctx, job, loaded, bucket):
            return _result(None, 0, _NO_SPEND, skipped=True)
        extracted = await _extract(ctx, loaded, started)
        spent = extracted.spend
        result = await _record(ctx, job, loaded, extracted, bucket)
    except asyncio.CancelledError:
        await _fail_cancelled(ctx, job, bucket, spent)
        raise
    except Exception as exc:
        # The SAVEPOINT (if any) is already rolled back. Settle first, then
        # write the failed state on a FRESH connection; the two side effects
        # are independent so neither blocks the other.
        await _settle_and_mark_failed(
            ctx, job, bucket, _failure_spend(ctx, exc, spent), _classify_error(exc)
        )
        raise
    if not result["skipped"]:
        await _publish_progress(
            ctx["redis"],
            user_id,
            job.conversation_id,
            job_id,
            result["topic_path"],
            result["entries_created"],
            job.log,
        )
        await job.log.info(
            "Extraction complete",
            user_id=user_id,
            conversation_id=job.conversation_id,
            topic_path=result["topic_path"],
            entries_created=result["entries_created"],
        )
    return result


async def _skip_by_rule(
    ctx: ExtractionContext, job: _Job, loaded: _Loaded, bucket: _Bucket
) -> bool:
    """Complete the job without a model call when a rule says there is nothing to extract."""
    skip_reason = evaluate_skip(
        loaded.messages,
        loaded.state.source,
        loaded.state.extracted_content_hash,
        loaded.state.extraction_version,
    )
    if skip_reason is None:
        return False
    await job.log.info(
        "extraction_skipped", conversation_id=job.conversation_id, reason=str(skip_reason)
    )
    await _complete_without_entries(job, content_hash(loaded.messages), _NO_SPEND)
    await _settle_budget(ctx, job, bucket, 0, "budget_refund_failed")
    return True


async def _record(
    ctx: ExtractionContext, job: _Job, loaded: _Loaded, extracted: _Extracted, bucket: _Bucket
) -> dict[str, Any]:
    """Persist what the pipeline found, or complete the job with nothing to record."""
    spent = extracted.spend
    if extracted.topic_path is None or not extracted.outcome.candidates:
        await job.log.info(
            "extraction_nothing_to_record",
            conversation_id=job.conversation_id,
            topic_usable=extracted.topic_path is not None,
        )
        await _complete_without_entries(job, content_hash(loaded.messages), spent)
        await _settle_budget(ctx, job, bucket, spent.cents, "budget_refund_failed")
        return _result(None, 0, spent, skipped=True)
    embeddings = await embed_texts(
        ctx["embedding_service"], [candidate.content for candidate in extracted.outcome.candidates]
    )
    entries_created = await _persist(ctx, job, loaded, extracted, extracted.topic_path, embeddings)
    if entries_created is None:
        await _settle_budget(ctx, job, bucket, spent.cents, "budget_delta_failed")
        return _result(None, 0, spent, skipped=True)
    # Best-effort and after commit: a Redis failure never rolls back the run.
    await _settle_budget(ctx, job, bucket, spent.cents, "budget_delta_failed")
    return _result(extracted.topic_path, entries_created, spent, skipped=False)
