"""Arq job: extract structured topics and entries from a saved conversation.

See ``extract_conversation`` function docstring for the CALLER CONTRACT
(security-critical).

Connection-split design (m-h5-h6):
  Phase 1 -- conn1: read-only load + early idempotency check.
  Phase 2 -- LLM phase: NO database connection held.
  Phase 3 -- conn2: persistence under an explicit nested SAVEPOINT transaction.

This ensures the database connection is released during the LLM calls
(which can take several seconds), so the pool is not exhausted when
max_jobs concurrent workers are all mid-LLM.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any, cast
from uuid import UUID, uuid4

import asyncpg
import structlog
from gubbi_common.audit.actions import Action
from gubbi_common.audit.targets import TargetKind
from gubbi_common.budget import PRE_CHARGE_CENTS, current_period_start
from gubbi_common.db.user_scoped import user_scoped_connection
from opentelemetry import metrics

from gubbi.audit import record_audit
from gubbi.crypto.cipher import ContentCipher
from gubbi.extraction.context import ExtractionContext
from gubbi.extraction.llm.provider import (
    LLMMessage,
    LLMProviderError,
    LLMRateLimitError,
)
from gubbi.extraction.service import (
    CategorizationResult,
    ExtractedEntry,
    ExtractionEntriesResult,
    ExtractionService,
)
from gubbi.storage.exceptions import TopicNotFoundError
from gubbi.storage.repositories import conversations as conv_repo
from gubbi.storage.repositories import entries as entry_repo
from gubbi.storage.repositories import extraction_jobs
from gubbi.storage.repositories import topics as topic_repo
from gubbi.validation import harden_llm_topic_path

__all__: list[str] = ["EXTRACTION_REFUND_SKIPPED", "extract_conversation"]

logger = structlog.get_logger(__name__)

# Counter incremented when the worker fails BEFORE the period_start lookup
# completes. In that window we cannot guarantee the runtime period equals the
# bucket the pre-charge debited (a period boundary may have been crossed since
# ingest), so the refund is skipped. Manual reconcile signal -- see backlog
# item from R1 / MEDIUM-2.
_meter = metrics.get_meter("gubbi")
EXTRACTION_REFUND_SKIPPED = _meter.create_counter(
    name="extraction.refund_skipped_total",
    description=(
        "Pre-charge refund skipped because the worker failed before the "
        "pre-charge period was known; partitioned by reason."
    ),
    unit="1",
)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


async def _check_idempotent(
    conn: asyncpg.Connection,
    conversation_id: int,
    user_id: str,
    log: structlog.stdlib.AsyncBoundLogger,
) -> bool:
    """Check whether this conversation was already processed.

    Returns True if *already* processed (caller short-circuits with skip
    result).  Returns False otherwise so processing can proceed.
    """
    already_processed = await conv_repo.get_processed_at(conn, conversation_id)
    if already_processed is not None:
        await log.info(
            "Conversation already processed, skipping",
            user_id=user_id,
            conversation_id=conversation_id,
        )
        return True
    return False


async def _load_conversation_for_extraction(
    conn: asyncpg.Connection,
    cipher: ContentCipher,
    conversation_id: int,
    user_id: str,
    log: structlog.stdlib.AsyncBoundLogger,
) -> tuple[Any, list[LLMMessage], list[str]]:
    """Load a conversation + its messages and existing topics.

    Returns ``(meta, message_dicts, existing_topics)`` where
    *message_dicts* is the list-of-dicts serialization needed by the LLM
    service, and *existing_topics* contains the raw topic strings.

    On load failure logs an error with extras and re-raises.
    """
    try:
        meta, messages, _total = await conv_repo.read_conversation_by_id(
            conn, cipher, conversation_id
        )
    except Exception:
        await log.error(
            "Failed to load conversation",
            user_id=user_id,
            conversation_id=conversation_id,
            exc_info=True,
        )
        raise

    existing_topic_metas, _ = await topic_repo.list_all(conn)
    existing_topics = [t.topic for t in existing_topic_metas]

    message_dicts: list[LLMMessage] = [{"role": m.role, "content": m.content} for m in messages]

    return meta, message_dicts, existing_topics


async def _categorize_and_resolve_topic(
    extraction_service: ExtractionService,
    message_dicts: list[LLMMessage],
    existing_topics: list[str],
    user_id: str,
    conversation_id: int,
    log: structlog.stdlib.AsyncBoundLogger,
) -> tuple[CategorizationResult, str | None]:
    """Categorise the conversation and produce a hardened topic path.

    Returns ``(categorization, topic_path)`` -- *topic_path* may be None if
    hardening rejects the LLM output (non-fatal skip signal).

    On categorization failure logs an error with extras and re-raises.
    """
    try:
        categorization = await extraction_service.categorize_conversation(
            message_dicts, existing_topics
        )
    except Exception:
        await log.error(
            "Categorization failed",
            user_id=user_id,
            conversation_id=conversation_id,
            exc_info=True,
        )
        raise

    raw_topic_path: str | None = categorization.topic_path
    topic_path: str | None = harden_llm_topic_path(raw_topic_path)

    if topic_path is None:
        await log.warning(
            "extraction: no usable topic_path from LLM, returning early",
            raw_topic_path_chars=len(raw_topic_path) if raw_topic_path else 0,
        )

    return categorization, topic_path


async def _mark_skipped_no_topic(
    pool: asyncpg.Pool,
    user_uuid: UUID,
    conversation_id: int,
    job_id: str,
) -> None:
    """Open a short-lived connection to mark a conversation as processed (no-topic path).

    Called when the LLM categorization returns a topic_path that hardening rejects.
    Uses a dedicated connection so the no-topic skip has exactly the same persistence
    contract as the full success path.
    """
    async with user_scoped_connection(pool, user_id=user_uuid) as conn:
        await conv_repo.mark_processed(conn, conversation_id)
        if job_id != "unknown":
            updated = await extraction_jobs.mark_completed(
                conn,
                UUID(job_id),
                topics_created=0,
                entries_created=0,
                cents_spent=0,
            )
            if updated:
                await record_audit(
                    conn,
                    action="extraction_job.completed",
                    actor_type="user",
                    actor_id=str(user_uuid),
                    target_kind=TargetKind.EXTRACTION_JOB,
                    target_id=job_id,
                    metadata={"conversation_id": conversation_id, "entries_created": 0},
                )


def _classify_error(exc: BaseException) -> str:
    """Map an exception to a short error_code string for extraction_jobs.error_code.

    Uses the provider-agnostic LLM* hierarchy so this layer does not import
    vendor SDKs. Anthropic-specific exceptions are translated to LLM* at the
    AnthropicProvider boundary (see gubbi.extraction.llm.anthropic_provider).
    """
    if isinstance(exc, LLMRateLimitError):
        return "llm_rate_limited"
    if isinstance(exc, LLMProviderError):
        return "llm_provider_error"
    return "internal_error"


async def _mark_job_failed(
    pool: asyncpg.Pool,
    user_uuid: UUID,
    job_id: str,
    conversation_id: int,
    error_code: str,
) -> None:
    """Open a fresh user_scoped_connection to write the failed terminal state.

    Called from the outer except block AFTER the SAVEPOINT has already been
    poisoned (rolled back). Must NOT reuse any existing connection or be
    called inside an existing transaction context.

    Mirrors the _mark_skipped_no_topic pattern: short dedicated connection,
    do the work, release.

    Does NOT re-raise -- callers must re-raise the original exception themselves.
    """
    try:
        async with user_scoped_connection(pool, user_id=user_uuid) as conn:
            updated = await extraction_jobs.mark_failed(conn, UUID(job_id), error_code=error_code)
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
    except Exception:  # noqa: BLE001
        # Swallow secondary failure -- the original exception is what Arq needs.
        # The job row may remain in 'running' and will be cleaned up by a
        # future monitor / TTL sweep.
        await logger.warning(
            "mark_job_failed_secondary_error",
            user_id=str(user_uuid),
            job_id=job_id,
            conversation_id=conversation_id,
            error_code=error_code,
            exc_info=True,
        )


async def _persist_extraction(
    conn: asyncpg.Connection,
    cipher: ContentCipher,
    user_uuid: UUID,
    conversation_id: int,
    topic_path: str,
    categorization: CategorizationResult,
    extracted_entries: tuple[ExtractedEntry, ...],
    extraction_attempt_id: str,
    job_id: str,
    cents_spent: int,
    log: structlog.stdlib.AsyncBoundLogger,
) -> int:
    """Upsert the topic (if needed) and persist pre-extracted entries.

    All writes happen inside the SAVEPOINT that the caller wraps around this
    function. No LLM calls here -- entries are pre-extracted before conn2 is
    acquired.

    Returns the count of persisted entries.

    NOTE-m-h5-h6: entry_repo.append uses a plain INSERT with no ON CONFLICT
    clause.  The entries table has no uniqueness constraint on
    (conversation_id, topic_id, content) that would make ON CONFLICT DO NOTHING
    meaningful without a schema migration.  The conn2 second-idempotency-check
    (caller) prevents double-insert under normal retry storms.  A partial conn2
    failure mid-batch (e.g. after some entries are written but before
    mark_processed) will roll back via the SAVEPOINT, so partial state is
    never committed.  Retries after that are safe: the SAVEPOINT re-runs from
    the start of Phase 3.
    """
    # Topic upsert (get_id / create with race-tolerance).
    topic_created = False
    try:
        await topic_repo.get_id(conn, topic_path)
    except TopicNotFoundError:
        topic_created = True
        try:
            await topic_repo.create(conn, topic_path, title=categorization.topic_title)
        except ValueError as exc:
            if "already exists" in str(exc):
                topic_created = False
                await log.debug("Topic race on create, proceeding", error=str(exc))
            else:
                raise

    # Persist entries (append loop -- no LLM calls here).
    entries_created = 0
    for entry in extracted_entries:
        await entry_repo.append(
            conn,
            cipher,
            topic=topic_path,
            content=entry.content,
            reasoning=entry.reasoning,
            tags=entry.tags,
            date=entry.entry_date,
        )
        entries_created += 1

    # Mark processed (WHERE processed_at IS NULL -- race-safe no-op if already set).
    await conv_repo.mark_processed(conn, conversation_id)

    # Audit row -- inside SAVEPOINT; rolled back on any failure above.
    await record_audit(
        conn,
        actor_type="user",
        actor_id=str(user_uuid),
        action=Action.CONVERSATION_EXTRACTED,
        target_kind=TargetKind.CONVERSATION,
        target_id=str(conversation_id),
        metadata={
            "via": "extraction-worker",
            "entries_created": entries_created,
            "topics_touched": 1,
            "extraction_attempt_id": extraction_attempt_id,
        },
    )

    # Lifecycle terminal: mark job completed with final counters.
    # Skip when job_id is the sentinel 'unknown' (tests without a real job row).
    if job_id != "unknown":
        topics_created_count = 1 if topic_created else 0
        updated = await extraction_jobs.mark_completed(
            conn,
            UUID(job_id),
            topics_created=topics_created_count,
            entries_created=entries_created,
            cents_spent=cents_spent,
        )
        if updated:
            await record_audit(
                conn,
                action="extraction_job.completed",
                actor_type="user",
                actor_id=str(user_uuid),
                target_kind=TargetKind.EXTRACTION_JOB,
                target_id=str(job_id),
                metadata={"conversation_id": conversation_id, "entries_created": entries_created},
            )

    return entries_created


async def _publish_progress(
    redis: Any,
    user_id: str,
    conversation_id: int,
    job_id: str,
    topic_path: str | None,
    entries_created: int,
    log: structlog.stdlib.AsyncBoundLogger,
) -> None:
    """Publish extraction completion event on the user's Redis channel.

    Redis publish failure is non-fatal: logs a warning and returns silently.
    """
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
# Public entry point
# ---------------------------------------------------------------------------


async def extract_conversation(
    ctx: ExtractionContext,
    conversation_id: int,
    user_id: str,
    job_id: str = "unknown",
) -> dict[str, Any]:
    """Arq job: categorize a conversation and write structured journal entries.

    NOTE: conversation_id is ``int`` (DB integer primary key) even though the
    original spec said ``UUID`` -- the conversations table uses integer PKs.

    CALLER CONTRACT (security-critical):
    The caller MUST authenticate that ``user_id`` owns ``conversation_id``
    BEFORE enqueueing this job. Today this is the API endpoint that
    enqueues extraction (POST /api/v1/extraction) -- it derives user_id
    from the authenticated request and only enqueues against
    conversations that belong to that user (verified via RLS-scoped
    SELECT).

    The job runs under user_scoped_connection(user_id=...), so all reads
    and writes are RLS-protected. A wrong user_id at enqueue time will:
      - Read the wrong tenant's conversation (RLS blocks; results in 0
        rows; the job loads an empty conversation and returns).
      - Or, if a future caller bypasses the API path with admin pool:
        could read across tenants. Don't bypass the API path.

    See llm_context/audit_contract.md for actor_type semantics on the
    summary audit row this job produces.

    Connection-split design (m-h5-h6):
      Phase 1 -- conn1: read-only load + early idempotency check.
      Phase 2 -- LLM phase: no database connection held.
      Phase 3 -- conn2: persistence under explicit nested SAVEPOINT.

    Lifecycle UPDATEs (Part 3):
      mark_running  -- called in Phase 1 after idempotency check passes.
      mark_completed -- called inside the Phase 3 SAVEPOINT via _persist_extraction.
      mark_failed   -- called on a FRESH connection in the outer except block,
                       AFTER the SAVEPOINT is already poisoned.

    Args:
        ctx: Arq worker context (pool, cipher, extraction_service, redis
            injected by on_startup).
        conversation_id: Database integer ID of the conversation to process.
        user_id: UUID string of the owning user (used for RLS scoping and
            pub/sub channel).
        job_id: UUID string of the extraction_jobs row created by ingest.
            Arq passes the value set via _job_id at enqueue time.  Defaults
            to "unknown" so the function remains callable from tests that
            pre-date the Part 3 signature change.

    Returns:
        Summary dict with topic_path, entries_created, input_tokens,
        output_tokens, cents_spent, and skipped (bool, True if idempotency
        check short-circuited).
    """
    pool = ctx["pool"]
    cipher = cast(ContentCipher, ctx["cipher"])
    extraction_service = ctx["extraction_service"]
    redis = ctx["redis"]

    # Bind an attempt-level correlation ID for trace correlation across the two DB spans.
    extraction_attempt_id = str(uuid4())

    log = cast(
        "structlog.stdlib.AsyncBoundLogger",
        logger.bind(
            component="extract_conversation",
            extraction_attempt_id=extraction_attempt_id,
        ),
    )
    user_uuid = user_id if isinstance(user_id, UUID) else UUID(user_id)

    # Pre-bind effective_period_start so the outer except block's refund call
    # always has a defined value, even when an exception raises before the
    # job-row lookup completes (e.g. asyncpg connection failure during conn1).
    # Phase 1 may overwrite this with the job row's period_start once known.
    #
    # _period_start_known gates the outer-except refund (R1 / MEDIUM-2):
    # only after the job row's period_start has been read (or there is no
    # real pre-charge to reconcile, i.e. job_id == "unknown") may the refund
    # be issued. Otherwise we cannot guarantee the runtime period matches
    # the bucket that ingest debited; a period boundary crossed mid-job
    # would refund the wrong bucket. Skipping (with a metric) is safer than
    # mis-bucketing.
    effective_period_start: date = current_period_start()
    _period_start_known: bool = job_id == "unknown"

    try:
        # ------------------------------------------------------------------
        # Phase 1 -- conn1: read-only load + early idempotency check.
        # conn1 is released before any LLM call.
        # ------------------------------------------------------------------
        # period_start loaded from the job row so the budget delta in Phase 3
        # lands in the same billing bucket that ingest pre-charged (B3-H2).
        # Falls back to current_period_start() when job_id is 'unknown' (tests)
        # or the row is not visible under RLS.
        job_period_start: date | None = None
        async with user_scoped_connection(pool, user_id=user_uuid) as conn1:
            if await _check_idempotent(conn1, conversation_id, user_id, log):
                return {
                    "topic_path": None,
                    "entries_created": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cents_spent": 0,
                    "skipped": True,
                }

            # Lifecycle: transition pending -> running.
            if job_id != "unknown":
                job_uuid_for_phase1 = UUID(job_id)
                await extraction_jobs.mark_running(conn1, job_uuid_for_phase1)
                job_period_start = await extraction_jobs.get_period_start(
                    conn1, job_uuid_for_phase1
                )
                # Mark the period_start authoritative for refund routing only
                # when the job row actually yielded a bucket. If the row is
                # missing under RLS / invalid job_id and get_period_start()
                # returns None, refunding against the runtime period would
                # re-introduce the wrong-bucket bug across a month boundary.
                _period_start_known = job_period_start is not None

            _meta, message_dicts, existing_topics = await _load_conversation_for_extraction(
                conn1, cipher, conversation_id, user_id, log
            )
        # conn1 released here -- pool slot returned before LLM calls.

        # Resolve period_start: prefer the DB value; fall back to runtime.
        effective_period_start = job_period_start or effective_period_start

        # ------------------------------------------------------------------
        # Phase 2 -- LLM phase: no database connection held.
        # ------------------------------------------------------------------
        categorization, topic_path = await _categorize_and_resolve_topic(
            extraction_service, message_dicts, existing_topics, user_id, conversation_id, log
        )

        if topic_path is None:
            # No usable topic -- mark processed via a short dedicated connection.
            await _mark_skipped_no_topic(pool, user_uuid, conversation_id, job_id)
            return {
                "topic_path": None,
                "entries_created": 0,
                "input_tokens": categorization.input_tokens,
                "output_tokens": categorization.output_tokens,
                "cents_spent": 0,
                "skipped": True,
            }

        # Extract entries while no DB connection is held.
        extraction_result: ExtractionEntriesResult
        try:
            extraction_result = await extraction_service.extract_entries(message_dicts, topic_path)
        except Exception:
            await log.error(
                "Entry extraction failed",
                user_id=user_id,
                conversation_id=conversation_id,
                exc_info=True,
            )
            raise

        extracted_entries = extraction_result.entries

        # Accumulate token counts across both LLM calls and compute cost.
        total_input_tokens = categorization.input_tokens + extraction_result.input_tokens
        total_output_tokens = categorization.output_tokens + extraction_result.output_tokens
        llm_provider = getattr(extraction_service, "_llm", None)
        cents_spent = 0
        if llm_provider is not None and callable(
            getattr(llm_provider, "estimate_cost_cents", None)
        ):
            try:
                raw_cost = llm_provider.estimate_cost_cents(total_input_tokens, total_output_tokens)
                # Guard against async mock returning a coroutine in tests.
                if isinstance(raw_cost, int | float):
                    cents_spent = int(round(raw_cost))
            except Exception:  # noqa: BLE001
                await log.warning("cost_estimation_failed", exc_info=True)
                cents_spent = 0

        # ------------------------------------------------------------------
        # Phase 3 -- conn2: persistence under explicit nested SAVEPOINT.
        # Second idempotency check here guards against a concurrent worker
        # that raced through Phase 2 while we were doing LLM calls.
        # ------------------------------------------------------------------
        async with user_scoped_connection(pool, user_id=user_uuid) as conn2:  # noqa: SIM117
            async with conn2.transaction():  # nested SAVEPOINT inside user_scoped_connection
                # Last-write-wins guard: if another worker committed first, bail.
                if await _check_idempotent(conn2, conversation_id, user_id, log):
                    return {
                        "topic_path": None,
                        "entries_created": 0,
                        "input_tokens": total_input_tokens,
                        "output_tokens": total_output_tokens,
                        "cents_spent": cents_spent,
                        "skipped": True,
                    }

                entries_created = await _persist_extraction(
                    conn2,
                    cipher,
                    user_uuid,
                    conversation_id,
                    topic_path,
                    categorization,
                    extracted_entries,
                    extraction_attempt_id,
                    job_id,
                    cents_spent,
                    log,
                )

                # Best-effort budget delta write. Must NOT affect the SAVEPOINT:
                # extraction succeeded and the row is committed; the pre-charge
                # already protected the cap. Log + continue on any Redis failure.
                # Uses effective_period_start (loaded from job row in Phase 1)
                # to ensure the delta lands in the same bucket as the pre-charge.
                helper = ctx.get("budget_helper")
                if helper is not None:
                    try:
                        await helper.record_actual_cost(
                            user_id=user_uuid,
                            period_start=effective_period_start,
                            actual_cents=cents_spent,
                            estimated_cents=PRE_CHARGE_CENTS,
                        )
                    except Exception:  # broad: redis errors come in many shapes
                        await log.warning("budget_delta_failed", exc_info=True)
        # conn2 released here -- SAVEPOINT committed atomically.

    except Exception as exc:
        # SAVEPOINT is already poisoned (rolled back). Open a FRESH connection
        # to record the failure terminal state -- do NOT reuse conn1 or conn2.
        # Pre-charge refund first: actual=0 against the original PRE_CHARGE_CENTS
        # estimate produces a negative delta, returning the budget the worker
        # never spent. Best-effort: a Redis failure here must not block the
        # extraction_jobs row update that follows. The two side effects are
        # independent so the operator never loses one because the other failed.
        #
        # Refund is GATED by _period_start_known (R1 / MEDIUM-2): if the
        # worker failed before the job row's period_start was read, we cannot
        # guarantee the runtime period matches the bucket the pre-charge
        # debited (a period boundary may have been crossed). In that window
        # the refund is skipped and a metric is emitted so an operator can
        # manually reconcile. Refunding into the wrong bucket is worse than
        # not refunding -- the user gets a credit they shouldn't AND has a
        # phantom debit in the original period.
        helper = ctx.get("budget_helper")
        if helper is not None:
            if not _period_start_known:
                await logger.warning(
                    "extraction.refund_skipped_unknown_period",
                    user_id=str(user_uuid),
                    conversation_id=conversation_id,
                    job_id=job_id,
                )
                EXTRACTION_REFUND_SKIPPED.add(
                    1,
                    attributes={"reason": "unknown_period"},
                )
            else:
                try:
                    await helper.record_actual_cost(
                        user_id=user_uuid,
                        period_start=effective_period_start,
                        actual_cents=0,
                        estimated_cents=PRE_CHARGE_CENTS,
                    )
                except Exception:  # broad: redis errors come in many shapes
                    await log.warning("budget_refund_failed", exc_info=True)
        if job_id != "unknown":
            error_code = _classify_error(exc)
            await _mark_job_failed(pool, user_uuid, job_id, conversation_id, error_code)
        raise

    # ------------------------------------------------------------------
    # Publish progress event (outside DB connection -- non-fatal).
    # ------------------------------------------------------------------
    await _publish_progress(
        redis, user_id, conversation_id, str(job_id), topic_path, entries_created, log
    )

    await log.info(
        "Extraction complete",
        user_id=user_id,
        conversation_id=conversation_id,
        topic_path=topic_path,
        entries_created=entries_created,
    )

    return {
        "topic_path": topic_path,
        "entries_created": entries_created,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "cents_spent": cents_spent,
        "skipped": False,
    }
