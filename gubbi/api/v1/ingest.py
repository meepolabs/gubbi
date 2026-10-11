"""REST API: POST /api/v1/ingest/conversations.

Accepts normalized conversation batches from the browser extension.
Transforms, dedupes, saves to the conversations table, and records one
pending extraction job per saved conversation with its scheduling lane, run
time and filtered token count. No LLM calls and no job dispatch: the
extraction scheduler admits and runs queued jobs.
"""

from __future__ import annotations

import contextlib
from dataclasses import asdict, dataclass

# datetime is a Pydantic field type on MessagePayload / ConversationPayload;
# UUID is a FastAPI Depends() Annotated type. Both are resolved at runtime
# (validator-build / route-registration), so they must stay runtime imports
# despite `from __future__ import annotations` -- ruff TC would otherwise
# push them under TYPE_CHECKING and break Pydantic model_rebuild().
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Annotated, Literal
from uuid import UUID  # noqa: TC003

import asyncpg
import structlog
from fastapi import APIRouter, Depends, Request
from gubbi_common.audit.targets import TargetKind
from gubbi_common.telemetry import bound_logger
from pydantic import BaseModel, Field

from gubbi.api.v1.auth import require_scope
from gubbi.app_state import require_app_ctx
from gubbi.audit.sql import record_audit
from gubbi.config import get_settings
from gubbi.crypto.guard import require_cipher
from gubbi.extraction.admission import (
    budget_period,
    monthly_allowance_exhausted,
    user_usage_from_totals,
)
from gubbi.extraction.intake import (
    TOKEN_COUNT_VERSION,
    assign_lane,
    count_filtered_tokens,
    run_after_for,
)
from gubbi.extraction.limits import ExtractionLimits, limits_for
from gubbi.models.conversation import Message, keep_conversation_roles
from gubbi.storage.connection import safe_acquire
from gubbi.storage.exceptions import TopicNotFoundError
from gubbi.storage.repositories import conversations as conv_repo
from gubbi.storage.repositories import extraction_jobs, extraction_queue
from gubbi.storage.repositories.extraction_jobs import ExtractionJobAlreadyInFlight
from gubbi.storage.repositories.topics import TopicAlreadyExists
from gubbi.storage.repositories.topics import create as create_topic
from gubbi.storage.repositories.topics import get_id as get_topic_id
from gubbi.validation import validate_title

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from gubbi.app_context import AppContext

__all__: list[str] = [
    "DEFAULT_INBOX_TOPIC",
    "MAX_CONVERSATIONS_PER_REQUEST",
    "ConversationPayload",
    "IngestConversationRequest",
    "IngestConversationResponse",
    "MessagePayload",
    "ingest_conversations",
    "router",
]

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/ingest", tags=["ingest"])

MAX_CONVERSATIONS_PER_REQUEST = 50
DEFAULT_INBOX_TOPIC = "inbox"


class MessagePayload(BaseModel):
    """A single message in a conversation payload."""

    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime | None = None


class ConversationPayload(BaseModel):
    """A single conversation payload from the ingest request."""

    platform: Literal["chatgpt", "claude"]
    platform_id: str = Field(min_length=1, max_length=512)
    title: str = ""
    created_at: datetime
    updated_at: datetime | None = None
    messages: Annotated[list[MessagePayload], Field(min_length=1)]


class IngestConversationRequest(BaseModel):
    """Top-level ingest request body."""

    source: Literal["extension_chatgpt", "extension_claude", "paste_memories", "zip_upload"]
    conversations: Annotated[
        list[ConversationPayload], Field(max_length=MAX_CONVERSATIONS_PER_REQUEST)
    ]


class IngestConversationResponse(BaseModel):
    """Response with per-batch save + extraction-queue counters.

    conversations_saved        -- rows committed to ``conversations``.
    conversations_skipped_dedupe -- rows skipped because the user already had
                                  this (platform, platform_id) pair.
    extractions_enqueued       -- saves with a queued extraction job: a job
                                  created by this request, or an earlier one
                                  for the same conversation still in flight.
    extractions_skipped_budget -- always 0: extraction work is queued, never
                                  refused for budget.
    extractions_skipped_error  -- saves whose extraction job could not be
                                  recorded; the conversation row is durable.
    budget_exhausted           -- True when the user has used up the current
                                  month's extraction allowance, so new work
                                  waits for the next period. Always False
                                  when budgets are disabled.

    Invariant:
    conversations_saved == extractions_enqueued + extractions_skipped_error
    """

    conversations_saved: int
    conversations_skipped_dedupe: int
    extractions_enqueued: int
    extractions_skipped_budget: int
    extractions_skipped_error: int
    budget_exhausted: bool


def _get_app_ctx(request: Request) -> AppContext:
    """Extract AppContext from the application state.

    Thin wrapper around :func:`gubbi.app_state.require_app_ctx` so the
    rest of this module reads as before. Populated during lifespan by
    ``main.py``.
    """
    return require_app_ctx(request)


@dataclass(frozen=True, slots=True)
class _PriorUploads:
    """The user's earliest sync uploads recorded before the current request."""

    any_platform: datetime | None
    by_platform: Mapping[str, datetime | None]


async def _read_prior_uploads(
    conn: asyncpg.Connection, user_id: UUID, platforms: Iterable[str]
) -> _PriorUploads:
    """Read the user's first uploads; RLS scopes the read to the user's own jobs."""
    async with conn.transaction():
        await _set_user_rls_local(conn, user_id)
        any_platform = await extraction_queue.first_upload_at(conn, user_id, None)
        by_platform = {
            platform: await extraction_queue.first_upload_at(conn, user_id, platform)
            for platform in sorted(set(platforms))
        }
    return _PriorUploads(any_platform=any_platform, by_platform=by_platform)


def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _activity_at(conv: ConversationPayload, kept: Iterable[MessagePayload]) -> datetime:
    """Latest timestamp among the kept messages, else the conversation's creation time."""
    stamps = [_as_utc(msg.timestamp) for msg in kept if msg.timestamp is not None]
    return max(stamps, default=_as_utc(conv.created_at))


async def _budget_exhausted(
    conn: asyncpg.Connection,
    user_id: UUID,
    limits: ExtractionLimits,
    now: datetime,
) -> bool:
    """Whether the user's monthly extraction allowance is used up for the current period."""
    async with conn.transaction():
        await _set_user_rls_local(conn, user_id)
        totals = await extraction_queue.user_period_usage(conn, [user_id], budget_period(now))
    usage = user_usage_from_totals(**asdict(totals[user_id]), limits=limits)
    return monthly_allowance_exhausted(usage, limits)


async def _set_user_rls_local(conn: asyncpg.Connection, user_id: UUID) -> None:
    """Apply the ``app.current_user_id`` RLS GUC scoped to the current transaction.

    PostgreSQL's ``set_config(name, value, is_local=true)`` is the
    expression-context equivalent of ``SET LOCAL`` -- it requires an
    open transaction to be transaction-scoped. Outside any transaction,
    the third argument is silently ignored and the GUC promotes to
    session scope; on a pooled connection that means the next checkout
    inherits the previous user's identity (the original bug).

    This helper centralises the SET-LOCAL incantation so the
    security-critical contract -- "set the GUC inside a transaction
    only" -- has one canonical home. Every call site in this module
    must already be inside an ``async with conn.transaction():`` block;
    the helper itself does not open one (callers vary in how they
    couple SET LOCAL to the rest of their transaction body).
    """
    await conn.execute(
        "SELECT set_config('app.current_user_id', $1, true)",
        str(user_id),
    )


@router.post("/conversations", response_model=IngestConversationResponse)
async def ingest_conversations(
    request: Request,
    body: IngestConversationRequest,
    auth: Annotated[tuple[UUID, frozenset[str]], Depends(require_scope("journal:write"))],
) -> IngestConversationResponse:
    """Ingest normalized conversation batches from browser extension clients.

    Each conversation is deduped by (user_id, platform, platform_id), saved
    under the default "inbox" topic, and gets a pending extraction job. The
    job carries its scheduling lane, when it may first run, the latest
    message time and the filtered token count; nothing is enqueued or charged
    here -- the extraction scheduler admits queued jobs against the budget.

    Lanes: a paste request is ``paste`` and runs now. A synced conversation
    is ``backfill`` (runs now) when it predates the user's first upload from
    its platform or arrives within the onboarding window after their first
    upload of any platform, else ``live`` (runs at the next nightly window).
    Those first uploads are read once, before this request records any job,
    so every conversation of one request is judged against the same history.

    Transaction design:
    - Acquires ONE bare connection for the request via ``safe_acquire``.
    - Each independent unit of DB work opens its own ``conn.transaction()``
      and applies the ``app.current_user_id`` RLS GUC with ``SET LOCAL``
      inside that transaction. ``SET LOCAL`` is transaction-scoped, so the
      GUC unwinds at COMMIT/ROLLBACK and the connection cannot return to
      the pool carrying a session-level GUC for the prior user.
    - Dedupe pre-check, conversation save (TXN 1), and extraction_jobs
      INSERT (TXN 2) are each their own top-level transaction. They commit
      independently: a TXN 2 failure does NOT roll back TXN 1's
      conversation row.
    - On TXN 2 failure: count as extractions_skipped_error, continue (HTTP 200).
    """
    user_id, _scopes = auth
    app_ctx = _get_app_ctx(request)
    cipher = require_cipher(app_ctx)
    log = bound_logger(request)

    now = datetime.now(UTC)
    limits = limits_for(user_id)

    conversations_saved = 0
    conversations_skipped_dedupe = 0
    extractions_enqueued = 0
    extractions_skipped_error = 0
    superseded_json_paths: list[str] = []

    async with safe_acquire(app_ctx.pool) as conn:
        # Ensure inbox topic exists -- own transaction, GUC set with SET LOCAL.
        # Concurrent ingest race: another request for the same user may have
        # created the inbox between our get and create. ``TopicAlreadyExists``
        # is the same-user-duplicate signal post-migration 0030
        # (topics_user_path_key UNIQUE (user_id, path)); treat as a no-op
        # since the topic now exists. The ``contextlib.suppress`` lives
        # OUTSIDE the transaction so the exception triggers ROLLBACK
        # (clearing PG's aborted-transaction state) before being swallowed.
        with contextlib.suppress(TopicAlreadyExists):
            async with conn.transaction():
                await _set_user_rls_local(conn, user_id)
                try:
                    await get_topic_id(conn, DEFAULT_INBOX_TOPIC)
                except TopicNotFoundError:
                    await create_topic(conn, DEFAULT_INBOX_TOPIC, title="Inbox")

        prior_uploads = await _read_prior_uploads(
            conn, user_id, (conv.platform for conv in body.conversations)
        )

        for conv in body.conversations:
            kept_payloads = keep_conversation_roles(conv.messages)
            messages = [
                Message(
                    role=msg.role,
                    content=msg.content,
                    timestamp=msg.timestamp.isoformat() if msg.timestamp else None,
                )
                for msg in kept_payloads
            ]
            if not messages:
                await log.info(
                    "ingest_conversation_empty_after_role_filter",
                    platform=conv.platform,
                    platform_id=conv.platform_id,
                    message_count=len(conv.messages),
                )
                continue

            # Dedupe pre-check in its own transaction so the SET LOCAL
            # GUC is properly transaction-scoped (and unwinds before the
            # connection is reused). The pre-check requires RLS to be
            # active to function correctly: without the GUC, the FORCE
            # ROW LEVEL SECURITY policy on conversations would filter all
            # rows and the read would falsely report "not exists",
            # leading to redundant TXN 1 inserts that the UNIQUE
            # constraint would then catch as dedupe-races. Setting the
            # GUC inside this short transaction restores the fast path.
            async with conn.transaction():
                await _set_user_rls_local(conn, user_id)
                existing = await conv_repo.exists_by_platform_id(
                    conn,
                    user_id,
                    conv.platform,
                    conv.platform_id,
                )
            if existing:
                conversations_skipped_dedupe += 1
                continue

            # Build a valid non-empty title (save_conversation requires it)
            try:
                title = validate_title(conv.title)
            except ValueError:
                title = validate_title(f"Conversation {conv.created_at.isoformat()}")

            # ============================================================
            # TRANSACTION 1: save conversation + set platform metadata.
            # Top-level transaction; commits independently of TXN 2 below.
            # SET LOCAL applies the RLS GUC for this transaction only.
            # ============================================================
            save_result = None
            try:
                async with conn.transaction():
                    await _set_user_rls_local(conn, user_id)
                    save_result = await conv_repo.save_conversation(
                        conn,
                        cipher,
                        conversations_json_dir=app_ctx.settings.conversations_json_dir,
                        topic=DEFAULT_INBOX_TOPIC,
                        title=title,
                        messages=messages,
                        summary="",
                        source=conv.platform,
                        date=conv.created_at.date().isoformat(),
                    )
                    await conv_repo.set_platform_metadata(
                        conn,
                        save_result.conversation_id,
                        conv.platform,
                        conv.platform_id,
                    )
            except asyncpg.UniqueViolationError:
                await log.warning(
                    "Dedupe race: platform_id already exists, treating as skip",
                    platform=conv.platform,
                    platform_id=conv.platform_id,
                )
                conversations_skipped_dedupe += 1
                continue

            # TXN 1 committed -- conversation row is durable.
            assert save_result is not None  # noqa: S101  -- mypy; TXN 1 sets this
            conversations_saved += 1
            if save_result.superseded_json_path is not None:
                superseded_json_paths.append(save_result.superseded_json_path)

            activity_at = _activity_at(conv, kept_payloads)
            lane = assign_lane(
                body.source,
                activity_at=activity_at,
                now=now,
                first_upload_at_platform=prior_uploads.by_platform[conv.platform],
                first_upload_at_any=prior_uploads.any_platform,
                onboarding_window=limits.onboarding_window,
            )

            # ============================================================
            # TRANSACTION 2: extraction_jobs INSERT + audit.
            # Top-level transaction; rollback here only undoes TXN 2,
            # leaving TXN 1's conversation row durable. SET LOCAL applies
            # the RLS GUC for this transaction only.
            # On failure: log, count, CONTINUE. HTTP 200 is still returned.
            # ============================================================
            try:
                async with conn.transaction():
                    await _set_user_rls_local(conn, user_id)
                    try:
                        job_uuid = await extraction_jobs.create_pending(
                            conn,
                            user_id=user_id,
                            conversation_id=save_result.conversation_id,
                            source=conv.platform,
                            lane=lane.value,
                            run_after=run_after_for(lane, now, limits.nightly_window_hour),
                            activity_at=activity_at,
                            filtered_tokens=count_filtered_tokens(messages),
                            token_count_version=TOKEN_COUNT_VERSION,
                        )
                    except ExtractionJobAlreadyInFlight as exc:
                        job_uuid = exc.existing_job_id

                    await record_audit(
                        conn,
                        actor_type="user",
                        actor_id=str(user_id),
                        action="extraction_job.created",
                        target_kind=TargetKind.EXTRACTION_JOB,
                        target_id=str(job_uuid),
                        metadata={
                            "conversation_id": save_result.conversation_id,
                            "source": conv.platform,
                            "lane": lane.value,
                        },
                    )
            except Exception:
                extractions_skipped_error += 1
                await log.warning(
                    "extraction_job_txn_failed",
                    user_id=str(user_id),
                    conversation_id=save_result.conversation_id,
                    exc_info=True,
                )
                continue  # do NOT re-raise; conv save is durable, return 200.

            extractions_enqueued += 1

        budget_exhausted = get_settings().llm.llm_budget_enabled and await _budget_exhausted(
            conn, user_id, limits, now
        )

    for path in superseded_json_paths:
        conv_repo.delete_superseded_json_archive(app_ctx.settings.conversations_json_dir, path)

    return IngestConversationResponse(
        conversations_saved=conversations_saved,
        conversations_skipped_dedupe=conversations_skipped_dedupe,
        extractions_enqueued=extractions_enqueued,
        extractions_skipped_budget=0,
        extractions_skipped_error=extractions_skipped_error,
        budget_exhausted=budget_exhausted,
    )
