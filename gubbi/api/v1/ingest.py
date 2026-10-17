"""REST API: POST /api/v1/ingest/conversations.

Accepts normalized conversation batches from the browser extension.
Merges each conversation into stored history by its platform id (and each
message by its source message id), and records a pending extraction job for
every conversation that is new or gained messages, with its scheduling lane
and run time. No LLM calls and no job dispatch: the extraction scheduler
admits and runs queued jobs.
"""

from __future__ import annotations

import contextlib
from dataclasses import asdict, dataclass

# datetime is a Pydantic field type on MessagePayload / ConversationPayload;
# UUID is a FastAPI Depends() Annotated type. Both are resolved at runtime
# (validator-build / route-registration), so they must stay runtime imports
# despite `from __future__ import annotations` -- ruff TC would otherwise
# push them under TYPE_CHECKING and break Pydantic model_rebuild().
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Annotated, Literal
from uuid import UUID  # noqa: TC003

import structlog
from fastapi import APIRouter, Depends, Request
from gubbi_common.audit.targets import TargetKind
from gubbi_common.telemetry import bound_logger
from pydantic import BaseModel, Field

from gubbi.api.v1.auth import require_scope
from gubbi.api.v1.body_limit import IngestBodyLimitRoute
from gubbi.api.v1.ingest_time import IngestTimestamp  # noqa: TC001 (Pydantic field type)
from gubbi.app_state import require_app_ctx
from gubbi.audit.sql import record_audit
from gubbi.config import get_settings
from gubbi.crypto.guard import require_cipher
from gubbi.extraction.intake import assign_lane
from gubbi.extraction.spend_rules import limits_for, run_after_for
from gubbi.models.conversation import keep_conversation_roles
from gubbi.storage.connection import safe_acquire
from gubbi.storage.exceptions import TopicNotFoundError
from gubbi.storage.repositories import extraction_jobs, extraction_queue
from gubbi.storage.repositories import wallet as wallet_repo
from gubbi.storage.repositories.extraction_jobs import ExtractionJobAlreadyInFlight
from gubbi.storage.repositories.synced_conversations import (
    MergeResult,
    SyncedConversation,
    SyncedMessage,
    merge_synced_conversation,
)
from gubbi.storage.repositories.topics import TopicAlreadyExists
from gubbi.storage.repositories.topics import create as create_topic
from gubbi.storage.repositories.topics import get_id as get_topic_id
from gubbi.telemetry.attrs import MetricNames
from gubbi.telemetry.metrics import record_counter
from gubbi.validation import validate_title

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    import asyncpg

    from gubbi.app_context import AppContext
    from gubbi.crypto.cipher import ContentCipher

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

router = APIRouter(prefix="/ingest", tags=["ingest"], route_class=IngestBodyLimitRoute)

MAX_CONVERSATIONS_PER_REQUEST = 50
DEFAULT_INBOX_TOPIC = "inbox"


# Length-constrained strings are checked as Unicode by pydantic-core, which
# refuses unpaired surrogates (string_unicode). Identifiers are stored and
# compared verbatim, so they rely on that instead of being repaired like text.
_SourceId = Annotated[str, Field(min_length=1, max_length=512)]


class MessagePayload(BaseModel):
    """A single message in a conversation payload."""

    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: IngestTimestamp | None = None
    message_id: _SourceId | None = None


class ConversationPayload(BaseModel):
    """A single conversation payload from the ingest request."""

    platform: Literal["chatgpt", "claude"]
    platform_id: _SourceId
    title: str = ""
    created_at: IngestTimestamp
    updated_at: IngestTimestamp | None = None
    messages: Annotated[list[MessagePayload], Field(min_length=1)]


class IngestConversationRequest(BaseModel):
    """Top-level ingest request body."""

    source: Literal["extension_chatgpt", "extension_claude", "paste_memories", "zip_upload"]
    conversations: Annotated[
        list[ConversationPayload], Field(max_length=MAX_CONVERSATIONS_PER_REQUEST)
    ]


class IngestConversationResponse(BaseModel):
    """Response with per-batch save + extraction-queue counters.

    conversations_saved        -- conversations stored for the first time or
                                  extended with new messages.
    conversations_skipped_dedupe -- conversations that gained no message: an
                                  identical re-send, a re-send that only
                                  edited stored messages, or one whose
                                  messages could not be matched to the stored
                                  thread (nothing written).
    extractions_enqueued       -- saves with a queued extraction job: a job
                                  created by this request, or an earlier one
                                  for the same conversation still in flight.
    extractions_skipped_budget -- always 0: extraction work is queued, never
                                  refused for budget.
    extractions_skipped_error  -- conversations that failed to save: the
                                  conversation and its extraction job are
                                  stored together or not at all, so nothing
                                  was kept and a re-send is accepted.
    budget_exhausted           -- True when the user's extraction wallet has
                                  nothing left (monthly allowance and credit
                                  used up), so new work waits for the next
                                  month or more credit. Always False when
                                  budgets are disabled.

    Invariant:
    conversations_saved == extractions_enqueued
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


def _activity_at(
    conv: ConversationPayload, kept: Iterable[MessagePayload], now: datetime
) -> datetime:
    """Latest timestamp among the kept messages, else the conversation's update, else creation.

    Never later than ``now``: a client clock ahead of the server cannot push a
    conversation ahead of others in its lane.
    """
    stamps = [msg.timestamp for msg in kept if msg.timestamp is not None]
    return min(max(stamps, default=conv.updated_at or conv.created_at), now)


@dataclass(frozen=True, slots=True)
class _QueueFields:
    """Scheduling inputs of a conversation's extraction job."""

    lane: str
    run_after: datetime
    activity_at: datetime


async def _queue_extraction(
    conn: asyncpg.Connection,
    user_id: UUID,
    conversation_id: int,
    source: str,
    fields: _QueueFields,
) -> None:
    """Create the conversation's pending job, or reuse and refresh the one in flight.

    Must run inside the caller's transaction, after the RLS GUC is set. Only a
    newly created job is audited as created.
    """
    try:
        job_uuid = await extraction_jobs.create_pending(
            conn,
            user_id=user_id,
            conversation_id=conversation_id,
            source=source,
            **asdict(fields),
        )
    except ExtractionJobAlreadyInFlight as exc:
        await extraction_jobs.refresh_queued(conn, exc.existing_job_id, **asdict(fields))
        return

    await record_audit(
        conn,
        actor_type="user",
        actor_id=str(user_id),
        action="extraction_job.created",
        target_kind=TargetKind.EXTRACTION_JOB,
        target_id=str(job_uuid),
        metadata={
            "conversation_id": conversation_id,
            "source": source,
            "lane": fields.lane,
        },
    )


async def _budget_exhausted(conn: asyncpg.Connection, user_id: UUID) -> bool:
    """Whether the user's wallet has nothing available."""
    async with conn.transaction():
        await _set_user_rls_local(conn, user_id)
        balances = await wallet_repo.available(conn, {user_id: limits_for(user_id).monthly_amount})
    return balances[user_id] <= 0


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


@dataclass(slots=True)
class _Tally:
    """Per-request counters for the response and the request log."""

    saved: int = 0
    skipped_dedupe: int = 0
    skipped_error: int = 0
    messages_truncated: int = 0


def _synced_conversation(
    conv: ConversationPayload, kept: Iterable[MessagePayload]
) -> SyncedConversation:
    """The payload as the merge reads it, filed under the inbox with a non-empty title."""
    try:
        title = validate_title(conv.title)
    except ValueError:
        title = validate_title(f"Conversation {conv.created_at.isoformat()}")
    return SyncedConversation(
        platform=conv.platform,
        platform_id=conv.platform_id,
        title=title,
        topic=DEFAULT_INBOX_TOPIC,
        created_at=conv.created_at,
        messages=tuple(
            SyncedMessage(
                role=msg.role,
                content=msg.content,
                timestamp=msg.timestamp,
                message_id=msg.message_id,
            )
            for msg in kept
        ),
    )


async def _merge_and_queue(
    conn: asyncpg.Connection,
    cipher: ContentCipher,
    user_id: UUID,
    synced: SyncedConversation,
    fields: _QueueFields,
) -> MergeResult:
    """Merge one conversation and queue its extraction when it gained messages.

    The merge, the job and the job's audit row commit together or not at all.
    """
    async with conn.transaction():
        await _set_user_rls_local(conn, user_id)
        result = await merge_synced_conversation(conn, cipher, synced)
        if result.created or result.has_new_messages:
            if result.conversation_id is None:
                raise RuntimeError("a merge that wrote messages returned no conversation id")
            await _queue_extraction(conn, user_id, result.conversation_id, synced.platform, fields)
    return result


@router.post("/conversations", response_model=IngestConversationResponse)
async def ingest_conversations(
    request: Request,
    body: IngestConversationRequest,
    auth: Annotated[tuple[UUID, frozenset[str]], Depends(require_scope("journal:write"))],
) -> IngestConversationResponse:
    """Ingest normalized conversation batches from browser extension clients.

    Each conversation is identified by (user_id, platform, platform_id) and
    merged into the stored thread under the default "inbox" topic: a new
    thread is stored whole, a known one gains the messages it does not hold
    yet, and stored messages whose text changed are rewritten in place. A
    conversation that is new or gained messages gets a pending extraction
    job; one that gained nothing (an identical re-send, edits only, or
    messages that do not match the stored thread) counts as skipped. The job
    carries its scheduling lane, when it may first run and the latest message
    time (never later than now); nothing is enqueued or charged here -- the
    extraction scheduler admits queued jobs against the budget, and a job is
    charged after it runs.

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
    - A conversation's merge, its extraction job and the job's audit row
      share ONE transaction: if any of them fails, nothing is stored, the
      conversation counts as extractions_skipped_error and the request
      continues (HTTP 200). A re-send of it is then merged afresh.
    - An existing in-flight job for the conversation is reused. While it is
      pending and not yet dispatched, its lane, run time and activity are
      refreshed from this upload; only a newly created job is audited as
      created.
    """
    user_id, _scopes = auth
    app_ctx = _get_app_ctx(request)
    cipher = require_cipher(app_ctx)
    log = bound_logger(request)

    now = datetime.now(UTC)
    llm_settings = get_settings().llm
    tally = _Tally()

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
            if not kept_payloads:
                await log.info(
                    "ingest_conversation_empty_after_role_filter",
                    platform=conv.platform,
                    platform_id=conv.platform_id,
                    message_count=len(conv.messages),
                )
                continue

            activity_at = _activity_at(conv, kept_payloads, now)
            lane = assign_lane(
                body.source,
                activity_at=activity_at,
                now=now,
                first_upload_at_platform=prior_uploads.by_platform[conv.platform],
                first_upload_at_any=prior_uploads.any_platform,
                onboarding_window=timedelta(days=llm_settings.onboarding_window_days),
            )
            queue_fields = _QueueFields(
                lane=lane.value,
                run_after=run_after_for(lane, now, llm_settings.nightly_window_hour),
                activity_at=activity_at,
            )

            try:
                result = await _merge_and_queue(
                    conn,
                    cipher,
                    user_id,
                    _synced_conversation(conv, kept_payloads),
                    queue_fields,
                )
            except Exception:
                tally.skipped_error += 1
                await log.warning(
                    "ingest_conversation_save_failed",
                    user_id=str(user_id),
                    platform=conv.platform,
                    platform_id=conv.platform_id,
                    exc_info=True,
                )
                continue

            tally.messages_truncated += result.truncated
            if result.created or result.has_new_messages:
                tally.saved += 1
            else:
                tally.skipped_dedupe += 1
            if result.legacy_identity_conflict:
                await log.warning("ingest_legacy_identity_conflict", platform=conv.platform)

        budget_exhausted = llm_settings.llm_budget_enabled and await _budget_exhausted(
            conn, user_id
        )

    if tally.messages_truncated:
        record_counter(MetricNames.INGEST_MESSAGES_TRUNCATED, amount=tally.messages_truncated)
    await log.info(
        "ingest_request_completed",
        source=body.source,
        conversations_saved=tally.saved,
        conversations_skipped_dedupe=tally.skipped_dedupe,
        extractions_skipped_error=tally.skipped_error,
        messages_truncated=tally.messages_truncated,
    )

    return IngestConversationResponse(
        conversations_saved=tally.saved,
        conversations_skipped_dedupe=tally.skipped_dedupe,
        extractions_enqueued=tally.saved,
        extractions_skipped_budget=0,
        extractions_skipped_error=tally.skipped_error,
        budget_exhausted=budget_exhausted,
    )
