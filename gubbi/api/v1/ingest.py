"""REST API: POST /api/v1/ingest/conversations.

Accepts normalized conversation batches from the browser extension.
Transforms, dedupes, and saves to conversations table.
No LLM calls. Pure data ingest.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

import asyncpg
import structlog
from fastapi import APIRouter, Depends, Request
from gubbi_common.audit.targets import TargetKind
from gubbi_common.budget import PRE_CHARGE_CENTS, current_period_start
from gubbi_common.telemetry import bound_logger
from pydantic import BaseModel, Field

from gubbi.api.v1.auth import require_scope
from gubbi.app_context import AppContext
from gubbi.app_state import get_optional_arq_pool, require_app_ctx
from gubbi.audit.sql import record_audit
from gubbi.crypto.guard import require_cipher
from gubbi.models.conversation import Message
from gubbi.storage.connection import safe_acquire
from gubbi.storage.exceptions import TopicNotFoundError
from gubbi.storage.repositories import conversations as conv_repo
from gubbi.storage.repositories import extraction_jobs
from gubbi.storage.repositories.extraction_jobs import ExtractionJobAlreadyInFlight
from gubbi.storage.repositories.topics import create as create_topic
from gubbi.storage.repositories.topics import get_id as get_topic_id
from gubbi.validation import validate_title

__all__: list[str] = [
    "ConversationPayload",
    "DEFAULT_INBOX_TOPIC",
    "IngestConversationRequest",
    "IngestConversationResponse",
    "MAX_CONVERSATIONS_PER_REQUEST",
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
    """Response with per-batch save + extraction-enqueue counters.

    conversations_saved        -- rows committed to ``conversations``.
    conversations_skipped_dedupe -- rows skipped because the user already had
                                  this (platform, platform_id) pair.
    extractions_enqueued       -- arq jobs enqueued for new saves.
    extractions_skipped_budget -- saves where pre_charge denied (budget cap hit).
    extractions_skipped_error  -- saves where TXN 2 (extraction_jobs INSERT) failed;
                                  conversation row is durable, pre-charge refunded.
    budget_exhausted           -- True if any conversation hit the cap in this batch.

    Invariant:
    conversations_saved == extractions_enqueued + extractions_skipped_budget
                        + extractions_skipped_error
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


@router.post("/conversations", response_model=IngestConversationResponse)
async def ingest_conversations(
    request: Request,
    body: IngestConversationRequest,
    auth: Annotated[tuple[UUID, frozenset[str]], Depends(require_scope("journal:write"))],
) -> IngestConversationResponse:
    """Ingest normalized conversation batches from browser extension clients.

    Each conversation is deduped by (user_id, platform, platform_id),
    saved under the default "inbox" topic, and returns counts of saved
    vs skipped conversations.

    Transaction design (B3-H1 fix):
    - Acquires ONE connection for the request (no outer transaction).
    - Each conversation save is its own top-level transaction (TXN 1).
    - Each extraction_jobs INSERT is its own top-level transaction (TXN 2).
    - TXN 1 and TXN 2 commit independently: a TXN 2 failure does NOT roll
      back TXN 1 (conversation save is durable per D1).
    - RLS GUC is set with SET LOCAL inside each transaction via set_config.
    - On TXN 2 failure: refund pre-charge, count as extractions_skipped_error,
      continue (HTTP 200).
    """
    user_id, _scopes = auth
    app_ctx = _get_app_ctx(request)
    cipher = require_cipher(app_ctx)
    log = bound_logger(request)

    helper = app_ctx.budget_helper  # may be None (self-host / budget disabled)
    period_start = current_period_start()  # compute once outside the loop

    conversations_saved = 0
    conversations_skipped_dedupe = 0
    extractions_skipped_budget = 0
    extractions_skipped_error = 0
    budget_exhausted = False
    superseded_json_paths: list[str] = []
    enqueue_tasks: list[tuple[UUID, int]] = []

    async with safe_acquire(app_ctx.pool) as conn:
        # Ensure inbox topic exists -- run in its own txn so failure does not
        # poison subsequent conversation saves.
        async with conn.transaction():
            await conn.execute(
                "SELECT set_config('app.current_user_id', $1, true)",
                str(user_id),
            )
            try:
                await get_topic_id(conn, DEFAULT_INBOX_TOPIC)
            except TopicNotFoundError:
                await create_topic(conn, DEFAULT_INBOX_TOPIC, title="Inbox")

        for conv in body.conversations:
            # Dedupe pre-check: read-only, no explicit txn needed; the UNIQUE
            # constraint on (platform, platform_id) inside TXN 1 catches the race.
            await conn.execute(
                "SELECT set_config('app.current_user_id', $1, true)",
                str(user_id),
            )
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

            messages = [
                Message(
                    role=msg.role,
                    content=msg.content,
                    timestamp=msg.timestamp.isoformat() if msg.timestamp else None,
                )
                for msg in conv.messages
            ]

            # ============================================================
            # TRANSACTION 1: save conversation + set platform metadata.
            # Top-level (no outer txn active) so it commits independently.
            # UniqueViolationError on the platform_id race rolls back ONLY
            # this txn; treat as dedupe skip.
            # ============================================================
            save_result = None
            try:
                async with conn.transaction():
                    await conn.execute(
                        "SELECT set_config('app.current_user_id', $1, true)",
                        str(user_id),
                    )
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

            # TXN 1 committed -- conversation row is durable (D1).
            assert save_result is not None  # noqa: S101  -- mypy; TXN 1 sets this
            conversations_saved += 1
            if save_result.superseded_json_path is not None:
                superseded_json_paths.append(save_result.superseded_json_path)

            # ============================================================
            # PRE-CHARGE: outside any txn.
            # helper is None -> budget disabled -> treat as pre_charged=True.
            # ============================================================
            if helper is None:
                pre_charged = True
            else:
                pre_charged = await helper.pre_charge(
                    user_id=user_id,
                    period_start=period_start,
                    estimated_cents=PRE_CHARGE_CENTS,
                )

            if not pre_charged:
                extractions_skipped_budget += 1
                budget_exhausted = True
                continue  # conv committed; just don't enqueue.

            # ============================================================
            # TRANSACTION 2: extraction_jobs INSERT + audit.
            # Top-level -- rolls back independently of TXN 1.
            # On failure: refund pre-charge (D4), log, count, CONTINUE.
            # HTTP 200 is still returned (D3); save is durable (D1).
            # ============================================================
            job_uuid: UUID | None = None
            try:
                async with conn.transaction():
                    await conn.execute(
                        "SELECT set_config('app.current_user_id', $1, true)",
                        str(user_id),
                    )
                    try:
                        job_uuid = await extraction_jobs.create_pending(
                            conn,
                            user_id=user_id,
                            conversation_id=save_result.conversation_id,
                            source=conv.platform,
                            period_start=period_start,
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
                        },
                    )
            except Exception:
                # TXN 2 rolled back.  Conv save (TXN 1) is already durable.
                # Refund pre-charge so user is not over-billed (D4).
                extractions_skipped_error += 1
                if helper is not None:
                    try:
                        await helper.record_actual_cost(
                            user_id=user_id,
                            period_start=period_start,
                            actual_cents=0,
                            estimated_cents=PRE_CHARGE_CENTS,
                        )
                    except Exception:  # noqa: BLE001
                        await log.warning(
                            "pre_charge_refund_failed",
                            user_id=str(user_id),
                            conversation_id=save_result.conversation_id,
                            exc_info=True,
                        )
                await log.warning(
                    "extraction_job_txn_failed",
                    user_id=str(user_id),
                    conversation_id=save_result.conversation_id,
                    exc_info=True,
                )
                continue  # do NOT re-raise; conv save is durable, return 200.

            # TXN 2 committed.  Collect for post-commit arq enqueue.
            assert job_uuid is not None  # noqa: S101  -- mypy; TXN 2 sets this
            enqueue_tasks.append((job_uuid, save_result.conversation_id))

    # Outside the connection.  Post-commit: enqueue arq jobs AFTER the DB
    # transactions have committed so the worker cannot race ahead of the rows.
    #
    # ``str(job_uuid)`` is passed BOTH as the 4th positional arg AND as the
    # ``_job_id=`` kwarg. They serve different roles -- arq does not bridge
    # them. ``_job_id=`` is arq-internal: it becomes the queue key + result
    # key (used for dedup + result lookup) and is NOT forwarded to the
    # worker function. The 4th positional is what the worker function
    # receives as its ``job_id`` parameter -- it is the one
    # ``extract_conversation`` reads to gate ``mark_running`` /
    # ``mark_completed`` / ``mark_failed`` and route the audit ``target_id``.
    # Dropping the positional makes the worker run with the
    # ``job_id="unknown"`` default and silently skips every FSM transition,
    # leaving ``extraction_jobs.status`` stuck at ``'pending'``.
    arq_pool = get_optional_arq_pool(request)
    if arq_pool is not None:
        for job_uuid, conversation_id in enqueue_tasks:
            await arq_pool.enqueue_job(
                "extract_conversation",
                conversation_id,
                str(user_id),
                str(job_uuid),
                _job_id=str(job_uuid),
            )

    for path in superseded_json_paths:
        conv_repo.delete_superseded_json_archive(app_ctx.settings.conversations_json_dir, path)

    return IngestConversationResponse(
        conversations_saved=conversations_saved,
        conversations_skipped_dedupe=conversations_skipped_dedupe,
        extractions_enqueued=len(enqueue_tasks),
        extractions_skipped_budget=extractions_skipped_budget,
        extractions_skipped_error=extractions_skipped_error,
        budget_exhausted=budget_exhausted,
    )
