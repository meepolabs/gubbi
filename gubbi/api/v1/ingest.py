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
from gubbi_common.telemetry import bound_logger
from pydantic import BaseModel, Field

from gubbi.api.v1.auth import require_scope
from gubbi.app_context import AppContext
from gubbi.app_state import get_optional_arq_pool, require_app_ctx
from gubbi.audit.sql import record_audit
from gubbi.crypto.guard import require_cipher
from gubbi.models.conversation import Message
from gubbi.storage.connection import safe_user_scoped_connection
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
    """Response indicating how many conversations were saved vs deduped."""

    conversations_saved: int
    conversations_skipped_dedupe: int


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
    """
    user_id, _scopes = auth
    app_ctx = _get_app_ctx(request)
    cipher = require_cipher(app_ctx)
    log = bound_logger(request)

    conversations_saved = 0
    conversations_skipped_dedupe = 0
    superseded_json_paths: list[str] = []
    # Collect (job_uuid, conversation_id, source) tuples for post-commit enqueue.
    enqueue_tasks: list[tuple[UUID, int, str]] = []

    async with safe_user_scoped_connection(app_ctx.pool, user_id=user_id) as conn:
        # Ensure the inbox topic exists before saving conversations
        try:
            await get_topic_id(conn, DEFAULT_INBOX_TOPIC)
        except TopicNotFoundError:
            await create_topic(conn, DEFAULT_INBOX_TOPIC, title="Inbox")

        for conv in body.conversations:
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

            # Wrap the save + platform UPDATE in a savepoint so that a
            # UniqueViolationError on the platform-id race does not abort
            # the outer per-request transaction (which would break every
            # subsequent loop iteration).
            try:
                async with conn.transaction():
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

                    # Enqueue extraction job for this newly saved conversation.
                    # On partial-unique conflict (same conversation already in-flight),
                    # reuse the existing job_id for idempotent semantics.
                    try:
                        job_uuid = await extraction_jobs.create_pending(
                            conn,
                            user_id=user_id,
                            conversation_id=save_result.conversation_id,
                            source=conv.platform,
                        )
                    except ExtractionJobAlreadyInFlight as exc:
                        job_uuid = exc.existing_job_id

                    await record_audit(
                        conn,
                        actor_type="user",
                        actor_id=str(user_id),
                        action="extraction_job.created",
                        target_kind="extraction_job",
                        target_id=str(job_uuid),
                        metadata={
                            "conversation_id": save_result.conversation_id,
                            "source": conv.platform,
                        },
                    )

            except asyncpg.UniqueViolationError:
                await log.warning(
                    "Dedupe race: platform_id already exists, treating as skip",
                    platform=conv.platform,
                    platform_id=conv.platform_id,
                )
                conversations_skipped_dedupe += 1
                continue

            if save_result.superseded_json_path is not None:
                superseded_json_paths.append(save_result.superseded_json_path)
            conversations_saved += 1
            # Collect for post-commit arq enqueue (outside the DB transaction).
            enqueue_tasks.append((job_uuid, save_result.conversation_id, conv.platform))

    # Post-commit: enqueue arq jobs AFTER the DB transaction has committed so
    # the worker cannot race ahead of the committed rows.
    arq_pool = get_optional_arq_pool(request)
    if arq_pool is not None:
        for job_uuid, conversation_id, _source in enqueue_tasks:
            await arq_pool.enqueue_job(
                "extract_conversation",
                conversation_id,
                str(user_id),
                _job_id=str(job_uuid),
            )

    for path in superseded_json_paths:
        conv_repo.delete_superseded_json_archive(app_ctx.settings.conversations_json_dir, path)

    return IngestConversationResponse(
        conversations_saved=conversations_saved,
        conversations_skipped_dedupe=conversations_skipped_dedupe,
    )
