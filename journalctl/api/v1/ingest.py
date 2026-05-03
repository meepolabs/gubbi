"""REST API: POST /api/v1/ingest/conversations

Accepts normalized conversation batches from the browser extension.
Transforms, dedupes, and saves to conversations table.
No LLM calls. Pure data ingest.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

import asyncpg
from fastapi import APIRouter, Depends, Request
from gubbi_common.db.user_scoped import user_scoped_connection
from pydantic import BaseModel, Field

from journalctl.api.v1.auth import require_scope
from journalctl.core.cipher_guard import require_cipher
from journalctl.core.context import AppContext
from journalctl.core.validation import validate_title
from journalctl.models.conversation import Message
from journalctl.storage.exceptions import TopicNotFoundError
from journalctl.storage.repositories import conversations as conv_repo
from journalctl.storage.repositories.topics import create as create_topic
from journalctl.storage.repositories.topics import get_id as get_topic_id

logger = logging.getLogger(__name__)

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

    Populated during lifespan by main.py.
    """
    ctx: object = request.app.state.app_ctx
    if not isinstance(ctx, AppContext):
        raise RuntimeError("app_ctx not set on application state or wrong type")
    return ctx


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

    conversations_saved = 0
    conversations_skipped_dedupe = 0
    superseded_json_paths: list[str] = []

    async with user_scoped_connection(app_ctx.pool, user_id=user_id) as conn:
        # Ensure the inbox topic exists before saving conversations
        try:
            await get_topic_id(conn, DEFAULT_INBOX_TOPIC)
        except TopicNotFoundError:
            await create_topic(conn, DEFAULT_INBOX_TOPIC, title="Inbox")

        for conv in body.conversations:
            existing = await conn.fetchval(
                "SELECT 1 FROM conversations"
                " WHERE user_id = $1 AND platform = $2 AND platform_id = $3",
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
                    await conn.execute(
                        "UPDATE conversations SET platform = $1, platform_id = $2 WHERE id = $3",
                        conv.platform,
                        conv.platform_id,
                        save_result.conversation_id,
                    )
            except asyncpg.UniqueViolationError:
                logger.warning(
                    "Dedupe race: platform_id already exists, treating as skip",
                    extra={"platform": conv.platform, "platform_id": conv.platform_id},
                )
                conversations_skipped_dedupe += 1
                continue

            if save_result.superseded_json_path is not None:
                superseded_json_paths.append(save_result.superseded_json_path)
            conversations_saved += 1

    for path in superseded_json_paths:
        conv_repo.delete_superseded_json_archive(app_ctx.settings.conversations_json_dir, path)

    return IngestConversationResponse(
        conversations_saved=conversations_saved,
        conversations_skipped_dedupe=conversations_skipped_dedupe,
    )
