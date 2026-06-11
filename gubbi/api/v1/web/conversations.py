"""Conversations read endpoints: list and detail.

``GET /api/v1/conversations``       -- paginated conversation list.
``GET /api/v1/conversations/{id}``  -- one conversation's meta + a page of
its messages; 404 if the id does not resolve for the user.

Mirrors the topics reference router: a per-item model, ``PaginatedList`` for
the list shape, the shared ``pagination`` builders for ``limit`` / ``offset``,
``errors`` for 404 mapping, and ``private_no_store_response`` because the body
carries decrypted content (titles, summaries, message text).

Decryption note: the conversation repository functions reused here
(``list_conversations`` / ``read_conversation_by_id_paginated``) decrypt
title, summary, and the requested message page internally and return plaintext
on ``ConversationMeta`` / ``Message``. They do NOT emit the entries repo's
``[decryption-failed]`` hyphen sentinel -- a corrupt row raises inside the repo
instead. So nothing re-runs ``decrypt_field`` here, and ``decryption_failed``
is carried (per the web ``DecryptableItem`` convention) but stays ``False`` on
these paths until the repo grows a soft-fail sentinel of its own.
"""

from __future__ import annotations

# datetime is not used directly; created_at/updated_at arrive as date strings
# from the repo. UUID is a FastAPI Depends() Annotated type that must resolve at
# route registration, so it stays a runtime import despite the future-annotations
# import (ruff TC would otherwise hide it under TYPE_CHECKING).
from typing import TYPE_CHECKING, Annotated
from uuid import UUID  # noqa: TC003

import structlog
from fastapi import APIRouter, Depends, Query, Request, Response
from gubbi_common.telemetry import bound_logger
from pydantic import BaseModel, Field

from gubbi.api.v1.auth import require_scope
from gubbi.api.v1.web.errors import conversation_not_found
from gubbi.api.v1.web.pagination import OffsetQuery  # noqa: TC001
from gubbi.api.v1.web.responses import private_no_store_response
from gubbi.api.v1.web.schemas import DecryptableItem, PaginatedList
from gubbi.app_state import require_app_ctx
from gubbi.crypto.guard import require_cipher
from gubbi.storage.connection import safe_user_scoped_connection
from gubbi.storage.exceptions import ConversationNotFoundError
from gubbi.storage.repositories import conversations as conv_repo

if TYPE_CHECKING:
    from gubbi.models.conversation import ConversationMeta, Message

__all__: list[str] = [
    "DEFAULT_CONVERSATIONS_LIMIT",
    "DEFAULT_MESSAGES_LIMIT",
    "MAX_CONVERSATIONS_LIMIT",
    "MAX_MESSAGES_LIMIT",
    "SUMMARY_PREVIEW_CHARS",
    "ConversationDetailResponse",
    "ConversationItem",
    "ConversationListResponse",
    "MessageItem",
    "get_conversation",
    "list_conversations",
    "router",
]

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/conversations", tags=["conversations"])

# Web-surface pagination contract (raised above the tool-layer caps per spec).
DEFAULT_CONVERSATIONS_LIMIT: int = 20
MAX_CONVERSATIONS_LIMIT: int = 100
DEFAULT_MESSAGES_LIMIT: int = 50
MAX_MESSAGES_LIMIT: int = 200

# Summaries are truncated in the list view; full text comes from the detail
# endpoint. The detail meta carries the untruncated summary.
SUMMARY_PREVIEW_CHARS: int = 280

LimitParam = Annotated[
    int,
    Query(ge=1, le=MAX_CONVERSATIONS_LIMIT, description="Page size (max 100)."),
]
MessagesLimitParam = Annotated[
    int,
    Query(ge=1, le=MAX_MESSAGES_LIMIT, description="Messages per page (max 200)."),
]


class ConversationItem(DecryptableItem):
    """A single conversation's metadata, mapped from ``ConversationMeta``.

    ``title`` and ``summary`` arrive decrypted from the repository. ``summary``
    is truncated to :data:`SUMMARY_PREVIEW_CHARS` in the list view. ``platform``
    is the originating platform when recorded; the repo's ``ConversationMeta``
    does not surface it, so it is ``None`` on this surface. ``created_at`` /
    ``updated_at`` are date strings (YYYY-MM-DD) as the repository returns them.
    """

    id: int
    topic_path: str
    title: str
    summary: str
    source: str
    platform: str | None
    message_count: int
    created_at: str
    updated_at: str


class ConversationListResponse(PaginatedList[ConversationItem]):
    """``GET /api/v1/conversations`` body: ``{conversations, total, ...}``."""

    items: list[ConversationItem] = Field(serialization_alias="conversations")


class MessageItem(DecryptableItem):
    """One message in a conversation, mapped from ``Message``.

    ``content`` arrives decrypted from the repository. ``position`` is the
    message's zero-based index in the full conversation (derived from the page
    offset), so a client can place a page within the whole transcript.
    """

    role: str
    content: str
    timestamp: str | None
    position: int


class ConversationDetailResponse(BaseModel):
    """``GET /api/v1/conversations/{id}`` body.

    ``conversation`` is the same meta shape as the list item. ``messages`` is
    one page of the transcript; ``messages_total`` is the full message count so
    the client can page without re-deriving it.
    """

    conversation: ConversationItem
    messages: list[MessageItem]
    messages_total: int
    messages_limit: int
    messages_offset: int


def _truncate_summary(summary: str) -> str:
    """Truncate a summary to the list-view preview length."""
    if len(summary) > SUMMARY_PREVIEW_CHARS:
        return summary[:SUMMARY_PREVIEW_CHARS]
    return summary


def _to_item(meta: ConversationMeta, *, truncate_summary: bool) -> ConversationItem:
    """Map a repository ``ConversationMeta`` to the API item shape.

    ``ConversationMeta.id`` is typed ``int | None``, but every row read from the
    DB carries a primary key. A ``None`` here means a schema invariant was
    violated, so fail loudly rather than emit a placeholder id.
    """
    if meta.id is None:
        raise RuntimeError(f"Conversation '{meta.title}' has no database id")
    summary = _truncate_summary(meta.summary) if truncate_summary else meta.summary
    return ConversationItem(
        id=meta.id,
        topic_path=meta.topic,
        title=meta.title,
        summary=summary,
        source=meta.source,
        platform=None,
        message_count=meta.message_count,
        created_at=meta.created,
        updated_at=meta.updated,
    )


def _to_message(message: Message, position: int) -> MessageItem:
    """Map a repository ``Message`` to the API message shape at ``position``."""
    return MessageItem(
        role=message.role,
        content=message.content,
        timestamp=message.timestamp,
        position=position,
    )


@router.get("", response_model=ConversationListResponse)
async def list_conversations(
    request: Request,
    auth: Annotated[tuple[UUID, frozenset[str]], Depends(require_scope("journal:read"))],
    limit: LimitParam = DEFAULT_CONVERSATIONS_LIMIT,
    offset: OffsetQuery = 0,
    topic_prefix: str | None = None,
) -> Response:
    """GET /api/v1/conversations.

    Lists the authenticated user's conversations, newest-created first,
    optionally filtered to those whose topic path starts with ``topic_prefix``.
    ``total`` is the full filtered count before paging. Each item's ``summary``
    is truncated to a preview length; the detail endpoint returns the full text.

    Cache-Control is ``private, no-store``: conversation content is per-user.
    """
    user_id, _scopes = auth
    app_ctx = require_app_ctx(request)
    cipher = require_cipher(app_ctx)
    log = bound_logger(request)

    async with safe_user_scoped_connection(app_ctx.pool, user_id=user_id) as conn:
        metas, total = await conv_repo.list_conversations(
            conn,
            cipher,
            topic_prefix=topic_prefix,
            limit=limit,
            offset=offset,
        )

    await log.info("web_conversations_list", result_count=len(metas), total=total)
    body = ConversationListResponse(
        items=[_to_item(m, truncate_summary=True) for m in metas],
        total=total,
        limit=limit,
        offset=offset,
    )
    return private_no_store_response(body)


@router.get("/{conversation_id}", response_model=ConversationDetailResponse)
async def get_conversation(
    request: Request,
    conversation_id: int,
    auth: Annotated[tuple[UUID, frozenset[str]], Depends(require_scope("journal:read"))],
    messages_limit: MessagesLimitParam = DEFAULT_MESSAGES_LIMIT,
    messages_offset: OffsetQuery = 0,
) -> Response:
    """GET /api/v1/conversations/{id}.

    Returns one conversation's metadata plus a page of its messages. Only the
    requested page is decrypted -- the repo pushes LIMIT/OFFSET into SQL.
    ``position`` on each message is its index in the full transcript.

    404 ``{"detail": "conversation_not_found"}`` when the id does not resolve
    for this user -- which, under RLS, also covers another user's conversation
    (no cross-tenant existence signal).

    Cache-Control is ``private, no-store``: conversation content is per-user.
    """
    user_id, _scopes = auth
    app_ctx = require_app_ctx(request)
    cipher = require_cipher(app_ctx)
    log = bound_logger(request)

    try:
        async with safe_user_scoped_connection(app_ctx.pool, user_id=user_id) as conn:
            meta, messages, messages_total = await conv_repo.read_conversation_by_id_paginated(
                conn,
                cipher,
                conversation_id,
                messages_limit=messages_limit,
                messages_offset=messages_offset,
            )
    except ConversationNotFoundError:
        raise conversation_not_found() from None

    await log.info("web_conversations_get", message_count=len(messages))
    body = ConversationDetailResponse(
        conversation=_to_item(meta, truncate_summary=False),
        messages=[_to_message(m, messages_offset + i) for i, m in enumerate(messages)],
        messages_total=messages_total,
        messages_limit=messages_limit,
        messages_offset=messages_offset,
    )
    return private_no_store_response(body)
