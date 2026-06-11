"""Entries read endpoints: cross-topic list and single-entry detail.

``GET /api/v1/entries``       -- paginated cross-topic entry list with filters.
``GET /api/v1/entries/{id}``  -- one entry's full detail (adds ``reasoning``);
404 if the id does not resolve for the user.

Mirrors the topics/conversations reference routers: a per-item model, a
``PaginatedList`` subclass for the list shape, the shared ``pagination``
builders for ``limit`` / ``offset``, ``errors`` for 404 mapping, and
``private_no_store_response`` because the body carries decrypted content.

Decryption: every encrypted column routes through the foundation's
``decrypt_field`` so a single corrupt row yields the ``[decryption failed]``
sentinel + ``decryption_failed: true`` rather than a 500. The list view omits
``reasoning`` (detail-only, matching the product model: reasoning loads on full
read); the detail view adds it. The shared ``EntryItem`` model and its mapper
are reused by the mutation router (``entries_admin``) so PATCH returns the same
detail shape.
"""

from __future__ import annotations

# UUID is a FastAPI Depends() Annotated type that must resolve at route
# registration, so it stays a runtime import despite the future-annotations
# import (ruff TC would otherwise hide it under TYPE_CHECKING).
from typing import TYPE_CHECKING, Annotated
from uuid import UUID  # noqa: TC003

import structlog
from fastapi import APIRouter, Depends, Query, Request, Response
from gubbi_common.telemetry import bound_logger
from pydantic import Field

from gubbi.api.v1.auth import require_scope
from gubbi.api.v1.web.decryption import decrypt_field
from gubbi.api.v1.web.errors import entry_not_found
from gubbi.api.v1.web.pagination import OffsetQuery  # noqa: TC001
from gubbi.api.v1.web.responses import private_no_store_response
from gubbi.api.v1.web.schemas import DecryptableItem, PaginatedList
from gubbi.app_state import require_app_ctx
from gubbi.crypto.guard import require_cipher
from gubbi.storage.connection import safe_user_scoped_connection
from gubbi.storage.repositories import entries as entries_repo

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    import asyncpg

    from gubbi.crypto.cipher import ContentCipher

__all__: list[str] = [
    "DEFAULT_ENTRIES_LIMIT",
    "MAX_ENTRIES_LIMIT",
    "EntryDetailResponse",
    "EntryItem",
    "EntryListResponse",
    "get_entry",
    "list_entries",
    "router",
    "row_to_detail",
    "row_to_item",
]

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/entries", tags=["entries"])

# Web-surface pagination contract (raised above the tool-layer caps per spec).
DEFAULT_ENTRIES_LIMIT: int = 20
MAX_ENTRIES_LIMIT: int = 100

LimitParam = Annotated[int, Query(ge=1, le=MAX_ENTRIES_LIMIT, description="Page size (max 100).")]
TagsParam = Annotated[
    list[str] | None,
    Query(description="Repeat to AND-filter: a row must carry every given tag."),
]
SortParam = Annotated[
    str,
    Query(pattern="^(newest|oldest)$", description="newest (default) | oldest."),
]


class EntryItem(DecryptableItem):
    """A single entry's list-view shape (no ``reasoning``).

    ``content`` arrives decrypted; on a decryption failure it holds the
    ``[decryption failed]`` sentinel and ``decryption_failed`` is ``True``.
    ``date`` is a YYYY-MM-DD string; ``created_at`` / ``updated_at`` are ISO
    timestamps as the repository returns them.
    """

    id: int
    topic_path: str
    date: str
    content: str
    tags: list[str]
    conversation_id: int | None
    created_at: str
    updated_at: str


class EntryDetailResponse(EntryItem):
    """``GET /api/v1/entries/{id}`` body: the list item plus ``reasoning``.

    ``reasoning`` is the decrypted reasoning text, or ``null`` when the entry has
    none. A reasoning decryption failure surfaces the sentinel and flips
    ``decryption_failed`` (shared with the content flag -- one bad column marks
    the item).
    """

    reasoning: str | None


class EntryListResponse(PaginatedList[EntryItem]):
    """``GET /api/v1/entries`` body: ``{entries, total, limit, offset}``."""

    items: list[EntryItem] = Field(serialization_alias="entries")


def _iso(value: Any) -> str:
    """Render a timestamp/date column as an ISO string."""
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def row_to_item(
    cipher: ContentCipher,
    row: asyncpg.Record | Mapping[str, Any],
) -> EntryItem:
    """Map a repository row to the list-item shape, decrypting ``content``."""
    content, failed = decrypt_field(cipher, row, "content_encrypted", "content_nonce")
    return EntryItem(
        id=row["id"],
        topic_path=row["topic_path"],
        date=str(row["date"]),
        content=content if content is not None else "",
        tags=list(row["tags"] or []),
        conversation_id=row["conversation_id"],
        created_at=_iso(row["created_at"]),
        updated_at=_iso(row["updated_at"]),
        decryption_failed=failed,
    )


def row_to_detail(
    cipher: ContentCipher,
    row: asyncpg.Record | Mapping[str, Any],
) -> EntryDetailResponse:
    """Map a repository row to the detail shape, decrypting content + reasoning."""
    content, content_failed = decrypt_field(cipher, row, "content_encrypted", "content_nonce")
    reasoning, reasoning_failed = decrypt_field(
        cipher, row, "reasoning_encrypted", "reasoning_nonce"
    )
    return EntryDetailResponse(
        id=row["id"],
        topic_path=row["topic_path"],
        date=str(row["date"]),
        content=content if content is not None else "",
        reasoning=reasoning,
        tags=list(row["tags"] or []),
        conversation_id=row["conversation_id"],
        created_at=_iso(row["created_at"]),
        updated_at=_iso(row["updated_at"]),
        decryption_failed=content_failed or reasoning_failed,
    )


@router.get("", response_model=EntryListResponse)
async def list_entries(
    request: Request,
    auth: Annotated[tuple[UUID, frozenset[str]], Depends(require_scope("journal:read"))],
    limit: LimitParam = DEFAULT_ENTRIES_LIMIT,
    offset: OffsetQuery = 0,
    topic: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    tags: TagsParam = None,
    source: str | None = None,
    sort: SortParam = "newest",
) -> Response:
    """GET /api/v1/entries.

    Lists the authenticated user's entries across topics (or one ``topic``),
    filtered by date range, tags (AND), and conversation ``source``, ordered by
    ``sort``. ``total`` is the full filtered count before paging. List items omit
    ``reasoning`` -- it loads only on the detail endpoint.

    Cache-Control is ``private, no-store``: entry content is per-user.
    """
    user_id, _scopes = auth
    app_ctx = require_app_ctx(request)
    cipher = require_cipher(app_ctx)
    log = bound_logger(request)

    async with safe_user_scoped_connection(app_ctx.pool, user_id=user_id) as conn:
        rows, total = await entries_repo.list_entries(
            conn,
            topic=topic,
            date_from=date_from,
            date_to=date_to,
            tags=tags,
            source=source,
            sort=sort,
            limit=limit,
            offset=offset,
        )

    await log.info("web_entries_list", result_count=len(rows), total=total)
    body = EntryListResponse(
        items=[row_to_item(cipher, r) for r in rows],
        total=total,
        limit=limit,
        offset=offset,
    )
    return private_no_store_response(body)


@router.get("/{entry_id}", response_model=EntryDetailResponse)
async def get_entry(
    request: Request,
    entry_id: int,
    auth: Annotated[tuple[UUID, frozenset[str]], Depends(require_scope("journal:read"))],
) -> Response:
    """GET /api/v1/entries/{id}.

    Returns one entry's full detail (list shape plus decrypted ``reasoning``).
    404 ``{"detail": "entry_not_found"}`` when the id does not resolve for this
    user -- absent, soft-deleted, or (under RLS) another user's entry.

    Cache-Control is ``private, no-store``: entry content is per-user.
    """
    user_id, _scopes = auth
    app_ctx = require_app_ctx(request)
    cipher = require_cipher(app_ctx)
    log = bound_logger(request)

    async with safe_user_scoped_connection(app_ctx.pool, user_id=user_id) as conn:
        row = await entries_repo.get_entry_by_id(conn, entry_id)

    if row is None:
        raise entry_not_found()

    await log.info("web_entries_get")
    return private_no_store_response(row_to_detail(cipher, row))
