"""Entry mutation endpoints: update, soft-delete, and bulk move.

``PATCH  /api/v1/entries/{id}``    -- replace-semantics partial update.
``DELETE /api/v1/entries/{id}``    -- soft-delete (idempotent 404 once gone).
``POST   /api/v1/entries/move``    -- bulk-move up to 100 entries to a topic.

Companion to the read router (``web/entries.py``): same idioms (one
``safe_user_scoped_connection`` per request, plain Pydantic bodies, ``errors``
not-found mapping, ``bound_logger``), but every endpoint requires the
``journal:write`` scope and wraps the repository mutation in a single
transaction so the change and any audit row commit together or not at all.

PATCH delegates to the same ``entries.update`` logic the MCP ``journal_update_entry``
tool uses -- always replace-mode (REST PATCH semantics), re-embedding when
content or reasoning changes. A ``topic_path`` in the body moves the entry via
``move_entries_to_topic`` in the same transaction, which writes the ``entry.moved``
audit row. PATCH returns the updated detail shape.
"""

from __future__ import annotations

import asyncio

# UUID is a FastAPI Depends() Annotated type, resolved at route registration, so
# it stays a runtime import despite the future-annotations import.
from typing import TYPE_CHECKING, Annotated
from uuid import UUID  # noqa: TC003

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from gubbi_common.telemetry import bound_logger
from pydantic import BaseModel

from gubbi.api.v1.auth import require_scope
from gubbi.api.v1.web.entries import EntryDetailResponse, row_to_detail
from gubbi.api.v1.web.errors import entry_not_found
from gubbi.api.v1.web.responses import private_no_store_response
from gubbi.app_state import require_app_ctx
from gubbi.crypto.guard import require_cipher
from gubbi.storage.connection import safe_user_scoped_connection
from gubbi.storage.exceptions import EntryNotFoundError, TopicNotFoundError
from gubbi.storage.repositories import entries as entries_repo
from gubbi.storage.repositories import topics as topics_repo
from gubbi.storage.repositories.entries import MAX_MOVE_IDS, EntriesMoveNotFound

if TYPE_CHECKING:
    from gubbi.crypto.cipher import ContentCipher

__all__: list[str] = [
    "EntryMoveRequest",
    "EntryMoveResponse",
    "EntryUpdateRequest",
    "delete_entry",
    "move_entries",
    "router",
    "update_entry",
]

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/entries", tags=["entries"])

# Stable machine-readable detail codes (clients match on these, not prose).
ENTRIES_MOVE_NOT_FOUND: str = "entries_not_found"


class EntryUpdateRequest(BaseModel):
    """``PATCH /entries/{id}`` body. All fields optional; at least one required.

    ``topic_path`` moves the entry to that topic. Replace semantics throughout:
    a present field overwrites; ``tags: []`` clears all tags.
    """

    content: str | None = None
    reasoning: str | None = None
    date: str | None = None
    tags: list[str] | None = None
    topic_path: str | None = None


class EntryMoveRequest(BaseModel):
    """``POST /entries/move`` body: ids to move and the destination topic id."""

    entry_ids: list[int]
    topic_id: int


class EntryMoveResponse(BaseModel):
    """Result of a bulk move: how many entries were reassigned."""

    entries_moved: int


def _has_update_field(body: EntryUpdateRequest) -> bool:
    """True when the body carries at least one field to change."""
    return any(
        v is not None for v in (body.content, body.reasoning, body.date, body.tags, body.topic_path)
    )


async def _reembed_if_changed(
    request: Request,
    user_id: UUID,
    cipher: ContentCipher,
    entry_id: int,
    *,
    content_changed: bool,
    reasoning_changed: bool,
) -> None:
    """Best-effort re-embed after a content/reasoning change.

    Mirrors the tool layer: encode outside the connection, then store + mark in
    one round-trip, skipping if the row changed again or was deleted meanwhile.
    A failure leaves the entry FTS-only (the stale embedding was already deleted
    inside ``entries.update``), never surfacing old content.
    """
    if not (content_changed or reasoning_changed):
        return
    app_ctx = require_app_ctx(request)
    log = bound_logger(request)
    try:
        async with safe_user_scoped_connection(app_ctx.pool, user_id=user_id) as conn:
            row_data = await entries_repo.get_text(conn, cipher, entry_id)
        if row_data is None:
            return
        embed_text = ((row_data[0] or "") + " " + (row_data[1] or "")).strip()
        embedding = await asyncio.to_thread(app_ctx.embedding_service.encode, embed_text)
        async with safe_user_scoped_connection(app_ctx.pool, user_id=user_id) as conn:
            current = await entries_repo.get_text(conn, cipher, entry_id)
            if current != row_data:
                return
            await app_ctx.embedding_service.save_by_vector(conn, entry_id, embedding)
            await entries_repo.mark_indexed(conn, entry_id)
    except Exception as exc:  # best-effort; never fail the request
        await log.error(
            "web_entry_reembed_failed",
            entry_id=entry_id,
            error=str(exc),
            error_type=type(exc).__name__,
        )


@router.patch("/{entry_id}", response_model=EntryDetailResponse)
async def update_entry(
    request: Request,
    entry_id: int,
    body: EntryUpdateRequest,
    auth: Annotated[tuple[UUID, frozenset[str]], Depends(require_scope("journal:write"))],
) -> Response:
    """PATCH /api/v1/entries/{id}.

    Applies a replace-mode partial update; ``topic_path`` moves the entry. 422
    when no field is supplied; 404 ``entry_not_found`` when the id does not
    resolve; 404 ``topic_not_found`` when ``topic_path`` does not resolve.
    Returns the updated detail shape.

    Cache-Control is ``private, no-store``: entry content is per-user.
    """
    user_id, _scopes = auth
    app_ctx = require_app_ctx(request)
    cipher = require_cipher(app_ctx)
    log = bound_logger(request)

    if not _has_update_field(body):
        raise HTTPException(status_code=422, detail="no_fields_to_update")

    content_changed = body.content is not None
    reasoning_changed = body.reasoning is not None
    field_update = (
        content_changed or reasoning_changed or body.date is not None or body.tags is not None
    )

    async with safe_user_scoped_connection(app_ctx.pool, user_id=user_id) as conn:
        try:
            async with conn.transaction():
                if field_update:
                    await entries_repo.update(
                        conn,
                        cipher,
                        entry_id=entry_id,
                        content=body.content,
                        reasoning=body.reasoning,
                        mode="replace",
                        date=body.date,
                        tags=body.tags,
                    )
                if body.topic_path is not None:
                    dest_id = await topics_repo.get_id(conn, body.topic_path)
                    await entries_repo.move_entries_to_topic(
                        conn, [entry_id], dest_id, actor_id=str(user_id)
                    )
        except EntryNotFoundError:
            raise entry_not_found() from None
        except EntriesMoveNotFound:
            raise entry_not_found() from None
        except TopicNotFoundError:
            raise HTTPException(status_code=404, detail="topic_not_found") from None
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None

    await _reembed_if_changed(
        request,
        user_id,
        cipher,
        entry_id,
        content_changed=content_changed,
        reasoning_changed=reasoning_changed,
    )

    async with safe_user_scoped_connection(app_ctx.pool, user_id=user_id) as conn:
        row = await entries_repo.get_entry_by_id(conn, entry_id)
    if row is None:
        raise entry_not_found()

    await log.info("web_entry_update", entry_id=entry_id)
    return private_no_store_response(row_to_detail(cipher, row))


@router.delete("/{entry_id}", status_code=204)
async def delete_entry(
    request: Request,
    entry_id: int,
    auth: Annotated[tuple[UUID, frozenset[str]], Depends(require_scope("journal:write"))],
) -> Response:
    """DELETE /api/v1/entries/{id}.

    Soft-deletes the entry (and drops its embedding). 204 on success; 404
    ``entry_not_found`` when the id does not resolve or is already deleted
    (idempotent: a second delete returns 404, not 204).
    """
    user_id, _scopes = auth
    app_ctx = require_app_ctx(request)
    log = bound_logger(request)

    async with safe_user_scoped_connection(app_ctx.pool, user_id=user_id) as conn:
        try:
            async with conn.transaction():
                await entries_repo.delete(conn, entry_id)
        except EntryNotFoundError:
            raise entry_not_found() from None

    await log.info("web_entry_delete", entry_id=entry_id)
    return Response(status_code=204)


@router.post("/move", response_model=EntryMoveResponse)
async def move_entries(
    request: Request,
    body: EntryMoveRequest,
    auth: Annotated[tuple[UUID, frozenset[str]], Depends(require_scope("journal:write"))],
) -> Response:
    """POST /api/v1/entries/move.

    Bulk-moves up to 100 entries to ``topic_id`` in one transaction (all-or-
    nothing). 422 when more than 100 ids are supplied or the list is empty; 404
    ``entries_not_found`` (detail lists the missing ids) when any entry id does
    not resolve; 404 ``topic_not_found`` when the destination does not resolve.
    """
    user_id, _scopes = auth
    app_ctx = require_app_ctx(request)
    log = bound_logger(request)

    if len(body.entry_ids) > MAX_MOVE_IDS:
        raise HTTPException(
            status_code=422,
            detail=f"cannot move more than {MAX_MOVE_IDS} entries at once",
        )

    async with safe_user_scoped_connection(app_ctx.pool, user_id=user_id) as conn:
        try:
            async with conn.transaction():
                moved = await entries_repo.move_entries_to_topic(
                    conn, body.entry_ids, body.topic_id, actor_id=str(user_id)
                )
        except EntriesMoveNotFound as exc:
            raise HTTPException(
                status_code=404,
                detail={"code": ENTRIES_MOVE_NOT_FOUND, "missing_ids": exc.missing_ids},
            ) from None
        except TopicNotFoundError:
            raise HTTPException(status_code=404, detail="topic_not_found") from None
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None

    await log.info("web_entries_move", entries_moved=moved, topic_id=body.topic_id)
    body_out = EntryMoveResponse(entries_moved=moved)
    return Response(
        status_code=200,
        media_type="application/json",
        content=body_out.model_dump_json(),
    )
