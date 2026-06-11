"""Topic management write endpoints: rename, delete-with-reassign, merge.

``POST   /api/v1/topics/{topic_id}/rename``  -- change a topic's path/title.
``DELETE /api/v1/topics/{topic_id}``         -- reassign rows, then delete.
``POST   /api/v1/topics/{topic_id}/merge``   -- fold a topic into another.

Companion to the read router (``web/topics.py``): same idioms (one
``safe_user_scoped_connection`` per request, plain Pydantic bodies, ``errors``
not-found mapping, ``bound_logger``), but every endpoint requires the
``journal:write`` scope and wraps the repository mutation in a single
transaction. The mutation + its audit row commit together or not at all.

Status codes:
* 404 ``{"detail": "topic_not_found"}`` -- a referenced topic does not resolve.
* 409 ``{"detail": "topic_path_exists"}`` -- rename collides with an existing
  path, or a merge/reassign collides on a conversation slug.
* 400 -- a delete needs a destination for its entries but none was given.
"""

from __future__ import annotations

# UUID is a FastAPI Depends() Annotated type, resolved at route registration,
# so it stays a runtime import despite ``from __future__ import annotations``.
from typing import Annotated
from uuid import UUID  # noqa: TC003

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from gubbi_common.telemetry import bound_logger
from pydantic import BaseModel

from gubbi.api.v1.auth import require_scope
from gubbi.api.v1.web.errors import topic_not_found
from gubbi.app_state import require_app_ctx
from gubbi.storage.connection import safe_user_scoped_connection
from gubbi.storage.exceptions import TopicNotFoundError
from gubbi.storage.repositories import topics as topics_repo
from gubbi.storage.repositories.topics import TopicAlreadyExists, TopicMergeConflict

__all__: list[str] = [
    "MergeRequest",
    "RenameRequest",
    "TopicMutationResponse",
    "delete_topic",
    "merge_topic",
    "rename_topic",
    "router",
]

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/topics", tags=["topics"])

# Stable machine-readable detail codes (clients match on these, not prose).
TOPIC_PATH_EXISTS: str = "topic_path_exists"


class RenameRequest(BaseModel):
    """``POST /topics/{id}/rename`` body: new path and optional new title."""

    path: str
    title: str | None = None


class MergeRequest(BaseModel):
    """``POST /topics/{id}/merge`` body: the destination topic id."""

    into_topic_id: int


class TopicMutationResponse(BaseModel):
    """Result of a topic mutation: how many rows moved, by kind."""

    entries_moved: int
    conversations_moved: int


def _path_exists_409() -> HTTPException:
    """Return 409 ``{"detail": "topic_path_exists"}`` for a path/slug collision."""
    return HTTPException(status_code=409, detail=TOPIC_PATH_EXISTS)


@router.post("/{topic_id}/rename", response_model=TopicMutationResponse)
async def rename_topic(
    request: Request,
    topic_id: int,
    body: RenameRequest,
    auth: Annotated[tuple[UUID, frozenset[str]], Depends(require_scope("journal:write"))],
) -> Response:
    """POST /api/v1/topics/{topic_id}/rename.

    409 ``topic_path_exists`` when the new path collides with another of the
    user's topics; 404 ``topic_not_found`` when the id does not resolve.
    """
    user_id, _scopes = auth
    app_ctx = require_app_ctx(request)
    log = bound_logger(request)

    async with safe_user_scoped_connection(app_ctx.pool, user_id=user_id) as conn:
        try:
            async with conn.transaction():
                await topics_repo.rename(
                    conn,
                    topic_id,
                    body.path,
                    body.title,
                    actor_id=str(user_id),
                )
        except TopicNotFoundError:
            raise topic_not_found() from None
        except TopicAlreadyExists:
            raise _path_exists_409() from None
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None

    await log.info("web_topic_rename", topic_id=topic_id)
    body_out = TopicMutationResponse(entries_moved=0, conversations_moved=0)
    return Response(
        status_code=200,
        media_type="application/json",
        content=body_out.model_dump_json(),
    )


@router.delete("/{topic_id}", response_model=TopicMutationResponse)
async def delete_topic(
    request: Request,
    topic_id: int,
    auth: Annotated[tuple[UUID, frozenset[str]], Depends(require_scope("journal:write"))],
    move_entries_to: int | None = None,
) -> Response:
    """DELETE /api/v1/topics/{topic_id}?move_entries_to={dest_topic_id}.

    Reassigns the topic's entries + conversations to ``move_entries_to``, then
    deletes the topic. 400 when the topic has rows but no destination was given
    (detail states how many entries need one); 404 when the source or
    destination does not resolve; 409 on a conversation slug collision.
    """
    user_id, _scopes = auth
    app_ctx = require_app_ctx(request)
    log = bound_logger(request)

    async with safe_user_scoped_connection(app_ctx.pool, user_id=user_id) as conn:
        try:
            async with conn.transaction():
                moved = await topics_repo.delete_with_reassign(
                    conn,
                    topic_id,
                    move_entries_to,
                    actor_id=str(user_id),
                )
        except TopicNotFoundError:
            raise topic_not_found() from None
        except TopicMergeConflict:
            raise _path_exists_409() from None
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None

    await log.info(
        "web_topic_delete",
        topic_id=topic_id,
        entries_moved=moved.entries,
        conversations_moved=moved.conversations,
    )
    body = TopicMutationResponse(
        entries_moved=moved.entries,
        conversations_moved=moved.conversations,
    )
    return Response(status_code=200, media_type="application/json", content=body.model_dump_json())


@router.post("/{topic_id}/merge", response_model=TopicMutationResponse)
async def merge_topic(
    request: Request,
    topic_id: int,
    body: MergeRequest,
    auth: Annotated[tuple[UUID, frozenset[str]], Depends(require_scope("journal:write"))],
) -> Response:
    """POST /api/v1/topics/{topic_id}/merge.

    Moves all entries + conversations from ``topic_id`` into
    ``into_topic_id``, then deletes the source. 422 on a self-merge; 404 when
    either topic does not resolve; 409 on a conversation slug collision.
    """
    user_id, _scopes = auth
    app_ctx = require_app_ctx(request)
    log = bound_logger(request)

    async with safe_user_scoped_connection(app_ctx.pool, user_id=user_id) as conn:
        try:
            async with conn.transaction():
                moved = await topics_repo.merge(
                    conn,
                    topic_id,
                    body.into_topic_id,
                    actor_id=str(user_id),
                )
        except TopicNotFoundError:
            raise topic_not_found() from None
        except TopicMergeConflict:
            raise _path_exists_409() from None
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None

    await log.info(
        "web_topic_merge",
        topic_id=topic_id,
        into_topic_id=body.into_topic_id,
        entries_moved=moved.entries,
        conversations_moved=moved.conversations,
    )
    resp_body = TopicMutationResponse(
        entries_moved=moved.entries,
        conversations_moved=moved.conversations,
    )
    return Response(
        status_code=200, media_type="application/json", content=resp_body.model_dump_json()
    )
