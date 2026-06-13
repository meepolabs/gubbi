"""Topics read endpoints: list and detail.

``GET /api/v1/topics``         -- paginated topic list, optional path prefix.
``GET /api/v1/topics/{path}``  -- single topic metadata by path; 404 if absent.

Reference implementation for the web REST surface. Sibling resource routers
copy this shape: declare a per-item model, compose ``PaginatedList`` for the
list response, parse pagination via the shared ``pagination`` builders, map
not-found via ``errors``, and (for any decrypted content) return through
``private_no_store_response`` after mapping each ciphertext column with
``decrypt_field``. Topics carry no encrypted columns, so this router shows the
no-decryption path; it still returns ``private, no-store`` because topic
metadata is per-user private content.
"""

from __future__ import annotations

# datetime is a Pydantic field type on TopicItem; UUID is a FastAPI Depends()
# Annotated type. Both resolve at runtime (validator-build / route
# registration), so they stay runtime imports despite
# ``from __future__ import annotations`` -- ruff TC would otherwise push them
# under TYPE_CHECKING and break Pydantic model_rebuild().
from typing import TYPE_CHECKING, Annotated
from uuid import UUID  # noqa: TC003

import structlog
from fastapi import APIRouter, Depends, Query, Request, Response
from gubbi_common.telemetry import bound_logger
from pydantic import BaseModel, Field

from gubbi.api.v1.auth import require_scope
from gubbi.api.v1.web.errors import invalid_filter, topic_not_found
from gubbi.api.v1.web.pagination import OffsetQuery  # noqa: TC001
from gubbi.api.v1.web.responses import private_no_store_response
from gubbi.api.v1.web.schemas import PaginatedList
from gubbi.app_state import require_app_ctx
from gubbi.storage.connection import safe_user_scoped_connection
from gubbi.storage.repositories import topics as topics_repo
from gubbi.validation import validate_topic

if TYPE_CHECKING:
    from gubbi.models.journal import TopicMeta

__all__: list[str] = [
    "DEFAULT_TOPICS_LIMIT",
    "MAX_TOPICS_LIMIT",
    "TopicItem",
    "TopicListResponse",
    "get_topic",
    "list_topics",
    "router",
]

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/topics", tags=["topics"])

# Web-surface pagination contract (raised above the tool-layer caps per spec).
DEFAULT_TOPICS_LIMIT: int = 50
MAX_TOPICS_LIMIT: int = 200

LimitParam = Annotated[int, Query(ge=1, le=MAX_TOPICS_LIMIT, description="Page size (max 200).")]


class TopicItem(BaseModel):
    """A single topic's metadata, mapped from ``TopicMeta``.

    ``path`` is the topic's slash-path; ``entry_count`` excludes soft-deleted
    entries. ``created_at`` / ``updated_at`` are date strings (YYYY-MM-DD) as
    the repository returns them.
    """

    id: int
    path: str
    title: str
    description: str
    entry_count: int
    created_at: str
    updated_at: str


class TopicListResponse(PaginatedList[TopicItem]):
    """``GET /api/v1/topics`` body: ``{topics, total, limit, offset}``."""

    items: list[TopicItem] = Field(serialization_alias="topics")


def _to_item(meta: TopicMeta) -> TopicItem:
    """Map a repository ``TopicMeta`` to the API item shape.

    ``TopicMeta.id`` is typed ``int | None``, but every row read from the DB
    carries a primary key. A ``None`` here means a schema invariant was
    violated, so fail loudly rather than emit a zero that looks like a real id.
    """
    if meta.id is None:
        raise RuntimeError(f"Topic '{meta.topic}' has no database id")
    return TopicItem(
        id=meta.id,
        path=meta.topic,
        title=meta.title,
        description=meta.description,
        entry_count=meta.entry_count,
        created_at=meta.created,
        updated_at=meta.updated,
    )


@router.get("", response_model=TopicListResponse)
async def list_topics(
    request: Request,
    auth: Annotated[tuple[UUID, frozenset[str]], Depends(require_scope("journal:read"))],
    limit: LimitParam = DEFAULT_TOPICS_LIMIT,
    offset: OffsetQuery = 0,
    prefix: str | None = None,
) -> Response:
    """GET /api/v1/topics.

    Lists the authenticated user's topics, newest-updated first, optionally
    filtered to those whose path starts with ``prefix``. ``total`` is the full
    filtered count before paging.

    Cache-Control is ``private, no-store``: topic metadata is per-user content.
    """
    user_id, _scopes = auth
    app_ctx = require_app_ctx(request)
    log = bound_logger(request)

    async with safe_user_scoped_connection(app_ctx.pool, user_id=user_id) as conn:
        try:
            metas, total = await topics_repo.list_all(
                conn,
                topic_prefix=prefix,
                limit=limit,
                offset=offset,
            )
        except ValueError as exc:
            raise invalid_filter(str(exc)) from None

    await log.info("web_topics_list", result_count=len(metas), total=total)
    body = TopicListResponse(
        items=[_to_item(m) for m in metas],
        total=total,
        limit=limit,
        offset=offset,
    )
    return private_no_store_response(body)


@router.get("/{path:path}", response_model=TopicItem)
async def get_topic(
    request: Request,
    path: str,
    auth: Annotated[tuple[UUID, frozenset[str]], Depends(require_scope("journal:read"))],
) -> Response:
    """GET /api/v1/topics/{path}.

    Returns one topic's metadata by path. 404 ``{"detail": "topic_not_found"}``
    when the path does not resolve for this user -- which, under RLS, also
    covers another user's topic (no cross-tenant existence signal). A
    syntactically invalid path likewise cannot name an existing topic, so it
    maps to the same 404 rather than a 422.

    Cache-Control is ``private, no-store``: topic metadata is per-user content.
    """
    user_id, _scopes = auth
    app_ctx = require_app_ctx(request)
    log = bound_logger(request)

    try:
        validated_path = validate_topic(path)
    except ValueError:
        # Invalid topic syntax cannot name an existing topic.
        raise topic_not_found() from None

    async with safe_user_scoped_connection(app_ctx.pool, user_id=user_id) as conn:
        meta = await topics_repo.get(conn, validated_path)

    if meta is None:
        raise topic_not_found()

    await log.info("web_topics_get")
    return private_no_store_response(_to_item(meta))
