"""Dashboard stats endpoint.

``GET /api/v1/stats`` -- one round-trip aggregate for the dashboard strip:
topic/entry/conversation totals, this-week entry count (in the user's
timezone), the most recent entry timestamp, and the top-5 most-active topics
over the last 30 days.

Mirrors the ``topics`` reference router: ``require_scope("journal:read")``,
an RLS-scoped connection, a plain Pydantic response, and a ``private,
no-store`` Cache-Control header (the body is per-user content).
"""

from __future__ import annotations

# datetime is a Pydantic field type on StatsResponse; UUID is a FastAPI
# Depends() Annotated type. Both resolve at runtime, so they stay runtime
# imports despite ``from __future__ import annotations``.
from datetime import datetime  # noqa: TC003
from typing import Annotated
from uuid import UUID  # noqa: TC003

import structlog
from fastapi import APIRouter, Depends, Request, Response
from gubbi_common.telemetry import bound_logger
from pydantic import BaseModel

from gubbi.api.v1.auth import require_scope
from gubbi.api.v1.web.responses import private_no_store_response
from gubbi.app_state import require_app_ctx
from gubbi.storage.connection import safe_user_scoped_connection
from gubbi.storage.repositories import stats as stats_repo

__all__: list[str] = [
    "MostActiveTopicItem",
    "StatsResponse",
    "get_stats",
    "router",
]

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/stats", tags=["stats"])


class MostActiveTopicItem(BaseModel):
    """A top-active topic: path, title, and entry count over the last 30 days."""

    path: str
    title: str
    entries_last_30d: int


class StatsResponse(BaseModel):
    """``GET /api/v1/stats`` body.

    ``last_entry_at`` is null when the journal has no entries. ``entries_this_week``
    counts entries dated within the current Monday-start week in the user's
    timezone.
    """

    topics_total: int
    entries_total: int
    entries_this_week: int
    conversations_total: int
    last_entry_at: datetime | None
    most_active_topics: list[MostActiveTopicItem]


def _to_response(stats: stats_repo.DashboardStats) -> StatsResponse:
    """Map the repository aggregate to the API response shape."""
    return StatsResponse(
        topics_total=stats.topics_total,
        entries_total=stats.entries_total,
        entries_this_week=stats.entries_this_week,
        conversations_total=stats.conversations_total,
        last_entry_at=stats.last_entry_at,
        most_active_topics=[
            MostActiveTopicItem(
                path=t.path,
                title=t.title,
                entries_last_30d=t.entries_last_30d,
            )
            for t in stats.most_active_topics
        ],
    )


@router.get("", response_model=StatsResponse)
async def get_stats(
    request: Request,
    auth: Annotated[tuple[UUID, frozenset[str]], Depends(require_scope("journal:read"))],
) -> Response:
    """GET /api/v1/stats.

    Returns the authenticated user's dashboard aggregate in one round-trip.

    Cache-Control is ``private, no-store``: the counts are per-user content.
    """
    user_id, _scopes = auth
    app_ctx = require_app_ctx(request)
    log = bound_logger(request)

    async with safe_user_scoped_connection(app_ctx.pool, user_id=user_id) as conn:
        stats = await stats_repo.get_dashboard_stats(conn)

    await log.info(
        "web_stats_get",
        topics_total=stats.topics_total,
        entries_total=stats.entries_total,
    )
    return private_no_store_response(_to_response(stats))
