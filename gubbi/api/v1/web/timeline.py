"""Timeline read endpoint: per-bucket entry/conversation counts.

``GET /api/v1/timeline`` -- counts of entries and conversations grouped by
day or month over a required date range. A navigation index for the calendar
view; day click-through is served by
``GET /api/v1/entries?date_from=X&date_to=X``, not a separate endpoint.

Counts only -- the underlying repository call runs ``title_only=True`` so no
ciphertext is decrypted. The response still carries ``Cache-Control: private,
no-store`` because per-day counts are themselves per-user private content.

Follows the topics reference: ``require_scope("journal:read")``,
``safe_user_scoped_connection``, the shared validation helpers, and
``private_no_store_response``. The bucketing is a pure helper so it can be
unit-tested without a database.
"""

from __future__ import annotations

# Annotated/UUID resolve at route-registration time, so they stay runtime
# imports despite ``from __future__ import annotations`` (see topics.py).
from collections import defaultdict
from datetime import date as date_cls
from typing import TYPE_CHECKING, Annotated, Literal
from uuid import UUID  # noqa: TC003

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from gubbi_common.telemetry import bound_logger
from pydantic import BaseModel

from gubbi.api.v1.auth import require_scope
from gubbi.api.v1.web.responses import private_no_store_response
from gubbi.app_state import require_app_ctx
from gubbi.storage.connection import safe_user_scoped_connection
from gubbi.storage.repositories import entries as entries_repo
from gubbi.validation import validate_date, validate_topic

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__: list[str] = [
    "MAX_TIMELINE_SPAN_DAYS",
    "TimelineBucket",
    "TimelineResponse",
    "aggregate_buckets",
    "get_timeline",
    "router",
]

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/timeline", tags=["timeline"])

# Largest inclusive date span the endpoint will aggregate in one request.
# Exceeding it is a 422 -- the caller must page by narrower windows.
MAX_TIMELINE_SPAN_DAYS: int = 366

BucketUnit = Literal["day", "month"]


class TimelineBucket(BaseModel):
    """Counts for one calendar bucket (a day ``YYYY-MM-DD`` or a month ``YYYY-MM``)."""

    date: str
    entry_count: int
    conversation_count: int


class TimelineResponse(BaseModel):
    """``GET /api/v1/timeline`` body.

    ``buckets`` is sorted ascending by ``date``; ``bucket`` echoes the grouping
    granularity so the caller can label the axis without re-deriving it.
    """

    buckets: list[TimelineBucket]
    date_from: str
    date_to: str
    bucket: BucketUnit


def _bucket_key(row_date: str, bucket: BucketUnit) -> str:
    """Map a ``YYYY-MM-DD`` row date to its bucket key.

    ``day`` keeps the full date; ``month`` truncates to ``YYYY-MM``.
    """
    if bucket == "month":
        return row_date[:7]
    return row_date


def aggregate_buckets(
    rows: Sequence[dict[str, object]],
    bucket: BucketUnit,
) -> list[TimelineBucket]:
    """Group title-only date-range rows into per-bucket counts.

    Each row is a ``get_by_date_range(..., title_only=True)`` dict carrying a
    ``doc_type`` of ``"entry"`` or ``"conversation"`` and an ``"updated"`` date
    string. Rows are grouped by day or month; ``entry_count`` and
    ``conversation_count`` are tallied per bucket. Buckets with no rows do not
    appear (the calendar view renders gaps as empty). Returns buckets sorted
    ascending by key.
    """
    entry_counts: dict[str, int] = defaultdict(int)
    conv_counts: dict[str, int] = defaultdict(int)

    for row in rows:
        row_date = row.get("updated")
        if not isinstance(row_date, str) or not row_date:
            continue
        key = _bucket_key(row_date, bucket)
        if row.get("doc_type") == "conversation":
            conv_counts[key] += 1
        else:
            entry_counts[key] += 1

    keys = sorted(set(entry_counts) | set(conv_counts))
    return [
        TimelineBucket(
            date=key,
            entry_count=entry_counts[key],
            conversation_count=conv_counts[key],
        )
        for key in keys
    ]


@router.get("", response_model=TimelineResponse)
async def get_timeline(
    request: Request,
    auth: Annotated[tuple[UUID, frozenset[str]], Depends(require_scope("journal:read"))],
    date_from: Annotated[str, Query(description="Range start, YYYY-MM-DD (inclusive).")],
    date_to: Annotated[str, Query(description="Range end, YYYY-MM-DD (inclusive).")],
    bucket: Annotated[BucketUnit, Query(description="Grouping granularity.")] = "day",
    topic_prefix: Annotated[
        str | None,
        Query(description="Restrict to topics whose path starts with this prefix."),
    ] = None,
) -> Response:
    """GET /api/v1/timeline.

    Returns per-bucket counts of entries and conversations over the inclusive
    ``[date_from, date_to]`` range, grouped by ``day`` (default) or ``month``.

    422 when: a date is not ``YYYY-MM-DD``; the range is inverted
    (``date_from`` after ``date_to``); the span exceeds
    ``MAX_TIMELINE_SPAN_DAYS``; or ``topic_prefix`` is not a valid topic path.

    Cache-Control is ``private, no-store``: per-day counts are per-user content.
    """
    user_id, _scopes = auth
    app_ctx = require_app_ctx(request)
    log = bound_logger(request)

    start, end = _parse_range(date_from, date_to)
    validated_prefix = _parse_prefix(topic_prefix)

    async with safe_user_scoped_connection(app_ctx.pool, user_id=user_id) as conn:
        rows = await entries_repo.get_by_date_range(
            conn,
            None,
            date_from,
            date_to,
            ascending=True,
            title_only=True,
            topic_prefix=validated_prefix,
        )

    buckets = aggregate_buckets(rows, bucket)

    await log.info(
        "web_timeline",
        bucket=bucket,
        span_days=(end - start).days,
        bucket_count=len(buckets),
    )
    body = TimelineResponse(
        buckets=buckets,
        date_from=date_from,
        date_to=date_to,
        bucket=bucket,
    )
    return private_no_store_response(body)


def _parse_range(date_from: str, date_to: str) -> tuple[date_cls, date_cls]:
    """Validate the date pair and enforce ordering + max span. 422 on failure."""
    try:
        validate_date(date_from)
        validate_date(date_to)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None

    start = date_cls.fromisoformat(date_from)
    end = date_cls.fromisoformat(date_to)
    if start > end:
        raise HTTPException(
            status_code=422,
            detail="date_from must be on or before date_to.",
        )
    # Inclusive span: a single day is span 0, so the cap is days + 1 calendar days.
    if (end - start).days > MAX_TIMELINE_SPAN_DAYS:
        raise HTTPException(
            status_code=422,
            detail=f"Date range exceeds the {MAX_TIMELINE_SPAN_DAYS}-day maximum.",
        )
    return start, end


def _parse_prefix(topic_prefix: str | None) -> str | None:
    """Validate an optional topic prefix. 422 on invalid syntax; None passes through."""
    if not topic_prefix:
        return None
    try:
        return validate_topic(topic_prefix)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
