"""Stats repository -- dashboard aggregation SQL for the web API.

Holds the single-round-trip dashboard query consumed by ``GET /api/v1/stats``.
A separate module (rather than extending ``entries.py`` / ``conversations.py``)
keeps the cross-table aggregation isolated and avoids touching the entry/
conversation read paths.

All queries run on an RLS-scoped connection, so user scoping is enforced by the
row-level-security policy -- no explicit ``user_id`` predicate appears in the
SQL (mirrors ``entries.get_stats``). The connection sees only the calling
user's rows, including their single ``users`` row, from which the week boundary
is computed in the user's own timezone.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date as date_cls
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from gubbi.tools.context import resolve_period
from gubbi.validation import local_today

if TYPE_CHECKING:
    from datetime import datetime

    import asyncpg

__all__: list[str] = [
    "DashboardStats",
    "MostActiveTopic",
    "get_dashboard_stats",
    "user_week_bounds",
]

# Top-N cap for the most-active-topics strip and the activity window.
MOST_ACTIVE_TOPICS_LIMIT: int = 5
ACTIVE_WINDOW_DAYS: int = 30


@dataclass(frozen=True)
class MostActiveTopic:
    """A topic in the most-active list with its recent-window entry count."""

    path: str
    title: str
    entries_last_30d: int


@dataclass(frozen=True)
class DashboardStats:
    """Aggregated dashboard counts for one user."""

    topics_total: int
    entries_total: int
    entries_this_week: int
    conversations_total: int
    last_entry_at: datetime | None
    most_active_topics: tuple[MostActiveTopic, ...]


def user_week_bounds(timezone: str) -> tuple[date_cls, date_cls]:
    """Return the (Monday, Sunday) date bounds of the current week in ``timezone``.

    Pure function: derives "today" in the user's IANA timezone and resolves the
    Monday-start week window via the shared period resolver, so the boundary
    matches what briefing/timeline compute -- just keyed off the user's tz
    rather than the server setting. Falls back to UTC for an unknown tz (the
    same fallback ``local_today`` applies).
    """
    today = date_cls.fromisoformat(local_today(timezone))
    date_from, date_to, _label = resolve_period("this-week", today=today)
    return date_cls.fromisoformat(date_from), date_cls.fromisoformat(date_to)


async def get_dashboard_stats(conn: asyncpg.Connection) -> DashboardStats:
    """Return dashboard stats for the RLS-scoped user in a single round-trip.

    Reads the user's timezone from their ``users`` row to bound "this week" in
    local time, then issues one query carrying the scalar totals plus the
    top-N most-active topics (last 30 days) as a JSON aggregate so everything
    arrives together.
    """
    timezone = await _read_user_timezone(conn)
    week_from, week_to = user_week_bounds(timezone)

    row = await conn.fetchrow(
        """
        SELECT
            (SELECT COUNT(*) FROM topics)                          AS topics_total,
            (SELECT COUNT(*) FROM entries
                WHERE deleted_at IS NULL)                          AS entries_total,
            (SELECT COUNT(*) FROM entries
                WHERE deleted_at IS NULL
                  AND date >= $1 AND date <= $2)                   AS entries_this_week,
            (SELECT COUNT(*) FROM conversations)                   AS conversations_total,
            (SELECT MAX(updated_at) FROM entries
                WHERE deleted_at IS NULL)                          AS last_entry_at,
            (
                SELECT COALESCE(
                    jsonb_agg(
                        jsonb_build_object(
                            'path', ranked.path,
                            'title', ranked.title,
                            'entries_last_30d', ranked.cnt
                        )
                        ORDER BY ranked.cnt DESC, ranked.path ASC
                    ),
                    '[]'::jsonb
                )
                FROM (
                    SELECT t.path AS path, t.title AS title, COUNT(e.id) AS cnt
                    FROM topics t
                    JOIN entries e ON e.topic_id = t.id
                        AND e.deleted_at IS NULL
                        AND e.date >= $3
                    GROUP BY t.id, t.path, t.title
                    ORDER BY cnt DESC, t.path ASC
                    LIMIT $4
                ) AS ranked
            )                                                      AS most_active_topics
        """,
        week_from,
        week_to,
        _active_window_start(week_to),
        MOST_ACTIVE_TOPICS_LIMIT,
    )

    if row is None:
        return DashboardStats(
            topics_total=0,
            entries_total=0,
            entries_this_week=0,
            conversations_total=0,
            last_entry_at=None,
            most_active_topics=(),
        )

    return DashboardStats(
        topics_total=int(row["topics_total"] or 0),
        entries_total=int(row["entries_total"] or 0),
        entries_this_week=int(row["entries_this_week"] or 0),
        conversations_total=int(row["conversations_total"] or 0),
        last_entry_at=row["last_entry_at"],
        most_active_topics=_parse_active_topics(row["most_active_topics"]),
    )


def _active_window_start(reference: date_cls) -> date_cls:
    """Return the inclusive start date of the most-active window.

    Anchored on the current local week end so the window tracks the user's
    local calendar rather than server UTC.
    """
    return reference - timedelta(days=ACTIVE_WINDOW_DAYS - 1)


def _parse_active_topics(value: Any) -> tuple[MostActiveTopic, ...]:
    """Map the JSON aggregate rows to ``MostActiveTopic`` instances.

    asyncpg returns ``jsonb`` as a Python ``str`` unless a codec is registered;
    handle both the decoded-list and raw-string cases so the repo does not
    depend on connection-level codec setup.
    """
    if value is None:
        return ()
    rows = json.loads(value) if isinstance(value, str) else value
    return tuple(
        MostActiveTopic(
            path=str(r["path"]),
            title=str(r["title"]),
            entries_last_30d=int(r["entries_last_30d"]),
        )
        for r in rows
    )


async def _read_user_timezone(conn: asyncpg.Connection) -> str:
    """Return the RLS-scoped user's IANA timezone, defaulting to UTC.

    The connection sees only the calling user's ``users`` row under RLS, so an
    unfiltered SELECT returns that single timezone. A missing row (user not yet
    persisted) falls back to UTC.
    """
    tz = await conn.fetchval("SELECT timezone FROM users LIMIT 1")
    return str(tz) if tz else "UTC"
