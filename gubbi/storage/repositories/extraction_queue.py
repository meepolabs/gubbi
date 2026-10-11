"""Repository module for scheduling and admission over extraction_jobs.

Every function runs on a connection from the admin pool (BYPASSRLS): the
scheduler works across tenants, so queries filter on user_id explicitly where
a result is per user.

A job is queued while it is ``pending`` with ``admitted_at IS NULL``. Admission
writes ``admitted_at`` and ``budget_period_start`` exactly once; usage figures
count admitted rows only. A row whose ``run_after`` is ``'infinity'`` is held
and is never runnable.

Public surface
--------------
user_period_usage       -- per-user period, period-backfill and lifetime-backfill usage
global_period_tokens    -- filtered tokens admitted in a period, all users
first_upload_at         -- earliest live/backfill job of a user, optionally per platform
select_runnable         -- queued or retryable-failed rows whose run_after has passed
terminal_failure_sql    -- SQL predicate: a failed row the retry policy never runs again
terminal_failure_args   -- the parameters terminal_failure_sql binds
admit                   -- charge rows to a period, once per row
defer                   -- record why a queued row was held back
stale_admitted_pending  -- admitted rows still pending after a given age
inflight_admitted_count -- admitted rows that are pending or running
UserPeriodUsage         -- dataclass returned by user_period_usage
RunnableJob             -- dataclass returned by select_runnable / stale_admitted_pending
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast
from uuid import UUID

from gubbi.extraction.intake import Lane
from gubbi.extraction.retry_policy import (
    DETERMINISTIC_CODES,
    DETERMINISTIC_MAX_ATTEMPTS,
    TERMINAL_CODES,
    TRANSIENT_CODES,
)

if TYPE_CHECKING:
    from collections.abc import Collection
    from datetime import date, datetime, timedelta

    import asyncpg

__all__: list[str] = [
    "RunnableJob",
    "UserPeriodUsage",
    "admit",
    "defer",
    "first_upload_at",
    "global_period_tokens",
    "inflight_admitted_count",
    "select_runnable",
    "stale_admitted_pending",
    "terminal_failure_args",
    "terminal_failure_sql",
    "user_period_usage",
]


@dataclass(frozen=True)
class UserPeriodUsage:
    """Admitted usage of one user.

    period_conversations / period_tokens : admitted in the requested budget period
    period_backfill_conversations / period_backfill_tokens : the backfill-lane
        subset of the period figures
    lifetime_backfill_conversations / lifetime_backfill_tokens : admitted in the
        backfill lane, any period

    Field names match the keywords of
    :func:`gubbi.extraction.admission.user_usage_from_totals` one-to-one.
    """

    period_conversations: int
    period_tokens: int
    period_backfill_conversations: int
    period_backfill_tokens: int
    lifetime_backfill_conversations: int
    lifetime_backfill_tokens: int


@dataclass(frozen=True)
class RunnableJob:
    """One extraction job as the scheduler sees it."""

    job_id: UUID
    user_id: UUID
    conversation_id: int
    source: str
    lane: Lane
    activity_at: datetime | None
    filtered_tokens: int
    attempts: int
    admitted_at: datetime | None


def _to_runnable_job(row: asyncpg.Record) -> RunnableJob:
    return RunnableJob(
        job_id=UUID(str(row["id"])),
        user_id=UUID(str(row["user_id"])),
        conversation_id=int(row["conversation_id"]),
        source=str(row["source"]),
        lane=Lane(row["lane"]),
        activity_at=row["activity_at"],
        filtered_tokens=int(row["filtered_tokens"]),
        attempts=int(row["attempts"]),
        admitted_at=row["admitted_at"],
    )


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------


async def user_period_usage(
    conn: asyncpg.Connection,
    user_ids: Collection[UUID],
    period: date,
) -> dict[UUID, UserPeriodUsage]:
    """Return the admitted usage of every requested user; a user with none maps to zeros."""
    rows = await conn.fetch(
        """
        SELECT
            u.user_id,
            count(j.id) FILTER (WHERE j.budget_period_start = $2) AS period_conversations,
            coalesce(sum(j.filtered_tokens) FILTER (WHERE j.budget_period_start = $2), 0)
                AS period_tokens,
            count(j.id) FILTER (WHERE j.budget_period_start = $2 AND j.lane = 'backfill')
                AS period_backfill_conversations,
            coalesce(
                sum(j.filtered_tokens)
                    FILTER (WHERE j.budget_period_start = $2 AND j.lane = 'backfill'),
                0
            ) AS period_backfill_tokens,
            count(j.id) FILTER (WHERE j.lane = 'backfill') AS lifetime_backfill_conversations,
            coalesce(sum(j.filtered_tokens) FILTER (WHERE j.lane = 'backfill'), 0)
                AS lifetime_backfill_tokens
        FROM (SELECT DISTINCT unnest($1::uuid[]) AS user_id) AS u
        LEFT JOIN extraction_jobs AS j
               ON j.user_id = u.user_id
              AND j.admitted_at IS NOT NULL
        GROUP BY u.user_id
        """,
        list(user_ids),
        period,
    )
    return {
        UUID(str(row["user_id"])): UserPeriodUsage(
            period_conversations=int(row["period_conversations"]),
            period_tokens=int(row["period_tokens"]),
            period_backfill_conversations=int(row["period_backfill_conversations"]),
            period_backfill_tokens=int(row["period_backfill_tokens"]),
            lifetime_backfill_conversations=int(row["lifetime_backfill_conversations"]),
            lifetime_backfill_tokens=int(row["lifetime_backfill_tokens"]),
        )
        for row in rows
    }


async def global_period_tokens(conn: asyncpg.Connection, period: date) -> int:
    """Return the filtered tokens admitted in ``period`` across every user."""
    total = await conn.fetchval(
        """
        SELECT coalesce(sum(filtered_tokens), 0)
        FROM extraction_jobs
        WHERE admitted_at IS NOT NULL
          AND budget_period_start = $1
        """,
        period,
    )
    return int(total)


async def first_upload_at(
    conn: asyncpg.Connection,
    user_id: UUID,
    platform: str | None,
) -> datetime | None:
    """Return the earliest created_at of the user's live or backfill jobs.

    ``platform`` matches ``extraction_jobs.source``; None spans every platform.
    Returns None when the user has no such job.
    """
    earliest = await conn.fetchval(
        """
        SELECT min(created_at)
        FROM extraction_jobs
        WHERE user_id = $1
          AND lane IN ('live', 'backfill')
          AND ($2::text IS NULL OR source = $2::text)
        """,
        user_id,
        platform,
    )
    return cast("datetime | None", earliest)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def terminal_failure_sql(first_param: int) -> str:
    """Return a two-valued SQL predicate over ``error_code`` and ``attempts``.

    True exactly when :func:`gubbi.extraction.retry_policy.is_terminal` is: a
    terminal code, or a deterministic code at or past its attempt limit. It
    binds :func:`terminal_failure_args` at ``$first_param`` onward.
    """
    terminal, deterministic, max_attempts = first_param, first_param + 1, first_param + 2
    return (
        f"coalesce(error_code = ANY(${terminal}::text[])"
        f" OR (error_code = ANY(${deterministic}::text[]) AND attempts >= ${max_attempts}),"
        " false)"
    )


def terminal_failure_args() -> tuple[list[str], list[str], int]:
    """Return the parameters :func:`terminal_failure_sql` binds, in order."""
    return sorted(TERMINAL_CODES), sorted(DETERMINISTIC_CODES), DETERMINISTIC_MAX_ATTEMPTS


async def select_runnable(conn: asyncpg.Connection, limit: int) -> list[RunnableJob]:
    """Return up to ``limit`` rows the scheduler may admit or dispatch now.

    Runnable: ``pending`` and not yet admitted, or ``failed`` with a transient
    ``error_code`` or a deterministic one below its attempt limit; in both
    cases ``run_after <= now()``. A failed row the retry policy never runs
    again is excluded whatever its ``run_after``. An admitted ``pending`` row
    is already dispatched and is excluded; see :func:`stale_admitted_pending`
    for lost dispatches.

    Order: lane ``paste``, ``live``, ``backfill``; newest ``activity_at`` first
    within a lane (unknown last); then ``created_at`` and ``id`` so equal rows
    come back in a stable order and ``limit`` never starves a higher lane.
    """
    rows = await conn.fetch(
        f"""
        SELECT id, user_id, conversation_id, source, lane, activity_at,
               filtered_tokens, attempts, admitted_at
        FROM extraction_jobs
        WHERE run_after <= now()
          AND (
                (status = 'pending' AND admitted_at IS NULL)
             OR (status = 'failed'
                 AND error_code = ANY($2::text[])
                 AND NOT {terminal_failure_sql(3)})
          )
        ORDER BY CASE lane WHEN 'paste' THEN 0 WHEN 'live' THEN 1 ELSE 2 END,
                 activity_at DESC NULLS LAST,
                 created_at,
                 id
        LIMIT $1
        """,  # noqa: S608 - the interpolated predicate is fixed SQL; values are bound
        limit,
        sorted(TRANSIENT_CODES | DETERMINISTIC_CODES),
        *terminal_failure_args(),
    )
    return [_to_runnable_job(row) for row in rows]


async def stale_admitted_pending(
    conn: asyncpg.Connection,
    age: timedelta,
) -> list[RunnableJob]:
    """Return admitted rows still ``pending`` after ``age`` since admission, oldest first."""
    rows = await conn.fetch(
        """
        SELECT id, user_id, conversation_id, source, lane, activity_at,
               filtered_tokens, attempts, admitted_at
        FROM extraction_jobs
        WHERE status = 'pending'
          AND admitted_at IS NOT NULL
          AND admitted_at < now() - $1::interval
        ORDER BY admitted_at, id
        """,
        age,
    )
    return [_to_runnable_job(row) for row in rows]


async def inflight_admitted_count(conn: asyncpg.Connection) -> int:
    """Return the number of admitted rows that are ``pending`` or ``running``."""
    count = await conn.fetchval(
        """
        SELECT count(*)
        FROM extraction_jobs
        WHERE admitted_at IS NOT NULL
          AND status IN ('pending', 'running')
        """
    )
    return int(count)


# ---------------------------------------------------------------------------
# Admission
# ---------------------------------------------------------------------------


async def admit(
    conn: asyncpg.Connection,
    job_ids: Collection[UUID],
    period: date,
) -> int:
    """Charge each row to ``period`` and clear its defer reason; return rows charged.

    A row already admitted is left untouched, so a retried or re-dispatched
    row is never charged twice and never moves to a later period.
    """
    result = await conn.execute(
        """
        UPDATE extraction_jobs
        SET admitted_at = now(),
            budget_period_start = $2,
            defer_reason = NULL
        WHERE id = ANY($1::uuid[])
          AND admitted_at IS NULL
        """,
        list(job_ids),
        period,
    )
    return int(str(result).rsplit(" ", 1)[-1])


async def defer(conn: asyncpg.Connection, job_id: UUID, reason: str) -> bool:
    """Set ``defer_reason`` on a ``pending`` or ``failed`` row; return whether a row changed.

    ``reason`` must be one the ``extraction_jobs_defer_reason_check`` constraint
    accepts; any other value raises ``asyncpg.CheckViolationError``.
    """
    result = await conn.execute(
        """
        UPDATE extraction_jobs
        SET defer_reason = $2
        WHERE id = $1
          AND status IN ('pending', 'failed')
        """,
        job_id,
        reason,
    )
    return str(result) == "UPDATE 1"
