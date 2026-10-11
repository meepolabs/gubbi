"""Repository module for scheduling and dispatch over extraction_jobs.

The scheduler calls these on a connection from the admin pool (BYPASSRLS) to
work across tenants, so queries filter on user_id explicitly where a result is
per user. ``first_upload_at`` is also correct on a user-scoped RLS connection
for that user's own id; other users then read as having no jobs.

A job is queued while it is ``pending`` with ``admitted_at IS NULL``.
Dispatch writes ``admitted_at``; a dispatched job that is pending or running
is in flight. A row whose ``run_after`` is ``'infinity'`` is held and is never
runnable. Budgets are not read here: balances live in the wallet tables.

Public surface
--------------
first_upload_at         -- earliest live/backfill job of a user, optionally per platform
supersede_failed        -- end failed rows that a newer job of their conversation replaces
select_runnable         -- lock queued or revivable failed rows whose run_after has passed,
                           at most a given number per user
terminal_failure_sql    -- SQL predicate: a failed row the retry policy never runs again
terminal_failure_args   -- the parameters terminal_failure_sql binds
admit                   -- mark pending rows dispatched now
revive_failed           -- move failed rows back to pending for another run
fail_enqueue_lost       -- fail a pending row whose enqueue raised, with growing backoff;
                           returns whether the row runs again
defer_lane              -- hold back a user's runnable rows of one lane with a reason
stale_admitted_pending  -- dispatched rows still pending after a given age
inflight_admitted_count -- dispatched rows that are pending or running
inflight_by_user        -- the same, per requested user
queue_snapshot          -- runnable depth and age and deferrals, for metrics
RunnableJob             -- dataclass returned by select_runnable / stale_admitted_pending
QueueSnapshot           -- dataclass returned by queue_snapshot
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast
from uuid import UUID

import asyncpg

from gubbi.extraction.intake import Lane
from gubbi.extraction.retry_policy import (
    DETERMINISTIC_CODES,
    DETERMINISTIC_MAX_ATTEMPTS,
    TERMINAL_CODES,
    ErrorCode,
    FailureOutcome,
    failure_outcome,
    next_run_after,
)

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping
    from datetime import datetime, timedelta

__all__: list[str] = [
    "QueueSnapshot",
    "RunnableJob",
    "admit",
    "defer_lane",
    "fail_enqueue_lost",
    "first_upload_at",
    "inflight_admitted_count",
    "inflight_by_user",
    "queue_snapshot",
    "revive_failed",
    "select_runnable",
    "stale_admitted_pending",
    "supersede_failed",
    "terminal_failure_args",
    "terminal_failure_sql",
]


@dataclass(frozen=True)
class RunnableJob:
    """One extraction job as the scheduler sees it."""

    job_id: UUID
    user_id: UUID
    conversation_id: int
    source: str
    lane: Lane
    activity_at: datetime | None
    attempts: int
    admitted_at: datetime | None
    status: str


def _to_runnable_job(row: asyncpg.Record) -> RunnableJob:
    return RunnableJob(
        job_id=UUID(str(row["id"])),
        user_id=UUID(str(row["user_id"])),
        conversation_id=int(row["conversation_id"]),
        source=str(row["source"]),
        lane=Lane(row["lane"]),
        activity_at=row["activity_at"],
        attempts=int(row["attempts"]),
        admitted_at=row["admitted_at"],
        status=str(row["status"]),
    )


# ---------------------------------------------------------------------------
# Upload history
# ---------------------------------------------------------------------------


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
    terminal code, or a deterministic code at or past its attempt limit; a NULL
    or unrecognised code is never terminal. Every query that decides whether a
    failed row may run again uses this one predicate. It
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


_JOB_COLUMNS = (
    "j.id, j.user_id, j.conversation_id, j.source, j.lane, j.activity_at,"
    " j.attempts, j.admitted_at, j.status"
)

_QUEUE_ORDER = (
    "CASE j.lane WHEN 'paste' THEN 0 WHEN 'live' THEN 1 ELSE 2 END,"
    " j.activity_at DESC NULLS LAST, j.created_at, j.id"
)

_NEWER_JOB_EXISTS = """
EXISTS (
    SELECT 1
    FROM extraction_jobs AS newer
    WHERE newer.user_id = j.user_id
      AND newer.conversation_id = j.conversation_id
      AND (newer.created_at, newer.id) > (j.created_at, j.id)
)
"""


def _runnable_sql(first_param: int) -> str:
    """Return the predicate over ``extraction_jobs AS j`` for a row runnable now.

    Binds :func:`terminal_failure_args` at ``$first_param`` onward. See
    :func:`select_runnable` for what makes a row runnable.
    """
    return f"""
        j.run_after <= now()
        AND (
              (j.status = 'pending' AND j.admitted_at IS NULL)
           OR (j.status = 'failed'
               AND NOT {terminal_failure_sql(first_param)}
               AND NOT EXISTS (
                      SELECT 1
                      FROM extraction_jobs AS active
                      WHERE active.user_id = j.user_id
                        AND active.conversation_id = j.conversation_id
                        AND active.status NOT IN ('completed', 'failed')
               )
               AND NOT {_NEWER_JOB_EXISTS})
        )
    """  # noqa: S608 - the interpolated predicates are fixed SQL; values are bound


async def supersede_failed(conn: asyncpg.Connection) -> int:
    """End every non-terminal ``failed`` row that has a newer job; return rows changed.

    A newer job of the same conversation reads its current content, so it
    covers whatever the older row would extract. The row stays ``failed``
    with the terminal code ``superseded`` and ``run_after = 'infinity'``,
    whatever the newer job's status. A held row is left until it is
    released, so the scan stays on the runnable index range; a row another
    transaction holds is skipped and ended on a later call.
    """
    result = await conn.execute(
        f"""
        UPDATE extraction_jobs
        SET error_code = $4,
            run_after = 'infinity'
        WHERE id IN (
            SELECT j.id
            FROM extraction_jobs AS j
            WHERE j.status = 'failed'
              AND j.run_after < 'infinity'
              AND NOT {terminal_failure_sql(1)}
              AND {_NEWER_JOB_EXISTS}
            FOR UPDATE OF j SKIP LOCKED
        )
        """,  # noqa: S608 - the interpolated predicates are fixed SQL; values are bound
        *terminal_failure_args(),
        ErrorCode.SUPERSEDED.value,
    )
    return int(str(result).rsplit(" ", 1)[-1])


async def select_runnable(conn: asyncpg.Connection, limit: int, per_user: int) -> list[RunnableJob]:
    """Lock and return up to ``limit`` rows the scheduler may dispatch now.

    Rows are locked ``FOR UPDATE SKIP LOCKED`` until the caller's transaction
    ends, so ingest cannot rewrite a row's queue inputs between this read and
    its dispatch; a row another transaction holds is skipped this time.

    Runnable: ``pending`` and not yet dispatched, or ``failed`` and not
    terminal per :func:`terminal_failure_sql` (a missing or unrecognised
    ``error_code`` retries as transient); in both cases ``run_after <=
    now()``. A failed row the retry policy never runs again is excluded
    whatever its ``run_after``. A dispatched ``pending`` row is excluded; see
    :func:`stale_admitted_pending` for lost dispatches.

    Only one job per conversation may be active (pending or running), so a
    failed row whose conversation has an active job is excluded, and so is a
    failed row with any newer job of its conversation (normally already ended
    by :func:`supersede_failed`). Both exclusions apply before ``limit``, so
    blocked failed rows never fill the batch. A job made active after this
    read is caught by :func:`revive_failed`.

    At most ``per_user`` rows of any one user are returned, taken in the
    order below and counted before ``limit``, so one user's backlog cannot
    fill the batch.

    Order: lane ``paste``, ``live``, ``backfill``; newest ``activity_at`` first
    within a lane (unknown last); then ``created_at`` and ``id`` so equal rows
    come back in a stable order and ``limit`` never starves a higher lane.
    """
    rows = await conn.fetch(
        f"""
        WITH ranked AS (
            SELECT j.id,
                   row_number() OVER (PARTITION BY j.user_id ORDER BY {_QUEUE_ORDER}) AS rank
            FROM extraction_jobs AS j
            WHERE {_runnable_sql(3)}
        )
        SELECT {_JOB_COLUMNS}
        FROM extraction_jobs AS j
        JOIN ranked AS r ON r.id = j.id
        WHERE r.rank <= $2
        ORDER BY {_QUEUE_ORDER}
        LIMIT $1
        FOR UPDATE OF j SKIP LOCKED
        """,  # noqa: S608 - the interpolated fragments are fixed SQL; values are bound
        limit,
        per_user,
        *terminal_failure_args(),
    )
    return [_to_runnable_job(row) for row in rows]


async def stale_admitted_pending(
    conn: asyncpg.Connection,
    age: timedelta,
) -> list[RunnableJob]:
    """Return dispatched rows still ``pending`` after ``age`` since dispatch, oldest first."""
    rows = await conn.fetch(
        f"""
        SELECT {_JOB_COLUMNS}
        FROM extraction_jobs AS j
        WHERE j.status = 'pending'
          AND j.admitted_at IS NOT NULL
          AND j.admitted_at < now() - $1::interval
        ORDER BY j.admitted_at, j.id
        """,  # noqa: S608 - the interpolated column list is fixed SQL; values are bound
        age,
    )
    return [_to_runnable_job(row) for row in rows]


async def inflight_admitted_count(conn: asyncpg.Connection) -> int:
    """Return the number of dispatched rows that are ``pending`` or ``running``."""
    count = await conn.fetchval(
        """
        SELECT count(*)
        FROM extraction_jobs
        WHERE admitted_at IS NOT NULL
          AND status IN ('pending', 'running')
        """
    )
    return int(count)


async def inflight_by_user(conn: asyncpg.Connection, user_ids: Collection[UUID]) -> dict[UUID, int]:
    """Return each requested user's dispatched rows that are ``pending`` or ``running``.

    A dispatched row not yet started counts. A user with none maps to 0.
    """
    rows = await conn.fetch(
        """
        SELECT user_id, count(*) AS inflight
        FROM extraction_jobs
        WHERE admitted_at IS NOT NULL
          AND status IN ('pending', 'running')
          AND user_id = ANY($1::uuid[])
        GROUP BY user_id
        """,
        list(user_ids),
    )
    counts = {UUID(str(row["user_id"])): int(row["inflight"]) for row in rows}
    return {user_id: counts.get(user_id, 0) for user_id in user_ids}


# ---------------------------------------------------------------------------
# Dispatch and deferral
# ---------------------------------------------------------------------------


async def admit(conn: asyncpg.Connection, job_ids: Collection[UUID]) -> int:
    """Mark each ``pending`` row dispatched now and clear its defer reason; return rows changed."""
    result = await conn.execute(
        """
        UPDATE extraction_jobs
        SET admitted_at = now(),
            defer_reason = NULL
        WHERE id = ANY($1::uuid[])
          AND status = 'pending'
        """,
        list(job_ids),
    )
    return int(str(result).rsplit(" ", 1)[-1])


async def defer_lane(
    conn: asyncpg.Connection,
    lane: str,
    reason: str,
    *,
    user_id: UUID | None,
    recheck: timedelta,
) -> int:
    """Hold back every runnable row of ``lane`` with ``reason``; return rows changed.

    ``user_id`` limits the deferral to one user's rows; None defers the lane
    for every user. Runnable means what :func:`select_runnable` selects. Each
    row's ``run_after`` becomes the earlier of the start of the next UTC month
    and ``now() + recheck``. A row another transaction holds is skipped.
    ``reason`` must be one the ``extraction_jobs_defer_reason_check``
    constraint accepts; any other value raises ``asyncpg.CheckViolationError``.
    """
    result = await conn.execute(
        f"""
        UPDATE extraction_jobs
        SET defer_reason = $2,
            run_after = least(
                (date_trunc('month', now() AT TIME ZONE 'UTC') + interval '1 month')
                    AT TIME ZONE 'UTC',
                now() + $4::interval
            )
        WHERE id IN (
            SELECT j.id
            FROM extraction_jobs AS j
            WHERE j.lane = $1
              AND ($3::uuid IS NULL OR j.user_id = $3::uuid)
              AND {_runnable_sql(5)}
            FOR UPDATE OF j SKIP LOCKED
        )
        """,  # noqa: S608 - the interpolated predicate is fixed SQL; values are bound
        lane,
        reason,
        user_id,
        recheck,
        *terminal_failure_args(),
    )
    return int(str(result).rsplit(" ", 1)[-1])


async def revive_failed(conn: asyncpg.Connection, job_ids: Collection[UUID]) -> set[UUID]:
    """Move ``failed`` rows back to ``pending``; return the ids moved.

    Each row moves in its own savepoint. A row whose conversation gained an
    active job meanwhile (an ingest racing this call) violates the one active
    job per conversation index and stays failed; the caller's transaction is
    unaffected.
    """
    revived: set[UUID] = set()
    for job_id in job_ids:
        try:
            async with conn.transaction():
                result = await conn.execute(
                    "UPDATE extraction_jobs SET status = 'pending'"
                    " WHERE id = $1 AND status = 'failed'",
                    job_id,
                )
        except asyncpg.UniqueViolationError:
            continue
        if str(result) == "UPDATE 1":
            revived.add(job_id)
    return revived


async def fail_enqueue_lost(conn: asyncpg.Connection, job_id: UUID) -> FailureOutcome | None:
    """Fail a ``pending`` row whose hand-off to the worker queue raised.

    Returns whether the row runs again, or None when it was not failed: the
    row is locked first, so a row a worker claimed meanwhile is left alone.
    ``dispatch_failures`` counts consecutive lost hand-offs and is the backoff
    exponent, so the delay grows through an outage; ``attempts`` counts worker
    runs only and is untouched, so a lost hand-off never uses up a
    deterministic retry.
    """
    async with conn.transaction():
        row = await conn.fetchrow(
            """
            SELECT dispatch_failures, clock_timestamp() AS db_now
            FROM extraction_jobs
            WHERE id = $1
              AND status = 'pending'
            FOR UPDATE
            """,
            job_id,
        )
        if row is None:
            return None
        failures = int(row["dispatch_failures"]) + 1
        retry_at = next_run_after(ErrorCode.ENQUEUE_LOST, failures, row["db_now"])
        await conn.execute(
            """
            UPDATE extraction_jobs
            SET status = 'failed',
                error_code = $2,
                run_after = $3::text::timestamptz,
                dispatch_failures = $4,
                completed_at = now()
            WHERE id = $1
            """,
            job_id,
            ErrorCode.ENQUEUE_LOST.value,
            "infinity" if retry_at is None else retry_at.isoformat(),
            failures,
        )
    return failure_outcome(ErrorCode.ENQUEUE_LOST, failures)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QueueSnapshot:
    """Queue figures read at one moment.

    oldest_runnable_age_seconds : ``now() - run_after`` of the runnable row
        whose ``run_after`` is earliest; 0 when nothing is runnable. A row
        counts from when it became runnable, so a deferral or retry backoff
        that has not yet passed never ages the queue.
    depth_by_lane : runnable rows per lane; a lane with none is absent
    deferred_by_reason : rows held back by admission and not yet due again
        (``defer_reason`` set, queued or failed, ``run_after`` in the future
        and not held at ``'infinity'``); a reason with none is absent.
        Disjoint from the runnable rows.
    """

    oldest_runnable_age_seconds: float
    depth_by_lane: Mapping[Lane, int]
    deferred_by_reason: Mapping[str, int]


async def queue_snapshot(conn: asyncpg.Connection) -> QueueSnapshot:
    """Read the queue figures the scheduler reports each tick.

    Takes no row locks. Runnable means exactly what :func:`select_runnable`
    selects, without its batch and per-user limits.
    """
    queue_rows = await conn.fetch(
        f"""
        SELECT 'lane' AS kind, j.lane AS key, count(*) AS rows,
               date_part('epoch', now() - min(j.run_after))::float8 AS oldest_age
        FROM extraction_jobs AS j
        WHERE {_runnable_sql(1)}
        GROUP BY j.lane
        UNION ALL
        SELECT 'deferred', defer_reason, count(*), NULL
        FROM extraction_jobs
        WHERE defer_reason IS NOT NULL
          AND ((status = 'pending' AND admitted_at IS NULL) OR status = 'failed')
          AND run_after > now()
          AND run_after < 'infinity'
        GROUP BY defer_reason
        """,  # noqa: S608 - the interpolated predicate is fixed SQL; values are bound
        *terminal_failure_args(),
    )
    lanes = [row for row in queue_rows if row["kind"] == "lane"]
    return QueueSnapshot(
        oldest_runnable_age_seconds=max((float(row["oldest_age"]) for row in lanes), default=0.0),
        depth_by_lane={Lane(row["key"]): int(row["rows"]) for row in lanes},
        deferred_by_reason={
            str(row["key"]): int(row["rows"]) for row in queue_rows if row["kind"] == "deferred"
        },
    )
