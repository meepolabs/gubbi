"""Repository module for the extraction_jobs table.

All functions take an asyncpg.Connection as their first argument and
execute inline SQL. RLS is enforced at the database layer via the
extraction_jobs_user_isolation policy (migration 0023); no explicit
user_id filter is needed in SELECT queries -- the session variable
app.current_user_id set by user_scoped_connection provides the scope.

INSERT and UPDATE operations do supply user_id explicitly because the
WITH CHECK clause requires it to match the session variable.

Public surface
--------------
create_pending      -- INSERT a new pending job; raises ExtractionJobAlreadyInFlight on conflict
mark_running        -- UPDATE status to 'running' (worker stub)
update_progress     -- monotonic UPDATE of topics/entries counters (worker stub)
mark_completed      -- terminal UPDATE to 'completed'
mark_failed         -- terminal UPDATE to 'failed'
get_status_counts   -- aggregate SELECT returning StatusCounts (for /me endpoint)
ExtractionJobAlreadyInFlight -- raised by create_pending on partial-unique conflict
StatusCounts        -- dataclass returned by get_status_counts
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast
from uuid import UUID

import asyncpg

if TYPE_CHECKING:
    from datetime import date, datetime

__all__: list[str] = [
    "ExtractionJobAlreadyInFlight",
    "StatusCounts",
    "create_pending",
    "get_period_start",
    "get_status_counts",
    "mark_completed",
    "mark_failed",
    "mark_running",
    "update_progress",
]


class ExtractionJobAlreadyInFlight(Exception):
    """Raised by create_pending when an active job already exists.

    Raised when an active job (status NOT IN 'completed', 'failed') already
    exists for the same (user_id, conversation_id, source) combination.

    The existing job_id is provided so the caller can reuse it for idempotent
    enqueue semantics -- a client retry that lands while a previous job is
    still pending/running receives the SAME job_id back.
    """

    def __init__(self, existing_job_id: UUID) -> None:
        self.existing_job_id: UUID = existing_job_id
        super().__init__(
            f"An extraction job is already in-flight for this conversation: {existing_job_id}"
        )


@dataclass
class StatusCounts:
    """Aggregated extraction job status counts for the current user (RLS-scoped).

    in_flight_count : number of jobs in status 'pending' or 'running'
    synced_count    : number of jobs that reached 'completed'
    last_sync_at    : completed_at of the most recently completed job, or None
    """

    in_flight_count: int
    synced_count: int
    last_sync_at: datetime | None


# ---------------------------------------------------------------------------
# create_pending
# ---------------------------------------------------------------------------


async def create_pending(
    conn: asyncpg.Connection,
    user_id: UUID,
    conversation_id: int,
    source: str,
    *,
    period_start: date,
) -> UUID:
    """INSERT a new extraction job row with status='pending'.

    Returns the generated UUID job id.

    If an active job (status NOT IN ('completed', 'failed')) already exists
    for the same (user_id, conversation_id, source), raises
    ExtractionJobAlreadyInFlight with the existing job's id so the caller
    can reuse it for idempotent enqueue semantics.

    The partial unique index idx_extraction_jobs_active_per_conversation
    (migration 0023) enforces this at the database level; we catch the
    UniqueViolationError and surface it as the typed exception.
    """
    try:
        row = await conn.fetchrow(
            """
            INSERT INTO extraction_jobs
                (user_id, conversation_id, source, status, period_start)
            VALUES ($1, $2, $3, 'pending', $4)
            RETURNING id
            """,
            user_id,
            conversation_id,
            source,
            period_start,
        )
    except asyncpg.UniqueViolationError:
        # The partial unique index fired -- an in-flight job exists.
        # Look up the existing job_id to return via the exception.
        existing_row = await conn.fetchrow(
            """
            SELECT id
            FROM extraction_jobs
            WHERE user_id = $1
              AND conversation_id = $2
              AND source = $3
              AND status NOT IN ('completed', 'failed')
            LIMIT 1
            """,
            user_id,
            conversation_id,
            source,
        )
        if existing_row is None:
            raise RuntimeError(
                "Conflict on "
                f"(user_id={user_id}, conversation_id={conversation_id}, source={source}) "
                "but no existing row found"
            ) from None
        existing_id = UUID(str(existing_row["id"]))
        raise ExtractionJobAlreadyInFlight(existing_id) from None

    if row is None:
        raise RuntimeError(
            f"INSERT INTO extraction_jobs returned no row for "
            f"user_id={user_id} conversation_id={conversation_id} source={source!r}"
        )
    return UUID(str(row["id"]))


# ---------------------------------------------------------------------------
# get_period_start
# ---------------------------------------------------------------------------


async def get_period_start(
    conn: asyncpg.Connection,
    job_id: UUID,
) -> date | None:
    """Fetch the period_start date stored on an extraction_jobs row.

    Returns None if no row matches (e.g. job not yet visible under RLS,
    or job_id is invalid). The caller should fall back to
    current_period_start() when None is returned.
    """
    row = await conn.fetchrow(
        "SELECT period_start FROM extraction_jobs WHERE id = $1",
        job_id,
    )
    if row is None:
        return None
    return cast("date", row["period_start"])


# ---------------------------------------------------------------------------
# mark_running
# ---------------------------------------------------------------------------


async def mark_running(conn: asyncpg.Connection, job_id: UUID) -> None:
    """Transition job from 'pending' or 'failed' to 'running'.

    Retry attempts are allowed to revive a previously failed row, so the
    transition resets started_at and clears any stale error_code. No-op if the
    row is already terminal-completed or currently running.

    Worker stub -- reserved for worker use; not called by ingest.
    """
    await conn.execute(
        """
        UPDATE extraction_jobs
        SET status = 'running',
            started_at = now(),
            error_code = NULL
        WHERE id = $1
          AND status IN ('pending', 'failed')
        """,
        job_id,
    )


# ---------------------------------------------------------------------------
# update_progress
# ---------------------------------------------------------------------------


async def update_progress(
    conn: asyncpg.Connection,
    job_id: UUID,
    *,
    topics_created: int | None = None,
    entries_created: int | None = None,
) -> None:
    """Monotonically update progress counters using GREATEST to avoid regressions.

    Accepts keyword-only arguments so callers are explicit about which counters
    are being updated. Passing None for a counter leaves it unchanged.

    Worker stub -- reserved for worker use; not called by ingest.
    """
    if topics_created is None and entries_created is None:
        return

    await conn.execute(
        """
        UPDATE extraction_jobs
        SET topics_created = CASE
                WHEN $2::integer IS NULL THEN topics_created
                ELSE GREATEST(topics_created, $2)
            END,
            entries_created = CASE
                WHEN $3::integer IS NULL THEN entries_created
                ELSE GREATEST(entries_created, $3)
            END
        WHERE id = $1
          AND status NOT IN ('completed', 'failed')
        """,
        job_id,
        topics_created,
        entries_created,
    )


# ---------------------------------------------------------------------------
# mark_completed
# ---------------------------------------------------------------------------


async def mark_completed(
    conn: asyncpg.Connection,
    job_id: UUID,
    *,
    topics_created: int,
    entries_created: int,
    cents_spent: int,
) -> bool:
    """Transition job to 'completed' and record final counters + timestamp.

    WHERE clause guards against overwriting a terminal state -- no-op if
    status is already 'completed' or 'failed'. This makes the function safe
    to call from idempotent worker retry paths.
    """
    result = await conn.execute(
        """
        UPDATE extraction_jobs
        SET status = 'completed',
            topics_created = $2,
            entries_created = $3,
            cents_spent = $4,
            completed_at = now()
        WHERE id = $1
          AND status NOT IN ('completed', 'failed')
        """,
        job_id,
        topics_created,
        entries_created,
        cents_spent,
    )
    return str(result) == "UPDATE 1"


# ---------------------------------------------------------------------------
# mark_failed
# ---------------------------------------------------------------------------


async def mark_failed(
    conn: asyncpg.Connection,
    job_id: UUID,
    *,
    error_code: str,
) -> bool:
    """Transition job to 'failed' and record the error_code.

    WHERE clause guards against overwriting a terminal state -- no-op if
    status is already 'completed' or 'failed'.
    """
    result = await conn.execute(
        """
        UPDATE extraction_jobs
        SET status = 'failed',
            error_code = $2,
            completed_at = now()
        WHERE id = $1
          AND status NOT IN ('completed', 'failed')
        """,
        job_id,
        error_code,
    )
    return str(result) == "UPDATE 1"


# ---------------------------------------------------------------------------
# get_status_counts
# ---------------------------------------------------------------------------


async def get_status_counts(conn: asyncpg.Connection) -> StatusCounts:
    """Return aggregate status counts for the current RLS-scoped user.

    RLS (extraction_jobs_user_isolation) automatically filters rows to the
    current user -- no explicit user_id predicate is needed.

    Uses FILTER aggregates to bucket counts in a single table scan:
      in_flight_count : pending + running
      synced_count    : completed
      last_sync_at    : max(completed_at) among completed rows
    """
    row = await conn.fetchrow(
        """
        SELECT
            COUNT(*) FILTER (WHERE status IN ('pending', 'running'))  AS in_flight_count,
            COUNT(*) FILTER (WHERE status = 'completed')              AS synced_count,
            MAX(completed_at) FILTER (WHERE status = 'completed')     AS last_sync_at
        FROM extraction_jobs
        """
    )
    if row is None:
        return StatusCounts(in_flight_count=0, synced_count=0, last_sync_at=None)
    return StatusCounts(
        in_flight_count=int(row["in_flight_count"] or 0),
        synced_count=int(row["synced_count"] or 0),
        last_sync_at=row["last_sync_at"],
    )
