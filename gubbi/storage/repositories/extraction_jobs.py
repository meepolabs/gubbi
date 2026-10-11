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
refresh_queued      -- rewrite the queue inputs of a pending, not yet admitted job
mark_running        -- claim a job for a run; returns the status it was claimed from
update_progress     -- monotonic UPDATE of topics/entries counters (worker stub)
mark_completed      -- terminal UPDATE to 'completed'; adds this run's usage to the row
mark_failed         -- terminal UPDATE to 'failed'; adds this run's usage to the row
get_status_counts   -- aggregate SELECT returning StatusCounts (for /me endpoint)
ExtractionJobAlreadyInFlight -- raised by create_pending on partial-unique conflict
StatusCounts        -- dataclass returned by get_status_counts
ClaimedFrom         -- status mark_running claimed a job from
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, cast
from uuid import UUID

import asyncpg

if TYPE_CHECKING:
    from datetime import date, datetime

_CREATE_PENDING_ATTEMPTS: Final = 3

__all__: list[str] = [
    "ClaimedFrom",
    "ExtractionJobAlreadyInFlight",
    "StatusCounts",
    "create_pending",
    "get_period_start",
    "get_status_counts",
    "mark_completed",
    "mark_failed",
    "mark_running",
    "refresh_queued",
    "update_progress",
]


ClaimedFrom = Literal["pending", "failed"]


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
    period_start: date | None = None,
    lane: str | None = None,
    run_after: datetime | None = None,
    activity_at: datetime | None = None,
    filtered_tokens: int | None = None,
    token_count_version: int | None = None,
) -> UUID:
    """INSERT a new extraction job row with status='pending'.

    Returns the generated UUID job id. Each keyword argument left as None is
    omitted from the INSERT, so the column default applies.

    If an active job (status NOT IN ('completed', 'failed')) already exists
    for the same (user_id, conversation_id, source), raises
    ExtractionJobAlreadyInFlight with the existing job's id so the caller
    can reuse it for idempotent enqueue semantics.

    The partial unique index idx_extraction_jobs_active_per_conversation
    (migration 0023) enforces this at the database level; we catch the
    UniqueViolationError and surface it as the typed exception. The INSERT
    runs in a savepoint so the lookup of the existing job still works when
    the caller holds an open transaction. When the conflicting job finished
    before the lookup found it, the INSERT is retried in a fresh savepoint,
    up to ``_CREATE_PENDING_ATTEMPTS`` times in total, then RuntimeError.
    """
    optional_columns = {
        "period_start": period_start,
        "lane": lane,
        "run_after": run_after,
        "activity_at": activity_at,
        "filtered_tokens": filtered_tokens,
        "token_count_version": token_count_version,
    }
    supplied = {name: value for name, value in optional_columns.items() if value is not None}
    columns = ["user_id", "conversation_id", "source", *supplied]
    values = [user_id, conversation_id, source, *supplied.values()]
    placeholders = ", ".join(f"${index}" for index in range(1, len(values) + 1))
    insert_sql = (
        f"INSERT INTO extraction_jobs ({', '.join(columns)}, status) "  # noqa: S608 - fixed column names
        f"VALUES ({placeholders}, 'pending') RETURNING id"
    )
    for _ in range(_CREATE_PENDING_ATTEMPTS):
        try:
            async with conn.transaction():
                row = await conn.fetchrow(insert_sql, *values)
        except asyncpg.UniqueViolationError:
            existing_id = await _find_in_flight_job(conn, user_id, conversation_id, source)
            if existing_id is not None:
                raise ExtractionJobAlreadyInFlight(existing_id) from None
            continue
        if row is None:
            raise RuntimeError(
                f"INSERT INTO extraction_jobs returned no row for "
                f"user_id={user_id} conversation_id={conversation_id} source={source!r}"
            )
        return UUID(str(row["id"]))
    raise RuntimeError(
        f"Conflict on (user_id={user_id}, conversation_id={conversation_id}, source={source}) "
        f"but no active row found after {_CREATE_PENDING_ATTEMPTS} attempts"
    )


async def _find_in_flight_job(
    conn: asyncpg.Connection,
    user_id: UUID,
    conversation_id: int,
    source: str,
) -> UUID | None:
    """Return the id of the active job that made create_pending's INSERT conflict.

    None when no active row exists any more: the conflicting job finished
    between the violation and this lookup.
    """
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
        return None
    return UUID(str(existing_row["id"]))


# ---------------------------------------------------------------------------
# refresh_queued
# ---------------------------------------------------------------------------


async def refresh_queued(
    conn: asyncpg.Connection,
    job_id: UUID,
    *,
    lane: str,
    run_after: datetime,
    activity_at: datetime,
    filtered_tokens: int,
    token_count_version: int,
) -> bool:
    """Replace the queue inputs of a job that is still pending and not yet admitted.

    Returns True when the row was rewritten. An admitted, running or finished
    job is left unchanged and False is returned: admission has already charged
    its token count against a budget period.
    """
    result = await conn.execute(
        """
        UPDATE extraction_jobs
        SET lane = $2,
            run_after = $3,
            activity_at = $4,
            filtered_tokens = $5,
            token_count_version = $6
        WHERE id = $1
          AND status = 'pending'
          AND admitted_at IS NULL
        """,
        job_id,
        lane,
        run_after,
        activity_at,
        filtered_tokens,
        token_count_version,
    )
    return str(result) == "UPDATE 1"


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


async def mark_running(conn: asyncpg.Connection, job_id: UUID) -> ClaimedFrom | None:
    """Claim a job for this execution: move it from 'pending' or 'failed' to 'running'.

    Returns the status the row was claimed from, or None, changing nothing,
    when the row is completed, already running, or not visible: another
    execution owns or finished the job. The row is locked before its status
    is read, so of concurrent claims exactly one succeeds. Reviving a failed
    row (a retry) resets started_at and clears the stale error_code.
    """
    claimed_from = await conn.fetchval(
        """
        UPDATE extraction_jobs AS job
        SET status = 'running',
            started_at = now(),
            error_code = NULL
        FROM (SELECT id, status FROM extraction_jobs WHERE id = $1 FOR UPDATE) AS prior
        WHERE job.id = prior.id
          AND prior.status IN ('pending', 'failed')
        RETURNING prior.status
        """,
        job_id,
    )
    return cast("ClaimedFrom | None", claimed_from)


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
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached_input_tokens: int = 0,
    reasoning_tokens: int = 0,
    llm_calls: int = 0,
    chunks: int = 0,
) -> bool:
    """Transition job to 'completed' and record final counters + timestamp.

    ``cents_spent`` and the usage counters are this run's totals over every
    model call, failed attempts included; they are ADDED to the row's
    counters, so a job revived after a failed run keeps that run's usage.

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
            cents_spent = cents_spent + $4,
            input_tokens = input_tokens + $5,
            output_tokens = output_tokens + $6,
            cached_input_tokens = cached_input_tokens + $7,
            reasoning_tokens = reasoning_tokens + $8,
            llm_calls = llm_calls + $9,
            chunks = chunks + $10,
            completed_at = now()
        WHERE id = $1
          AND status NOT IN ('completed', 'failed')
        """,
        job_id,
        topics_created,
        entries_created,
        cents_spent,
        input_tokens,
        output_tokens,
        cached_input_tokens,
        reasoning_tokens,
        llm_calls,
        chunks,
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
    cents_spent: int = 0,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached_input_tokens: int = 0,
    reasoning_tokens: int = 0,
    llm_calls: int = 0,
    chunks: int = 0,
) -> bool:
    """Transition job to 'failed' and record the error_code.

    ``cents_spent`` and the usage counters are this run's totals over every
    model call made before the failure, failed attempts included; they are
    ADDED to the row's counters, as in :func:`mark_completed`.

    WHERE clause guards against overwriting a terminal state -- no-op if
    status is already 'completed' or 'failed'.
    """
    result = await conn.execute(
        """
        UPDATE extraction_jobs
        SET status = 'failed',
            error_code = $2,
            cents_spent = cents_spent + $3,
            input_tokens = input_tokens + $4,
            output_tokens = output_tokens + $5,
            cached_input_tokens = cached_input_tokens + $6,
            reasoning_tokens = reasoning_tokens + $7,
            llm_calls = llm_calls + $8,
            chunks = chunks + $9,
            completed_at = now()
        WHERE id = $1
          AND status NOT IN ('completed', 'failed')
        """,
        job_id,
        error_code,
        cents_spent,
        input_tokens,
        output_tokens,
        cached_input_tokens,
        reasoning_tokens,
        llm_calls,
        chunks,
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
