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
refresh_queued      -- rewrite the queue inputs of a pending, not yet dispatched job
mark_running        -- claim a dispatched, runnable pending job and count the attempt
update_progress     -- monotonic UPDATE of topics/entries counters (worker stub)
mark_completed      -- terminal UPDATE to 'completed'; adds this run's usage to the row
mark_failed         -- UPDATE to 'failed' with the retry policy's run_after; adds this run's usage;
                       returns whether the job runs again
finished_attempt    -- identity of a job's latest attempt and whether its conversation changed
                       during it
create_follow_up    -- queue a new job for a conversation re-saved during a finished run
get_status_counts   -- aggregate SELECT returning StatusCounts (for /me endpoint)
ExtractionJobAlreadyInFlight -- raised by create_pending on partial-unique conflict
FinishedAttempt     -- dataclass returned by finished_attempt
StatusCounts        -- dataclass returned by get_status_counts
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final
from uuid import UUID

import asyncpg

from gubbi.extraction.retry_policy import (
    ErrorCode,
    FailureOutcome,
    failure_outcome,
    next_run_after,
)

if TYPE_CHECKING:
    from datetime import date, datetime

_CREATE_PENDING_ATTEMPTS: Final = 3

__all__: list[str] = [
    "ExtractionJobAlreadyInFlight",
    "FinishedAttempt",
    "StatusCounts",
    "create_follow_up",
    "create_pending",
    "finished_attempt",
    "get_status_counts",
    "mark_completed",
    "mark_failed",
    "mark_running",
    "refresh_queued",
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
    period_start: date | None = None,
    lane: str | None = None,
    run_after: datetime | None = None,
    activity_at: datetime | None = None,
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
    the caller holds an open transaction. The lookup locks the active row
    until the caller's transaction ends. When the conflicting job finished
    before the lookup found it, the INSERT is retried in a fresh savepoint,
    up to ``_CREATE_PENDING_ATTEMPTS`` times in total, then RuntimeError.
    """
    optional_columns = {
        "period_start": period_start,
        "lane": lane,
        "run_after": run_after,
        "activity_at": activity_at,
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
    """Lock and return the id of the active job that made create_pending's INSERT conflict.

    The row lock holds until the caller's transaction ends, so the job cannot
    finish while the caller still refreshes and commits work for it. None when
    no active row exists any more: the conflicting job finished between the
    violation and this lookup.
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
        FOR UPDATE
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
) -> bool:
    """Replace the queue inputs of a job that is still pending and not yet dispatched.

    Returns True when the row was rewritten. A dispatched, running or
    finished job is left unchanged and False is returned.
    """
    result = await conn.execute(
        """
        UPDATE extraction_jobs
        SET lane = $2,
            run_after = $3,
            activity_at = $4
        WHERE id = $1
          AND status = 'pending'
          AND admitted_at IS NULL
        """,
        job_id,
        lane,
        run_after,
        activity_at,
    )
    return str(result) == "UPDATE 1"


# ---------------------------------------------------------------------------
# mark_running
# ---------------------------------------------------------------------------


async def mark_running(conn: asyncpg.Connection, job_id: UUID) -> bool:
    """Claim a job for this execution: move it from 'pending' to 'running'.

    Only a row the scheduler dispatched and whose ``run_after`` has passed is
    claimed. Returns False, changing nothing, for any other row: one not yet
    dispatched, failed (the scheduler moves a retryable failed row back to
    pending before it dispatches it again), backed off, running, completed
    or not visible. A delayed or duplicate delivery of a job is therefore a
    no-op. The row is locked before its status is read, so of concurrent
    claims exactly one succeeds. Every claim counts one attempt, resets
    started_at and dispatch_failures and clears the previous run's
    error_code.
    """
    claimed = await conn.fetchval(
        """
        UPDATE extraction_jobs AS job
        SET status = 'running',
            started_at = now(),
            error_code = NULL,
            attempts = job.attempts + 1,
            dispatch_failures = 0
        FROM (
            SELECT id, status, admitted_at, run_after
            FROM extraction_jobs
            WHERE id = $1
            FOR UPDATE
        ) AS prior
        WHERE job.id = prior.id
          AND prior.status = 'pending'
          AND prior.admitted_at IS NOT NULL
          AND prior.run_after <= now()
        RETURNING true
        """,
        job_id,
    )
    return bool(claimed)


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
    error_code: ErrorCode,
    cents_spent: int = 0,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached_input_tokens: int = 0,
    reasoning_tokens: int = 0,
    llm_calls: int = 0,
    chunks: int = 0,
) -> FailureOutcome | None:
    """Transition job to 'failed', record the error_code and when it may run again.

    ``run_after`` comes from :func:`retry_policy.next_run_after` over the
    row's ``attempts``; a failure the policy never retries is held at
    ``'infinity'``. ``cents_spent`` and the usage counters are this run's
    totals over every model call made before the failure, failed attempts
    included; they are ADDED to the row's counters, as in :func:`mark_completed`.

    Returns whether the job runs again, or None, changing nothing, if the
    row is already 'completed' or 'failed' or is not visible. Runs in a
    savepoint so the row lock taken to read ``attempts`` is held until the
    UPDATE.
    """
    async with conn.transaction():
        row = await conn.fetchrow(
            """
            SELECT attempts, clock_timestamp() AS db_now
            FROM extraction_jobs
            WHERE id = $1
              AND status NOT IN ('completed', 'failed')
            FOR UPDATE
            """,
            job_id,
        )
        if row is None:
            return None
        attempts = int(row["attempts"])
        run_after = _run_after_text(error_code, attempts, row["db_now"])
        result = await conn.execute(
            """
            UPDATE extraction_jobs
            SET status = 'failed',
                error_code = $2,
                run_after = $3::text::timestamptz,
                cents_spent = cents_spent + $4,
                input_tokens = input_tokens + $5,
                output_tokens = output_tokens + $6,
                cached_input_tokens = cached_input_tokens + $7,
                reasoning_tokens = reasoning_tokens + $8,
                llm_calls = llm_calls + $9,
                chunks = chunks + $10,
                completed_at = now()
            WHERE id = $1
            """,
            job_id,
            error_code.value,
            run_after,
            cents_spent,
            input_tokens,
            output_tokens,
            cached_input_tokens,
            reasoning_tokens,
            llm_calls,
            chunks,
        )
    if str(result) != "UPDATE 1":
        return None
    return failure_outcome(error_code, max(attempts, 1))


def _run_after_text(code: ErrorCode, attempts: int, now: datetime) -> str:
    """``run_after`` for a failure as timestamptz text; ``'infinity'`` when never retried.

    The policy counts from the first run, so a row failed before any claim
    is treated as on its first attempt.
    """
    retry_at = next_run_after(code, max(attempts, 1), now)
    return "infinity" if retry_at is None else retry_at.isoformat()


# ---------------------------------------------------------------------------
# finished_attempt / create_follow_up
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FinishedAttempt:
    """The latest attempt of a job, read in the transaction that ends it.

    attempt            : the row's ``attempts``, the number of this attempt
    resaved_during_run : the conversation holds a message above its extraction
                         watermark; with no watermark yet, it was saved again
                         after this attempt started. Either way the run may
                         not have read its latest content
    """

    job_id: UUID
    user_id: UUID
    conversation_id: int
    source: str
    lane: str
    attempt: int
    resaved_during_run: bool


async def finished_attempt(conn: asyncpg.Connection, job_id: UUID) -> FinishedAttempt | None:
    """Return the latest attempt of ``job_id``; None when the row is not visible."""
    row = await conn.fetchrow(
        """
        SELECT j.user_id, j.conversation_id, j.source, j.lane, j.attempts,
               coalesce(
                   CASE
                       WHEN c.extracted_through_position IS NULL
                           THEN c.updated_at > j.started_at
                       ELSE (
                           SELECT max(m.position) FROM messages AS m
                           WHERE m.conversation_id = c.id
                       ) > c.extracted_through_position
                   END,
                   false
               ) AS resaved_during_run
        FROM extraction_jobs AS j
        LEFT JOIN conversations AS c ON c.id = j.conversation_id
        WHERE j.id = $1
        """,
        job_id,
    )
    if row is None:
        return None
    return FinishedAttempt(
        job_id=job_id,
        user_id=UUID(str(row["user_id"])),
        conversation_id=int(row["conversation_id"]),
        source=str(row["source"]),
        lane=str(row["lane"]),
        attempt=int(row["attempts"]),
        resaved_during_run=bool(row["resaved_during_run"]),
    )


async def create_follow_up(
    conn: asyncpg.Connection, attempt: FinishedAttempt, run_after: datetime
) -> UUID | None:
    """Queue a new pending job for the conversation of a run that read stale content.

    Must run inside the caller's transaction, after the finished job left the
    active statuses. The conversation's ``processed_at`` is cleared so the new
    job extracts it again; only messages above its extraction watermark are
    new to that job, so content already extracted is not extracted twice. The
    new job's ``activity_at`` is the conversation's latest message timestamp
    as saved now, else its ``updated_at``, never later than now. Returns the
    new job id, or None when another job of the conversation is already active.
    """
    activity_at = await conn.fetchval(
        """
        UPDATE conversations AS c
        SET processed_at = NULL
        WHERE c.id = $1
        RETURNING least(
            coalesce(
                (SELECT max(m.timestamp) FROM messages AS m WHERE m.conversation_id = c.id),
                c.updated_at
            ),
            now()
        )
        """,
        attempt.conversation_id,
    )
    try:
        return await create_pending(
            conn,
            attempt.user_id,
            attempt.conversation_id,
            attempt.source,
            lane=attempt.lane,
            run_after=run_after,
            activity_at=activity_at,
        )
    except ExtractionJobAlreadyInFlight:
        return None


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
