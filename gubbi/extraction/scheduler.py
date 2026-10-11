"""The extraction queue scheduler: admits queued jobs and dispatches them to Arq.

One tick, serialized across API replicas by a session advisory lock:

1. Select runnable rows: queued ``pending`` rows and retryable ``failed`` rows
   whose ``run_after`` has passed, in lane order.
2. Decide admission for the rows not yet charged, against per-user and global
   limits (everything is admitted while budgets are disabled). A failed row
   that was charged before runs again without a second decision.
3. Dispatch at most ``dispatch_bound`` admitted jobs pending or running at
   once. Only rows dispatched this tick are charged; the rest stay queued and
   are decided again on a later tick, so a large backfill waits in Postgres,
   where later live work still overtakes it. A failed row moves back to
   ``pending`` before it is dispatched, so the worker claims it as usual.
4. Re-enqueue admitted rows still ``pending`` after ``stale_after``; a lost
   Arq job is restored. A row whose enqueue failed ``max_enqueue_failures``
   times in a row is failed as ``enqueue_lost`` and retried after the retry
   policy's backoff.

A row that does not fit a limit is deferred with a reason and never refused.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any, Final, Protocol

import structlog
from redis.exceptions import RedisError

from gubbi.extraction.admission import (
    Candidate,
    Decision,
    UserUsage,
    budget_period,
    decide,
    order_candidates,
    user_usage_from_totals,
)
from gubbi.extraction.limits import ExtractionLimits, limits_for
from gubbi.extraction.retry_policy import ErrorCode
from gubbi.extraction.supervised_loop import run_every
from gubbi.storage.pg_setup import advisory_unlock, try_advisory_lock
from gubbi.storage.repositories import extraction_jobs
from gubbi.storage.repositories import extraction_queue as queue_repo

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from uuid import UUID

    import asyncpg

    from gubbi.config import Settings
    from gubbi.storage.repositories.extraction_queue import RunnableJob

__all__: list[str] = [
    "EXTRACT_CONVERSATION_JOB",
    "SCHEDULER_LOCK_KEY",
    "SCHEDULER_TASK_NAME",
    "ArqEnqueuer",
    "ExtractionScheduler",
    "SchedulerConfig",
    "TickResult",
    "arq_job_id",
    "run_extraction_scheduler",
]

logger = structlog.get_logger(__name__)

# Unique across every advisory lock key in the application.
SCHEDULER_LOCK_KEY: Final = 1722469133
SCHEDULER_TASK_NAME: Final = "extraction_scheduler"
# Name the worker registers the extraction job function under.
EXTRACT_CONVERSATION_JOB: Final = "extract_conversation"

RUNNABLE_BATCH: Final = 500
STALE_DISPATCH_AGE: Final = timedelta(minutes=5)
MAX_ENQUEUE_FAILURES: Final = 3

_ENQUEUE_ERRORS: Final = (RedisError, OSError, TimeoutError)


class ArqEnqueuer(Protocol):
    """The part of the Arq pool the scheduler uses."""

    async def enqueue_job(
        self, function: str, *args: Any, _job_id: str | None = None, **kwargs: Any
    ) -> object:
        """Enqueue ``function``; returns None when a job with ``_job_id`` already exists."""
        ...


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    """Scheduler behaviour, normally read from settings."""

    paused: bool
    budget_enabled: bool
    dispatch_bound: int
    interval_secs: float = 30.0
    batch_size: int = RUNNABLE_BATCH
    stale_after: timedelta = STALE_DISPATCH_AGE
    max_enqueue_failures: int = MAX_ENQUEUE_FAILURES

    @classmethod
    def from_settings(cls, settings: Settings) -> SchedulerConfig:
        """Build the config from the extraction settings."""
        llm = settings.llm
        return cls(
            paused=llm.extraction_scheduler_paused,
            budget_enabled=llm.llm_budget_enabled,
            dispatch_bound=llm.extraction_dispatch_bound,
            interval_secs=llm.extraction_scheduler_interval_secs,
        )


@dataclass(frozen=True, slots=True)
class TickResult:
    """What one tick did; every field is a row count."""

    admitted: int = 0
    deferred: int = 0
    enqueued: int = 0
    enqueue_failed: int = 0
    reenqueued: int = 0
    enqueue_lost: int = 0


@dataclass(frozen=True, slots=True)
class _Plan:
    dispatch: tuple[RunnableJob, ...]
    deferrals: tuple[Decision, ...]


@dataclass(frozen=True, slots=True)
class _DispatchOutcome:
    enqueued_ids: frozenset[UUID]
    failed_ids: frozenset[UUID]


def arq_job_id(job: RunnableJob) -> str:
    """Arq job id of ``job``'s next run: each run gets its own id."""
    return f"{job.job_id}:{job.attempts}"


class ExtractionScheduler:
    """Runs scheduler ticks against the admin pool and the Arq pool."""

    def __init__(
        self,
        admin_pool: asyncpg.Pool,
        arq_pool: ArqEnqueuer,
        config: SchedulerConfig,
    ) -> None:
        self._admin_pool = admin_pool
        self._arq_pool = arq_pool
        self._config = config
        self._enqueue_failures: Mapping[UUID, int] = {}

    async def tick(self) -> TickResult | None:
        """Run one tick; None when paused or another instance holds the lock."""
        if self._config.paused:
            return None
        async with self._admin_pool.acquire() as conn:
            if not await try_advisory_lock(conn, SCHEDULER_LOCK_KEY):
                return None
            try:
                result = await self._locked_tick(conn)
            finally:
                await advisory_unlock(conn, SCHEDULER_LOCK_KEY)
        await _log_tick(result)
        return result

    async def _locked_tick(self, conn: asyncpg.Connection) -> TickResult:
        period = budget_period(datetime.now(UTC))
        async with conn.transaction():
            plan = await self._plan(conn, period)
            dispatch = await _claim(conn, plan.dispatch, period)
            for decision in plan.deferrals:
                await queue_repo.defer(conn, decision.job_id, str(decision.defer_reason))
        dispatched = await self._enqueue_all(dispatch)
        attempted = dispatched.enqueued_ids | dispatched.failed_ids
        stale = [
            job
            for job in await queue_repo.stale_admitted_pending(conn, self._config.stale_after)
            if job.job_id not in attempted
        ]
        redispatched = await self._enqueue_all(stale)
        failed = dispatched.failed_ids | redispatched.failed_ids
        lost = await self._record_enqueue_failures(
            conn, failed, dispatched.enqueued_ids | redispatched.enqueued_ids
        )
        return TickResult(
            admitted=sum(1 for job in dispatch if job.admitted_at is None),
            deferred=len(plan.deferrals),
            enqueued=len(dispatched.enqueued_ids),
            enqueue_failed=len(failed),
            reenqueued=len(redispatched.enqueued_ids),
            enqueue_lost=lost,
        )

    async def _plan(self, conn: asyncpg.Connection, period: date) -> _Plan:
        jobs = await queue_repo.select_runnable(conn, self._config.batch_size)
        if not jobs:
            return _Plan(dispatch=(), deferrals=())
        fresh = [job for job in jobs if job.admitted_at is None]
        decisions = await self._decide(conn, fresh, period)
        capacity = self._config.dispatch_bound - await queue_repo.inflight_admitted_count(conn)
        return _plan_dispatch(jobs, decisions, capacity)

    async def _decide(
        self, conn: asyncpg.Connection, fresh: Sequence[RunnableJob], period: date
    ) -> list[Decision]:
        candidates = [_candidate(job) for job in fresh]
        if not candidates:
            return []
        if not self._config.budget_enabled:
            return decide(candidates, {}, 0, None)
        # decide applies one limits set to every candidate; limits_for returns
        # the same values for every user today.
        limits = limits_for(candidates[0].user_id)
        usage = await _usage_by_user(conn, {c.user_id for c in candidates}, period, limits)
        global_used = await queue_repo.global_period_tokens(conn, period)
        return decide(candidates, usage, global_used, limits)

    async def _enqueue_all(self, jobs: Sequence[RunnableJob]) -> _DispatchOutcome:
        enqueued: set[UUID] = set()
        failed: set[UUID] = set()
        for job in jobs:
            try:
                await self._arq_pool.enqueue_job(
                    EXTRACT_CONVERSATION_JOB,
                    job.conversation_id,
                    str(job.user_id),
                    str(job.job_id),
                    _job_id=arq_job_id(job),
                )
            except _ENQUEUE_ERRORS as exc:
                await logger.warning("extraction_enqueue_failed", error_class=type(exc).__name__)
                failed.add(job.job_id)
            else:
                enqueued.add(job.job_id)
        return _DispatchOutcome(enqueued_ids=frozenset(enqueued), failed_ids=frozenset(failed))

    async def _record_enqueue_failures(
        self,
        conn: asyncpg.Connection,
        failed_ids: frozenset[UUID],
        enqueued_ids: frozenset[UUID],
    ) -> int:
        """Count consecutive enqueue failures per job; fail jobs that reached the limit.

        A successful enqueue resets a job's count. Returns the jobs failed.
        """
        counts = {
            job_id: n for job_id, n in self._enqueue_failures.items() if job_id not in enqueued_ids
        }
        for job_id in failed_ids:
            counts[job_id] = counts.get(job_id, 0) + 1
        lost = {job_id for job_id, n in counts.items() if n >= self._config.max_enqueue_failures}
        failed = 0
        for job_id in lost:
            failed += await _fail_enqueue_lost(conn, job_id)
        self._enqueue_failures = {job_id: n for job_id, n in counts.items() if job_id not in lost}
        return failed


def _plan_dispatch(
    jobs: Sequence[RunnableJob],
    decisions: Sequence[Decision],
    capacity: int,
) -> _Plan:
    """Pick the jobs to dispatch and the deferrals to record, in admission order.

    A job charged before (``admitted_at`` set) is dispatchable without a
    decision. Dispatch takes admitted jobs until ``capacity`` is reached. The
    walk stops at the first admitted job past capacity: every later decision
    counted that job's usage, which this tick does not charge, so later
    deferrals are not recorded either.
    """
    by_id = {d.job_id: d for d in decisions}
    jobs_by_id = {job.job_id: job for job in jobs}
    dispatch: list[RunnableJob] = []
    deferrals: list[Decision] = []
    for candidate in order_candidates(_candidate(job) for job in jobs):
        job = jobs_by_id[candidate.job_id]
        decision = by_id.get(job.job_id, Decision(job.job_id, None))
        if not decision.admitted:
            deferrals.append(decision)
            continue
        if len(dispatch) >= capacity:
            break
        dispatch.append(job)
    return _Plan(dispatch=tuple(dispatch), deferrals=tuple(deferrals))


async def _claim(
    conn: asyncpg.Connection, jobs: Sequence[RunnableJob], period: date
) -> list[RunnableJob]:
    """Charge and revive the planned jobs; return those now admitted and pending.

    A failed job that cannot move back to pending (its conversation has an
    active job) is dropped and not charged.
    """
    failed = [job.job_id for job in jobs if job.status == "failed"]
    revived = await queue_repo.revive_failed(conn, failed) if failed else set()
    claimed = [job for job in jobs if job.status != "failed" or job.job_id in revived]
    fresh = [job.job_id for job in claimed if job.admitted_at is None]
    if fresh:
        await queue_repo.admit(conn, fresh, period)
    return claimed


async def _fail_enqueue_lost(conn: asyncpg.Connection, job_id: UUID) -> int:
    """Fail a job as ``enqueue_lost`` unless a worker claimed it meanwhile; return rows failed."""
    async with conn.transaction():
        if not await queue_repo.lock_if_pending(conn, job_id):
            return 0
        failed = await extraction_jobs.mark_failed(conn, job_id, error_code=ErrorCode.ENQUEUE_LOST)
    return int(failed)


def _candidate(job: RunnableJob) -> Candidate:
    return Candidate(
        job_id=job.job_id,
        user_id=job.user_id,
        lane=job.lane,
        activity_at=job.activity_at,
        filtered_tokens=job.filtered_tokens,
    )


async def _usage_by_user(
    conn: asyncpg.Connection,
    user_ids: set[UUID],
    period: date,
    limits: ExtractionLimits,
) -> dict[UUID, UserUsage]:
    totals = await queue_repo.user_period_usage(conn, user_ids, period)
    return {
        user_id: user_usage_from_totals(**asdict(total), limits=limits)
        for user_id, total in totals.items()
    }


async def _log_tick(result: TickResult) -> None:
    if result == TickResult():
        return
    await logger.info("extraction_scheduler_tick", **asdict(result))


async def run_extraction_scheduler(
    admin_pool: asyncpg.Pool,
    arq_pool: ArqEnqueuer,
    config: SchedulerConfig,
) -> None:
    """Tick every ``config.interval_secs`` until cancelled."""
    scheduler = ExtractionScheduler(admin_pool, arq_pool, config)
    await run_every(SCHEDULER_TASK_NAME, scheduler.tick, interval_secs=config.interval_secs)
