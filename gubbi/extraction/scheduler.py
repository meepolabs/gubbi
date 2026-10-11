"""The extraction queue scheduler: admits queued jobs and dispatches them to Arq.

One tick, serialized across API replicas by a session advisory lock:

1. End every retryable ``failed`` row that a newer job of its conversation
   replaces, as ``superseded``. Then lock runnable rows: queued ``pending``
   rows and retryable ``failed`` rows whose ``run_after`` has passed, in lane
   order. A failed row whose conversation already has an active job is left
   for a later tick, and so is a row with no recorded token count.
2. Decide admission for the rows not yet charged, against per-user and global
   limits (everything is admitted while budgets are disabled). A failed row
   that was charged before runs again without a second decision or charge.
3. Dispatch at most ``dispatch_bound`` admitted jobs pending or running at
   once. Only rows dispatched this tick are charged; a row that fits but finds
   no capacity stays queued and uncharged, so a large backfill waits in
   Postgres, where later live work still overtakes it. Usage is counted over
   dispatched rows only, so every row deferred for a limit is a real
   deferral; it is rechecked at the earlier of the next budget period and
   ``defer_recheck`` from now. A failed row moves back to ``pending`` before
   it is dispatched, so the worker claims it as usual.
4. Re-enqueue admitted rows still ``pending`` after ``stale_after``; a lost
   Arq job is restored. A row whose enqueue raises is failed as
   ``enqueue_lost`` and revived after the retry policy's backoff, which
   grows with each consecutive lost enqueue.
5. Record the queue and budget gauges from aggregate reads taken after the
   claim transaction, so they reflect this tick's admissions and deferrals.

A row that does not fit a limit is deferred with a reason and never refused.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any, Final, Protocol

import asyncpg
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
from gubbi.extraction.failure_metrics import record_job_failures
from gubbi.extraction.limits import ExtractionLimits, limits_for
from gubbi.extraction.queue_metrics import record_queue_metrics
from gubbi.extraction.retry_policy import ErrorCode
from gubbi.extraction.supervised_loop import run_every
from gubbi.storage.pg_setup import advisory_unlock, try_advisory_lock
from gubbi.storage.repositories import extraction_queue as queue_repo

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterable, Sequence
    from uuid import UUID

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
DEFER_RECHECK: Final = timedelta(hours=1)

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
    defer_recheck: timedelta = DEFER_RECHECK

    @classmethod
    def from_settings(cls, settings: Settings) -> SchedulerConfig:
        """Build the config from the extraction settings."""
        llm = settings.llm
        return cls(
            paused=llm.extraction_scheduler_paused,
            budget_enabled=llm.llm_budget_enabled,
            dispatch_bound=llm.extraction_dispatch_bound,
            interval_secs=llm.extraction_scheduler_interval_secs,
            defer_recheck=timedelta(seconds=llm.extraction_defer_recheck_secs),
        )


@dataclass(frozen=True, slots=True)
class TickResult:
    """What one tick did; every field is a row count."""

    admitted: int = 0
    superseded: int = 0
    deferred: int = 0
    enqueued: int = 0
    enqueue_failed: int = 0
    reenqueued: int = 0
    enqueue_lost: int = 0


type _Decider = Callable[[Iterable[RunnableJob]], list[Decision]]


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
            superseded = await queue_repo.supersede_failed(conn)
            plan = await self._plan(conn, period)
            dispatch = await _claim(conn, plan.dispatch, period)
            await self._record_deferrals(conn, plan.deferrals, period)
        dispatched = await self._enqueue_all(dispatch)
        attempted = dispatched.enqueued_ids | dispatched.failed_ids
        stale = [
            job
            for job in await queue_repo.stale_admitted_pending(conn, self._config.stale_after)
            if job.job_id not in attempted
        ]
        redispatched = await self._enqueue_all(stale)
        failed = dispatched.failed_ids | redispatched.failed_ids
        lost = 0
        for job_id in failed:
            outcome = await queue_repo.fail_enqueue_lost(conn, job_id)
            if outcome is not None:
                lost += 1
                record_job_failures(ErrorCode.ENQUEUE_LOST, outcome)
        await self._record_metrics(conn, period)
        return TickResult(
            admitted=sum(1 for job in dispatch if job.admitted_at is None),
            superseded=superseded,
            deferred=len(plan.deferrals),
            enqueued=len(dispatched.enqueued_ids),
            enqueue_failed=len(failed),
            reenqueued=len(redispatched.enqueued_ids),
            enqueue_lost=lost,
        )

    async def _record_metrics(self, conn: asyncpg.Connection, period: date) -> None:
        """Record the queue gauges; a failed read loses this tick's gauges, not its work."""
        try:
            await record_queue_metrics(conn, period, budget_enabled=self._config.budget_enabled)
        except (asyncpg.PostgresError, asyncpg.InterfaceError, OSError) as exc:
            await logger.warning("extraction_queue_metrics_failed", error_class=type(exc).__name__)

    async def _plan(self, conn: asyncpg.Connection, period: date) -> _Plan:
        jobs = await queue_repo.select_runnable(conn, self._config.batch_size)
        if not jobs:
            return _Plan(dispatch=(), deferrals=())
        capacity = self._config.dispatch_bound - await queue_repo.inflight_admitted_count(conn)
        fresh = [job for job in jobs if job.admitted_at is None]
        decide_fresh = await self._decider(conn, fresh, period)
        unplaced: frozenset[UUID] = frozenset()
        while True:
            decisions = decide_fresh(job for job in fresh if job.job_id not in unplaced)
            plan, overflow = _plan_dispatch(jobs, decisions, capacity, unplaced)
            if not overflow:
                return plan
            unplaced |= overflow

    async def _decider(
        self, conn: asyncpg.Connection, fresh: Sequence[RunnableJob], period: date
    ) -> _Decider:
        """Return a function deciding a subset of ``fresh`` against the usage read now."""
        if not fresh or not self._config.budget_enabled:
            return lambda jobs: decide([_candidate(job) for job in jobs], {}, 0, None)
        # decide applies one limits set to every candidate; limits_for returns
        # the same values for every user today.
        limits = limits_for(fresh[0].user_id)
        usage = await _usage_by_user(conn, {job.user_id for job in fresh}, period, limits)
        global_used = await queue_repo.global_period_tokens(conn, period)
        return lambda jobs: decide([_candidate(job) for job in jobs], usage, global_used, limits)

    async def _record_deferrals(
        self, conn: asyncpg.Connection, deferrals: Sequence[Decision], period: date
    ) -> None:
        by_reason: dict[str, list[UUID]] = {}
        for decision in deferrals:
            by_reason.setdefault(str(decision.defer_reason), []).append(decision.job_id)
        for reason, job_ids in by_reason.items():
            await queue_repo.defer(
                conn,
                job_ids,
                reason,
                next_period_start=_next_period_start(period),
                recheck=self._config.defer_recheck,
            )

    async def _enqueue_all(self, jobs: Sequence[RunnableJob]) -> _DispatchOutcome:
        enqueued: set[UUID] = set()
        failed: set[UUID] = set()
        for job in jobs:
            try:
                # Arq keeps no job results, so a None return means a job with
                # this id is already queued or running: treated as enqueued.
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


def _plan_dispatch(
    jobs: Sequence[RunnableJob],
    decisions: Sequence[Decision],
    capacity: int,
    unplaced: Collection[UUID],
) -> tuple[_Plan, frozenset[UUID]]:
    """Pick the jobs to dispatch and the deferrals to record, in admission order.

    A job charged before (``admitted_at`` set) needs no decision. Dispatch
    takes admitted jobs until ``capacity`` is reached. ``unplaced`` jobs fit
    but found no capacity on an earlier pass; they were left out of
    ``decisions`` and are skipped. Returns the plan and the newly admitted
    jobs past capacity: the deferrals after them were decided with their
    usage counted, so the caller decides again without them until none remain.
    """
    by_id = {d.job_id: d for d in decisions}
    jobs_by_id = {job.job_id: job for job in jobs}
    dispatch: list[RunnableJob] = []
    deferrals: list[Decision] = []
    overflow: set[UUID] = set()
    for candidate in order_candidates(_candidate(job) for job in jobs):
        if candidate.job_id in unplaced:
            continue
        job = jobs_by_id[candidate.job_id]
        decision = by_id.get(job.job_id, Decision(job.job_id, None))
        if not decision.admitted:
            deferrals.append(decision)
        elif len(dispatch) < capacity:
            dispatch.append(job)
        elif job.admitted_at is None:
            overflow.add(job.job_id)
    plan = _Plan(dispatch=tuple(dispatch), deferrals=tuple(deferrals))
    return plan, frozenset(overflow)


async def _claim(
    conn: asyncpg.Connection, jobs: Sequence[RunnableJob], period: date
) -> list[RunnableJob]:
    """Charge and revive the planned jobs; return those now admitted and pending.

    A failed job that cannot move back to pending (an ingest gave its
    conversation an active job meanwhile) is dropped and not charged.
    """
    failed = [job.job_id for job in jobs if job.status == "failed"]
    revived = await queue_repo.revive_failed(conn, failed) if failed else set()
    claimed = [job for job in jobs if job.status != "failed" or job.job_id in revived]
    fresh = [job.job_id for job in claimed if job.admitted_at is None]
    if fresh:
        await queue_repo.admit(conn, fresh, period)
    return claimed


def _next_period_start(period: date) -> datetime:
    following = (period.replace(day=28) + timedelta(days=4)).replace(day=1)
    return datetime(following.year, following.month, 1, tzinfo=UTC)


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
