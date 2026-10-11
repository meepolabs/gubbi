"""The extraction queue scheduler: admits queued jobs and dispatches them to Arq.

One tick, serialized across API replicas by a session advisory lock:

1. End every retryable ``failed`` row that a newer job of its conversation
   replaces, as ``superseded``. Then lock runnable rows: queued ``pending``
   rows and retryable ``failed`` rows whose ``run_after`` has passed, in lane
   order, at most the largest running-job cap per user. A failed row whose conversation
   already has an active job is left for a later tick.
2. Resolve each distinct user's wallet limits, then read in three batched
   queries each user's available balance, each user's jobs in flight and the
   deployment's weighted usage this month (the balances only while budgets
   are enabled). Decide every row: the global ceiling share and the user's
   stop line of its lane's spend rule, then the user's running-job cap and
   the dispatch capacity (``dispatch_bound`` minus jobs already in flight).
3. A user stopped by a lane's stop line has every runnable row of that lane
   deferred in one statement, and a lane stopped by the global ceiling is
   deferred for every user; both are rechecked at the earlier of the next
   UTC month and ``defer_recheck`` from now, so a credit grant takes effect
   within one recheck. A row held back by the running-job cap or capacity
   stays queued for the next tick. A failed row moves back to ``pending``
   before it is dispatched, so the worker claims it as usual.
4. Re-enqueue dispatched rows still ``pending`` after ``stale_after``; a lost
   Arq job is restored. A row whose enqueue raises is failed as
   ``enqueue_lost`` and revived after the retry policy's backoff, which
   grows with each consecutive lost enqueue.
5. Record the queue and budget gauges from aggregate reads taken after the
   claim transaction, so they reflect this tick's dispatches and deferrals.

Nothing is charged here: a job is charged after it runs, from the usage the
provider reported. A row that does not fit is deferred or left queued, never
refused.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Final, Protocol

import asyncpg
import structlog
from redis.exceptions import RedisError

from gubbi.extraction.admission import (
    Budget,
    Candidate,
    Decision,
    DeferReason,
    UserState,
    decide,
)
from gubbi.extraction.failure_metrics import record_job_failures
from gubbi.extraction.queue_metrics import record_queue_metrics
from gubbi.extraction.retry_policy import ErrorCode
from gubbi.extraction.spend_rules import global_ceiling, limits_for, max_running_cap
from gubbi.extraction.supervised_loop import run_every
from gubbi.storage.pg_setup import advisory_unlock, try_advisory_lock
from gubbi.storage.repositories import extraction_queue as queue_repo
from gubbi.storage.repositories import wallet as wallet_repo

if TYPE_CHECKING:
    from collections.abc import Sequence
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
    """What one tick did; every field is a row count.

    ``admitted`` counts rows dispatched by this tick's decisions, revived
    failed rows included; ``reenqueued`` counts lost dispatches sent again.
    """

    admitted: int = 0
    superseded: int = 0
    deferred: int = 0
    enqueued: int = 0
    enqueue_failed: int = 0
    reenqueued: int = 0
    enqueue_lost: int = 0


@dataclass(frozen=True, slots=True)
class _Plan:
    dispatch: tuple[RunnableJob, ...]
    deferrals: frozenset[_Deferral]


@dataclass(frozen=True, slots=True)
class _Deferral:
    """A lane to defer, for one user or (``user_id`` None) for every user."""

    lane: str
    reason: str
    user_id: UUID | None


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
        async with conn.transaction():
            superseded = await queue_repo.supersede_failed(conn)
            plan = await self._plan(conn)
            dispatch = await _claim(conn, plan.dispatch)
            deferred = await self._record_deferrals(conn, plan.deferrals)
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
        await self._record_metrics(conn)
        return TickResult(
            admitted=len(dispatch),
            superseded=superseded,
            deferred=deferred,
            enqueued=len(dispatched.enqueued_ids),
            enqueue_failed=len(failed),
            reenqueued=len(redispatched.enqueued_ids),
            enqueue_lost=lost,
        )

    async def _record_metrics(self, conn: asyncpg.Connection) -> None:
        """Record the queue gauges; a failed read loses this tick's gauges, not its work."""
        try:
            await record_queue_metrics(conn, budget_enabled=self._config.budget_enabled)
        except (asyncpg.PostgresError, asyncpg.InterfaceError, OSError) as exc:
            await logger.warning("extraction_queue_metrics_failed", error_class=type(exc).__name__)

    async def _plan(self, conn: asyncpg.Connection) -> _Plan:
        jobs = await queue_repo.select_runnable(conn, self._config.batch_size, max_running_cap())
        if not jobs:
            return _Plan(dispatch=(), deferrals=frozenset())
        capacity = self._config.dispatch_bound - await queue_repo.inflight_admitted_count(conn)
        users = await self._user_states(conn, {job.user_id for job in jobs})
        budget = await self._budget(conn)
        decisions = decide(
            [_candidate(job) for job in jobs], users, budget, capacity=max(capacity, 0)
        )
        jobs_by_id = {job.job_id: job for job in jobs}
        return _Plan(
            dispatch=tuple(jobs_by_id[d.job_id] for d in decisions if d.admitted),
            deferrals=frozenset(_deferral(d) for d in decisions if d.defer_reason is not None),
        )

    async def _user_states(
        self, conn: asyncpg.Connection, user_ids: set[UUID]
    ) -> dict[UUID, UserState]:
        limits = {user_id: limits_for(user_id) for user_id in user_ids}
        inflight = await queue_repo.inflight_by_user(conn, user_ids)
        available = (
            await wallet_repo.available(
                conn, {user_id: lim.monthly_amount for user_id, lim in limits.items()}
            )
            if self._config.budget_enabled
            else {}
        )
        return {
            user_id: UserState(
                limits=lim,
                available=available.get(user_id, 0),
                inflight=inflight[user_id],
            )
            for user_id, lim in limits.items()
        }

    async def _budget(self, conn: asyncpg.Connection) -> Budget | None:
        if not self._config.budget_enabled:
            return None
        return Budget(global_used=await wallet_repo.global_used(conn), ceiling=global_ceiling())

    async def _record_deferrals(
        self, conn: asyncpg.Connection, deferrals: frozenset[_Deferral]
    ) -> int:
        deferred = 0
        for deferral in sorted(deferrals, key=_deferral_order):
            deferred += await queue_repo.defer_lane(
                conn,
                deferral.lane,
                deferral.reason,
                user_id=deferral.user_id,
                recheck=self._config.defer_recheck,
            )
        return deferred

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


async def _claim(conn: asyncpg.Connection, jobs: Sequence[RunnableJob]) -> list[RunnableJob]:
    """Revive and mark dispatched the planned jobs; return those now dispatched and pending.

    A failed job that cannot move back to pending (an ingest gave its
    conversation an active job meanwhile) is dropped.
    """
    failed = [job.job_id for job in jobs if job.status == "failed"]
    revived = await queue_repo.revive_failed(conn, failed) if failed else set()
    claimed = [job for job in jobs if job.status != "failed" or job.job_id in revived]
    if claimed:
        await queue_repo.admit(conn, [job.job_id for job in claimed])
    return claimed


def _deferral(decision: Decision) -> _Deferral:
    """The lane deferral a deferred decision calls for: per user, or lane-wide for the ceiling."""
    reason = decision.defer_reason
    user_id = None if reason is DeferReason.GLOBAL else decision.user_id
    return _Deferral(lane=decision.lane.value, reason=str(reason), user_id=user_id)


def _deferral_order(deferral: _Deferral) -> tuple[str, str, str]:
    return (deferral.lane, deferral.reason, str(deferral.user_id))


def _candidate(job: RunnableJob) -> Candidate:
    return Candidate(
        job_id=job.job_id,
        user_id=job.user_id,
        lane=job.lane,
        activity_at=job.activity_at,
    )


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
