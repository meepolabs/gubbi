"""Stuck-running extraction_jobs sweep.

A row stays in 'running' indefinitely if the Arq worker dies between
mark_running and the SAVEPOINT commit (OOM, container kill, hard crash).
The sweep fails such a row as ``worker_lost``, a transient error, with a
backoff ``run_after`` so the scheduler runs it again.

The threshold is EXTRACTION_JOB_TIMEOUT_SECS + ARQ_JOB_TIMEOUT_SECS
(40 minutes): past the extraction function's own Arq timeout, at which
Arq cancels the job, so a live long-running job is never swept.

A running row with no ``started_at`` is measured from its ``created_at``.

Pending rows are never swept: a pending row is queued work the scheduler
owns, however long it waits for its lane or budget.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import asyncpg
import structlog
from opentelemetry.metrics import Counter, get_meter

from gubbi.constants import ARQ_JOB_TIMEOUT_SECS, EXTRACTION_JOB_TIMEOUT_SECS
from gubbi.extraction.failure_metrics import record_job_failures
from gubbi.extraction.retry_policy import ErrorCode, failure_outcome, next_run_after
from gubbi.extraction.supervised_loop import run_every

if TYPE_CHECKING:
    from datetime import datetime

__all__: list[str] = ["ORPHAN_CLEANUP_TASK_NAME", "run_orphan_cleanup"]

logger = structlog.get_logger(__name__)

ORPHAN_CLEANUP_TASK_NAME = "orphan_cleanup"


@functools.lru_cache(maxsize=1)
def _get_orphan_cleanup_swept_counter() -> Counter:
    """Lazily create the orphan-cleanup swept counter against the live meter.

    A module-scope ``create_counter`` would bind at import time, before
    ``configure_otel`` runs during the FastAPI lifespan, to a NoOp
    instrument that discards every ``.add(...)``. Deferring creation to
    first call (and re-priming via ``rebind_metrics_after_configure``)
    binds the counter to the SDK provider configured at lifespan time.
    Mirrors the canonical pattern in ``gubbi.telemetry.metrics.initialize_metrics``.
    """
    return get_meter("gubbi").create_counter(
        name="extraction_jobs.orphan_cleanup_swept_total",
        description="Count of orphan extraction_jobs sweep cycles, partitioned by state",
        unit="1",
    )


# Stuck-running threshold: any row still 'running' beyond this is presumed
# worker-lost (worker died between mark_running and SAVEPOINT commit).
# Arq cancels extract_conversation at EXTRACTION_JOB_TIMEOUT_SECS, so no live
# run outlasts it; the extra ARQ_JOB_TIMEOUT_SECS is margin past that hard
# cancel so a row is only swept once its job can no longer be running.
_RUNNING_THRESHOLD_SECS: int = EXTRACTION_JOB_TIMEOUT_SECS + ARQ_JOB_TIMEOUT_SECS


def _retry_at(attempts: int, now: datetime) -> str:
    retry_at = next_run_after(ErrorCode.WORKER_LOST, max(attempts, 1), now)
    return "infinity" if retry_at is None else retry_at.isoformat()


async def _sweep_running(admin_pool: asyncpg.Pool) -> int:
    """Fail every row running past the threshold as worker_lost; return rows swept.

    The stale rows are locked before they are read, so a row is never failed
    after another transaction moved it on. Each failure is counted once the
    transaction commits.
    """
    async with admin_pool.acquire() as conn, conn.transaction():
        stale = await conn.fetch(
            """
            SELECT id, attempts, clock_timestamp() AS db_now
            FROM extraction_jobs
            WHERE status = 'running'
              AND coalesce(started_at, created_at) < now() - ($1 * interval '1 second')
            FOR UPDATE SKIP LOCKED
            """,
            _RUNNING_THRESHOLD_SECS,
        )
        if not stale:
            return 0
        # No wallet charge is written here: the lost attempt's usage died with
        # its worker and is deliberately left uncharged. The global ceiling
        # and the provider's spend cap bound what such attempts can cost; do
        # not add a charge estimate or a refund to this path.
        await conn.executemany(
            """
            UPDATE extraction_jobs
            SET status = 'failed',
                error_code = $2,
                run_after = $3::text::timestamptz,
                completed_at = now()
            WHERE id = $1
            """,
            [
                (
                    row["id"],
                    ErrorCode.WORKER_LOST.value,
                    _retry_at(int(row["attempts"]), row["db_now"]),
                )
                for row in stale
            ],
        )
    for row in stale:
        outcome = failure_outcome(ErrorCode.WORKER_LOST, max(int(row["attempts"]), 1))
        record_job_failures(ErrorCode.WORKER_LOST, outcome)
    return len(stale)


async def _sweep_once(admin_pool: asyncpg.Pool) -> None:
    """Run one sweep and count its outcome; a database or socket error is logged and skipped."""
    log = logger.bind(component="orphan_cleanup")
    try:
        swept = await _sweep_running(admin_pool)
    except (asyncpg.PostgresError, OSError):
        await log.warning("orphan_cleanup_failed", exc_info=True)
        return
    if swept > 0:
        await log.info("orphan_cleanup_swept", swept=swept, state="running")
        _get_orphan_cleanup_swept_counter().add(
            swept, attributes={"result": "swept", "state": "running"}
        )
    else:
        _get_orphan_cleanup_swept_counter().add(
            1, attributes={"result": "none", "state": "running"}
        )


async def run_orphan_cleanup(admin_pool: asyncpg.Pool, *, sleep_seconds: int = 300) -> None:
    """Forever-loop cron: every ``sleep_seconds``, fail rows stuck in 'running'.

    Uses admin_pool (BYPASSRLS) so the sweep is cross-tenant.
    Runs until the task is cancelled (e.g. lifespan teardown).
    """
    await run_every(
        ORPHAN_CLEANUP_TASK_NAME,
        functools.partial(_sweep_once, admin_pool),
        interval_secs=sleep_seconds,
    )
