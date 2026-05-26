"""Orphan-pending extraction_jobs cleanup cron.

Periodically marks stuck rows as failed:

  pending sweep  -- error_code='enqueue_lost'
    A row gets stuck if ingest INSERTs it but the subsequent Arq enqueue
    fails (rare; Redis blip, broker outage). Without cleanup the partial
    unique index holds the slot and prevents user retries.

  running sweep  -- error_code='worker_lost'
    A row stays in 'running' indefinitely if the Arq worker dies between
    mark_running and the SAVEPOINT commit (OOM, container kill, hard crash).
    The threshold is 2 * ARQ_JOB_TIMEOUT_SECS (20 minutes), well past any
    legitimate completion. Each swept row also triggers a best-effort budget
    refund so the user does not lose budget the worker never spent.

Cadence: every {sleep_seconds} seconds. Threshold: {threshold_minutes} minutes
(config-tunable via llm.orphan_cleanup_threshold_minutes for the pending
sweep; the running-sweep threshold is fixed at 2 * ARQ_JOB_TIMEOUT_SECS).
"""

from __future__ import annotations

import asyncio
import functools
from typing import Any

import asyncpg
import structlog
from gubbi_common.budget import PRE_CHARGE_CENTS
from opentelemetry.metrics import Counter, get_meter

from gubbi.constants import ARQ_JOB_TIMEOUT_SECS

__all__: list[str] = ["run_orphan_cleanup"]

logger = structlog.get_logger(__name__)


@functools.lru_cache(maxsize=1)
def _get_orphan_cleanup_swept_counter() -> Counter:
    """Lazily create the orphan-cleanup swept counter against the live meter.

    The previous module-scope ``_meter.create_counter``
    bound at import time, well before ``configure_otel`` ran during the
    FastAPI lifespan -- so the counter held a NoOp instrument and silently
    discarded every ``.add(...)``. Deferring creation to first call (and
    re-priming via ``rebind_metrics_after_configure``) ensures the counter
    binds to the SDK provider configured at lifespan time. Mirrors the
    canonical pattern in ``gubbi.telemetry.metrics.initialize_metrics``.
    """
    return get_meter("gubbi").create_counter(
        name="extraction_jobs.orphan_cleanup_swept_total",
        description="Count of orphan extraction_jobs sweep cycles, partitioned by state",
        unit="1",
    )


# Stuck-running threshold: any row still 'running' beyond this is presumed
# worker-lost (worker died between mark_running and SAVEPOINT commit).
# 2x the Arq job timeout leaves headroom for legitimate slow runs.
_RUNNING_THRESHOLD_SECS: int = 2 * ARQ_JOB_TIMEOUT_SECS


async def _refund_swept_row(
    helper: Any,
    user_id: Any,
    period_start: Any,
    log: structlog.stdlib.AsyncBoundLogger,
) -> None:
    """Best-effort pre-charge refund for a sweep-flipped row.

    Failure (Redis blip, helper missing key) is logged and swallowed -- the
    row update is the durable record of the worker_lost outcome; refund is a
    convenience layer.
    """
    try:
        await helper.record_actual_cost(
            user_id=user_id,
            period_start=period_start,
            actual_cents=0,
            estimated_cents=PRE_CHARGE_CENTS,
        )
    except Exception:  # broad: redis errors come in many shapes
        await log.warning(
            "orphan_cleanup_refund_failed",
            user_id=str(user_id),
            period_start=str(period_start),
            exc_info=True,
        )


async def run_orphan_cleanup(
    admin_pool: asyncpg.Pool,
    *,
    threshold_minutes: int = 30,
    sleep_seconds: int = 300,
    budget_helper: Any = None,
) -> None:
    """Forever-loop cron: marks stale pending rows failed, then stale running rows failed.

    pending  -> error_code='enqueue_lost' (threshold_minutes)
    running  -> error_code='worker_lost'  (2 * ARQ_JOB_TIMEOUT_SECS, refund per row)

    Uses admin_pool (BYPASSRLS) so the sweep is cross-tenant.
    Runs until the task is cancelled (e.g. lifespan teardown).
    """
    log = logger.bind(component="orphan_cleanup")
    while True:
        await asyncio.sleep(sleep_seconds)
        try:
            # ---- pending sweep (enqueue_lost) -------------------------------
            result = await admin_pool.fetchval(
                """
                WITH updated AS (
                    UPDATE extraction_jobs
                    SET status = 'failed',
                        error_code = 'enqueue_lost',
                        completed_at = now()
                    WHERE status = 'pending'
                      AND created_at < now() - ($1 * interval '1 minute')
                    RETURNING id
                )
                SELECT count(*) FROM updated
                """,
                threshold_minutes,
            )
            swept_pending = int(result or 0)
            if swept_pending > 0:
                await log.info("orphan_cleanup_swept", swept=swept_pending, state="pending")
                _get_orphan_cleanup_swept_counter().add(
                    swept_pending, attributes={"result": "swept", "state": "pending"}
                )
            else:
                _get_orphan_cleanup_swept_counter().add(
                    1, attributes={"result": "none", "state": "pending"}
                )

            # ---- running sweep (worker_lost) --------------------------------
            running_rows = await admin_pool.fetch(
                """
                UPDATE extraction_jobs
                SET status = 'failed',
                    error_code = 'worker_lost',
                    completed_at = now()
                WHERE status = 'running'
                  AND started_at < now() - ($1 * interval '1 second')
                RETURNING id, user_id, period_start
                """,
                _RUNNING_THRESHOLD_SECS,
            )
            swept_running = len(running_rows)
            if swept_running > 0:
                await log.info("orphan_cleanup_swept", swept=swept_running, state="running")
                _get_orphan_cleanup_swept_counter().add(
                    swept_running, attributes={"result": "swept", "state": "running"}
                )
                # Best-effort refund per swept row -- logged + swallowed on failure.
                if budget_helper is not None:
                    for row in running_rows:
                        await _refund_swept_row(
                            budget_helper,
                            row["user_id"],
                            row["period_start"],
                            log,
                        )
            else:
                _get_orphan_cleanup_swept_counter().add(
                    1, attributes={"result": "none", "state": "running"}
                )
        except (asyncpg.PostgresError, OSError):
            await log.warning("orphan_cleanup_failed", exc_info=True)
            continue
