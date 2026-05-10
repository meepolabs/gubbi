"""Orphan-pending extraction_jobs cleanup cron.

Periodically marks stuck pending rows as failed with error_code='enqueue_lost'.
A row gets stuck if ingest INSERTs it but the subsequent Arq enqueue fails
(rare; Redis blip, broker outage). Without cleanup the partial unique index
holds the slot and prevents user retries.

Cadence: every {sleep_seconds} seconds. Threshold: {threshold_minutes} minutes
(config-tunable via llm.orphan_cleanup_threshold_minutes).
"""

from __future__ import annotations

import asyncio

import asyncpg
import structlog
from opentelemetry import metrics

__all__: list[str] = ["run_orphan_cleanup"]

logger = structlog.get_logger(__name__)

_meter = metrics.get_meter("gubbi")
ORPHAN_CLEANUP_SWEPT = _meter.create_counter(
    name="extraction_jobs.orphan_cleanup_swept_total",
    description="Count of orphan-pending extraction_jobs sweep cycles",
    unit="1",
)


async def run_orphan_cleanup(
    admin_pool: asyncpg.Pool,
    *,
    threshold_minutes: int = 30,
    sleep_seconds: int = 300,
) -> None:
    """Forever-loop cron: marks stale pending rows failed with error_code='enqueue_lost'.

    Uses admin_pool (BYPASSRLS) so the sweep is cross-tenant.
    Runs until the task is cancelled (e.g. lifespan teardown).
    """
    log = logger.bind(component="orphan_cleanup")
    while True:
        await asyncio.sleep(sleep_seconds)
        try:
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
            swept = int(result or 0)
            if swept > 0:
                await log.info("orphan_cleanup_swept", swept=swept)
                ORPHAN_CLEANUP_SWEPT.add(swept, attributes={"result": "swept"})
            else:
                ORPHAN_CLEANUP_SWEPT.add(1, attributes={"result": "none"})
        except (asyncpg.PostgresError, OSError):
            await log.warning("orphan_cleanup_failed", exc_info=True)
            continue
