"""Integration test: orphan cleanup sweeps stale rows and frees the unique slot.

Seeds extraction_jobs rows for a real tenant conversation, runs exactly one
cleanup cycle against the database, then verifies:
  1. A stale pending row flips to status='failed' with error_code='enqueue_lost'.
  2. The partial unique index holds the (user_id, conversation_id, source) slot
     while the stale row is pending, and frees it once the sweep has run.
  3. A stuck running row flips to 'failed'/worker_lost with a zero-cost refund.
  4. A recent running row is left untouched.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import asyncpg
import pytest

from gubbi.extraction import orphan_cleanup
from gubbi.extraction.orphan_cleanup import run_orphan_cleanup
from tests.fixtures.tenants import TenantSeed

pytestmark = [
    pytest.mark.asyncio(loop_scope="session"),
    pytest.mark.integration,
]

# ``patch`` on the module's ``asyncio.sleep`` replaces the attribute on the
# shared asyncio module, so the fake only intercepts calls carrying this
# value and delegates every other sleep to the real implementation.
_CYCLE_SLEEP_SECONDS = 7
_PENDING_THRESHOLD_MINUTES = 30
_real_sleep = asyncio.sleep


async def _run_one_cycle(admin_pool: asyncpg.Pool, *, budget_helper: Any = None) -> None:
    """Run exactly one sweep: the first loop sleep returns, the second parks until cancel."""
    body_done = asyncio.Event()
    cycle_sleeps = 0

    async def _fake_sleep(seconds: float, *args: Any, **kwargs: Any) -> Any:
        nonlocal cycle_sleeps
        if seconds != _CYCLE_SLEEP_SECONDS:
            return await _real_sleep(seconds, *args, **kwargs)
        cycle_sleeps += 1
        if cycle_sleeps > 1:
            body_done.set()
            await asyncio.Event().wait()
        return None

    with patch("gubbi.extraction.orphan_cleanup.asyncio.sleep", side_effect=_fake_sleep):
        task = asyncio.create_task(
            run_orphan_cleanup(
                admin_pool,
                threshold_minutes=_PENDING_THRESHOLD_MINUTES,
                sleep_seconds=_CYCLE_SLEEP_SECONDS,
                budget_helper=budget_helper,
            )
        )
        await asyncio.wait_for(body_done.wait(), timeout=10)
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


async def _seed_job(
    admin_pool: asyncpg.Pool,
    seed: TenantSeed,
    *,
    status: str,
    at: datetime,
) -> UUID:
    """Insert an extraction_jobs row for ``seed``'s conversation, timestamped ``at``."""
    assert seed.conversation_id is not None
    job_id = uuid4()
    await admin_pool.execute(
        """
        INSERT INTO extraction_jobs
            (id, user_id, conversation_id, source, status, period_start,
             created_at, started_at)
        VALUES ($1, $2, $3, 'test', $4, '2026-05-01'::date, $5, $6)
        """,
        job_id,
        seed.user_id,
        seed.conversation_id,
        status,
        at,
        at if status == "running" else None,
    )
    return job_id


async def _job_state(admin_pool: asyncpg.Pool, job_id: UUID) -> tuple[str, str | None]:
    row = await admin_pool.fetchrow(
        "SELECT status, error_code FROM extraction_jobs WHERE id = $1", job_id
    )
    assert row is not None
    return row["status"], row["error_code"]


async def test_stale_pending_row_flipped_to_failed(
    admin_pool: asyncpg.Pool, seeded_a: TenantSeed
) -> None:
    """A pending row older than the threshold becomes failed/enqueue_lost."""
    stale = datetime.now(UTC) - timedelta(minutes=_PENDING_THRESHOLD_MINUTES + 5)
    job_id = await _seed_job(admin_pool, seeded_a, status="pending", at=stale)

    await _run_one_cycle(admin_pool)

    assert await _job_state(admin_pool, job_id) == ("failed", "enqueue_lost")


async def test_unique_slot_freed_after_cleanup(
    admin_pool: asyncpg.Pool, seeded_a: TenantSeed
) -> None:
    """The stale pending row blocks a second pending row until the sweep fails it."""
    stale = datetime.now(UTC) - timedelta(minutes=_PENDING_THRESHOLD_MINUTES + 5)
    await _seed_job(admin_pool, seeded_a, status="pending", at=stale)
    with pytest.raises(asyncpg.UniqueViolationError):
        await _seed_job(admin_pool, seeded_a, status="pending", at=datetime.now(UTC))

    await _run_one_cycle(admin_pool)

    new_job_id = await _seed_job(admin_pool, seeded_a, status="pending", at=datetime.now(UTC))
    assert await _job_state(admin_pool, new_job_id) == ("pending", None)


async def test_stuck_running_row_flipped_to_worker_lost(
    admin_pool: asyncpg.Pool, seeded_a: TenantSeed
) -> None:
    """A running row past 2 * ARQ_JOB_TIMEOUT_SECS becomes failed/worker_lost and is refunded."""
    stuck = datetime.now(UTC) - timedelta(seconds=orphan_cleanup._RUNNING_THRESHOLD_SECS + 300)
    job_id = await _seed_job(admin_pool, seeded_a, status="running", at=stuck)
    helper = MagicMock()
    helper.record_actual_cost = AsyncMock()

    await _run_one_cycle(admin_pool, budget_helper=helper)

    assert await _job_state(admin_pool, job_id) == ("failed", "worker_lost")
    helper.record_actual_cost.assert_awaited_once()
    assert helper.record_actual_cost.await_args.kwargs["user_id"] == seeded_a.user_id
    assert helper.record_actual_cost.await_args.kwargs["actual_cents"] == 0


async def test_recent_running_row_untouched(admin_pool: asyncpg.Pool, seeded_a: TenantSeed) -> None:
    """A running row newer than the reaper threshold stays running."""
    recent = datetime.now(UTC) - timedelta(minutes=5)
    job_id = await _seed_job(admin_pool, seeded_a, status="running", at=recent)

    await _run_one_cycle(admin_pool)

    assert await _job_state(admin_pool, job_id) == ("running", None)
