"""Integration test: the orphan sweep fails stuck running rows for a retry.

Seeds extraction_jobs rows for a real tenant conversation, runs exactly one
cleanup cycle against the database, then verifies:
  1. A stuck running row flips to 'failed'/worker_lost with a future run_after,
     keeping its admission charge.
  2. A recent running row is left untouched.
  3. A pending row, however old, is queued work and is left untouched.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import patch
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

# ``patch`` on the loop helper's ``asyncio.sleep`` replaces the attribute on the
# shared asyncio module, so the fake only intercepts calls carrying this
# value and delegates every other sleep to the real implementation.
_CYCLE_SLEEP_SECONDS = 7
_real_sleep = asyncio.sleep


async def _run_one_cycle(admin_pool: asyncpg.Pool) -> None:
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

    with patch("gubbi.extraction.supervised_loop.asyncio.sleep", side_effect=_fake_sleep):
        task = asyncio.create_task(
            run_orphan_cleanup(admin_pool, sleep_seconds=_CYCLE_SLEEP_SECONDS)
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
    attempts: int = 1,
) -> UUID:
    """Insert a dispatched extraction_jobs row for ``seed``'s conversation, timestamped ``at``."""
    assert seed.conversation_id is not None
    job_id = uuid4()
    await admin_pool.execute(
        """
        INSERT INTO extraction_jobs
            (id, user_id, conversation_id, source, status, period_start,
             created_at, started_at, attempts, admitted_at)
        VALUES ($1, $2, $3, 'test', $4, '2026-05-01'::date, $5, $6, $7, $5)
        """,
        job_id,
        seed.user_id,
        seed.conversation_id,
        status,
        at,
        at if status == "running" else None,
        attempts,
    )
    return job_id


async def _job_state(admin_pool: asyncpg.Pool, job_id: UUID) -> tuple[str, str | None]:
    row = await admin_pool.fetchrow(
        "SELECT status, error_code FROM extraction_jobs WHERE id = $1", job_id
    )
    assert row is not None
    return row["status"], row["error_code"]


async def test_stuck_running_row_is_failed_as_worker_lost_with_a_future_run_after(
    admin_pool: asyncpg.Pool, seeded_a: TenantSeed
) -> None:
    stuck = datetime.now(UTC) - timedelta(seconds=orphan_cleanup._RUNNING_THRESHOLD_SECS + 300)
    job_id = await _seed_job(admin_pool, seeded_a, status="running", at=stuck)

    await _run_one_cycle(admin_pool)

    row = await admin_pool.fetchrow(
        "SELECT status, error_code, run_after > now() AS deferred,"
        " run_after < 'infinity' AS retried, admitted_at"
        " FROM extraction_jobs WHERE id = $1",
        job_id,
    )
    charges = await admin_pool.fetchval(
        "SELECT count(*) FROM llm_charges WHERE job_id = $1", job_id
    )
    assert row is not None
    assert (row["status"], row["error_code"]) == ("failed", "worker_lost")
    assert (row["deferred"], row["retried"]) == (True, True)
    assert row["admitted_at"] == stuck
    assert charges == 0


async def test_an_old_pending_row_is_left_queued(
    admin_pool: asyncpg.Pool, seeded_a: TenantSeed
) -> None:
    old = datetime.now(UTC) - timedelta(days=2)
    job_id = await _seed_job(admin_pool, seeded_a, status="pending", at=old, attempts=0)

    await _run_one_cycle(admin_pool)

    assert await _job_state(admin_pool, job_id) == ("pending", None)


async def test_recent_running_row_untouched(admin_pool: asyncpg.Pool, seeded_a: TenantSeed) -> None:
    """A running row newer than the reaper threshold stays running."""
    recent = datetime.now(UTC) - timedelta(minutes=5)
    job_id = await _seed_job(admin_pool, seeded_a, status="running", at=recent)

    await _run_one_cycle(admin_pool)

    assert await _job_state(admin_pool, job_id) == ("running", None)
