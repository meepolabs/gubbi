"""Integration test: orphan-pending rows are cleaned up and unique slot is freed.

Seeds a pending extraction_jobs row with created_at past the threshold,
runs one cleanup cycle, then verifies:
  1. The row flips to status='failed' with error_code='enqueue_lost'.
  2. A second pending row for the same (user_id, conversation_id, source) can
     be inserted after cleanup (the partial unique index slot is freed).

Requires a running PostgreSQL instance with migrations applied through head.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import asyncpg
import pytest

from gubbi.extraction.orphan_cleanup import run_orphan_cleanup

pytestmark = [
    pytest.mark.asyncio(loop_scope="session"),
    pytest.mark.integration,
]

_USER_UUID = UUID("cccccccc-dddd-eeee-ffff-000000000001")


async def _seed_pending_row(
    conn: asyncpg.Connection,
    *,
    conversation_id: int,
    user_id: UUID,
    created_at: datetime,
) -> UUID:
    """Insert a pending extraction_jobs row with a specific created_at."""
    job_id = uuid4()
    await conn.execute(
        """
        INSERT INTO extraction_jobs
            (id, user_id, conversation_id, source, status, period_start, created_at, updated_at)
        VALUES ($1, $2, $3, 'test', 'pending', '2026-05-01'::date, $4, $4)
        """,
        job_id,
        user_id,
        conversation_id,
        created_at,
    )
    return job_id


@pytest.mark.skip(reason="Requires live DB with extraction_jobs table and partial unique index")
async def test_stale_pending_row_flipped_to_failed(admin_pool: asyncpg.Pool) -> None:
    """Orphan cleanup flips a stale pending row to failed with error_code='enqueue_lost'."""
    # Arrange: seed a pending row with created_at 35 minutes ago
    conversation_id = await admin_pool.fetchval("SELECT nextval('conversations_id_seq')")
    stale_time = datetime.now(UTC) - timedelta(minutes=35)

    async with admin_pool.acquire() as conn:
        job_id = await _seed_pending_row(
            conn,
            conversation_id=conversation_id,
            user_id=_USER_UUID,
            created_at=stale_time,
        )

    # Act: run one cleanup cycle with 30min threshold
    async def _single_cycle(seconds: float) -> None:
        raise asyncio.CancelledError

    with patch("gubbi.extraction.orphan_cleanup.asyncio.sleep", side_effect=_single_cycle):
        task = asyncio.create_task(
            run_orphan_cleanup(admin_pool, threshold_minutes=30, sleep_seconds=1)
        )
        with suppress(asyncio.CancelledError):
            await task

    # Assert: row is now failed with enqueue_lost
    row = await admin_pool.fetchrow(
        "SELECT status, error_code FROM extraction_jobs WHERE id = $1",
        job_id,
    )
    assert row is not None
    assert row["status"] == "failed"
    assert row["error_code"] == "enqueue_lost"


@pytest.mark.skip(reason="Requires live DB with extraction_jobs table and partial unique index")
async def test_unique_slot_freed_after_cleanup(admin_pool: asyncpg.Pool) -> None:
    """After cleanup, a second pending row for the same (user_id, conversation_id) can be inserted."""
    conversation_id = await admin_pool.fetchval("SELECT nextval('conversations_id_seq')")
    stale_time = datetime.now(UTC) - timedelta(minutes=35)

    async with admin_pool.acquire() as conn:
        await _seed_pending_row(
            conn,
            conversation_id=conversation_id,
            user_id=_USER_UUID,
            created_at=stale_time,
        )

    async def _single_cycle(seconds: float) -> None:
        raise asyncio.CancelledError

    with patch("gubbi.extraction.orphan_cleanup.asyncio.sleep", side_effect=_single_cycle):
        task = asyncio.create_task(
            run_orphan_cleanup(admin_pool, threshold_minutes=30, sleep_seconds=1)
        )
        with suppress(asyncio.CancelledError):
            await task

    # Now the slot should be free -- a new pending row must insert successfully
    async with admin_pool.acquire() as conn:
        new_job_id = await _seed_pending_row(
            conn,
            conversation_id=conversation_id,
            user_id=_USER_UUID,
            created_at=datetime.now(UTC),
        )

    row = await admin_pool.fetchrow(
        "SELECT status FROM extraction_jobs WHERE id = $1",
        new_job_id,
    )
    assert row is not None
    assert row["status"] == "pending"


# ---------------------------------------------------------------------------
# Stuck-running reaper
# ---------------------------------------------------------------------------


async def _seed_running_row(
    conn: asyncpg.Connection,
    *,
    conversation_id: int,
    user_id: UUID,
    started_at: datetime,
) -> UUID:
    """Insert a 'running' extraction_jobs row with a specific started_at."""
    job_id = uuid4()
    await conn.execute(
        """
        INSERT INTO extraction_jobs
            (id, user_id, conversation_id, source, status, period_start,
             created_at, updated_at, started_at)
        VALUES ($1, $2, $3, 'test', 'running', '2026-05-01'::date, $4, $4, $4)
        """,
        job_id,
        user_id,
        conversation_id,
        started_at,
    )
    return job_id


@pytest.mark.skip(reason="Requires live DB with extraction_jobs table")
async def test_stuck_running_row_flipped_to_worker_lost(admin_pool: asyncpg.Pool) -> None:
    """Orphan cleanup flips 'running' rows older than 2 * ARQ_JOB_TIMEOUT_SECS to 'failed'/worker_lost."""
    from unittest.mock import MagicMock

    conversation_id = await admin_pool.fetchval("SELECT nextval('conversations_id_seq')")
    # 25 minutes ago is well past the 20-minute reaper threshold.
    stuck_time = datetime.now(UTC) - timedelta(minutes=25)

    async with admin_pool.acquire() as conn:
        job_id = await _seed_running_row(
            conn,
            conversation_id=conversation_id,
            user_id=_USER_UUID,
            started_at=stuck_time,
        )

    # Stub a budget helper -- the reaper should call record_actual_cost once
    # per swept row with actual_cents=0.
    helper = MagicMock()
    helper.record_actual_cost = AsyncMock()

    async def _single_cycle(seconds: float) -> None:
        raise asyncio.CancelledError

    with patch("gubbi.extraction.orphan_cleanup.asyncio.sleep", side_effect=_single_cycle):
        task = asyncio.create_task(
            run_orphan_cleanup(
                admin_pool,
                threshold_minutes=30,
                sleep_seconds=1,
                budget_helper=helper,
            )
        )
        with suppress(asyncio.CancelledError):
            await task

    row = await admin_pool.fetchrow(
        "SELECT status, error_code FROM extraction_jobs WHERE id = $1",
        job_id,
    )
    assert row is not None
    assert row["status"] == "failed"
    assert row["error_code"] == "worker_lost"

    # Refund attempted with actual=0 against PRE_CHARGE_CENTS estimate.
    helper.record_actual_cost.assert_awaited_once()
    kwargs = helper.record_actual_cost.await_args.kwargs
    assert kwargs["actual_cents"] == 0


@pytest.mark.skip(reason="Requires live DB with extraction_jobs table")
async def test_recent_running_row_untouched(admin_pool: asyncpg.Pool) -> None:
    """Running rows newer than the 20-minute threshold remain unchanged."""
    conversation_id = await admin_pool.fetchval("SELECT nextval('conversations_id_seq')")
    recent_time = datetime.now(UTC) - timedelta(minutes=5)

    async with admin_pool.acquire() as conn:
        job_id = await _seed_running_row(
            conn,
            conversation_id=conversation_id,
            user_id=_USER_UUID,
            started_at=recent_time,
        )

    async def _single_cycle(seconds: float) -> None:
        raise asyncio.CancelledError

    with patch("gubbi.extraction.orphan_cleanup.asyncio.sleep", side_effect=_single_cycle):
        task = asyncio.create_task(
            run_orphan_cleanup(admin_pool, threshold_minutes=30, sleep_seconds=1)
        )
        with suppress(asyncio.CancelledError):
            await task

    row = await admin_pool.fetchrow(
        "SELECT status FROM extraction_jobs WHERE id = $1",
        job_id,
    )
    assert row is not None
    assert row["status"] == "running"
