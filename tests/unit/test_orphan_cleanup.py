"""Unit tests for gubbi.extraction.orphan_cleanup.run_orphan_cleanup."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from unittest.mock import AsyncMock, MagicMock, patch

import asyncpg
import pytest

from gubbi.extraction.orphan_cleanup import (
    _RUNNING_THRESHOLD_SECS,
    ORPHAN_CLEANUP_SWEPT,
    run_orphan_cleanup,
)


async def _run_one_cycle(
    pool: asyncpg.Pool,
    threshold_minutes: int = 30,
    *,
    budget_helper: object | None = None,
) -> None:
    """Run the cleanup loop for exactly one iteration then cancel."""
    pre_body = asyncio.Event()  # set just before body executes (first sleep done)
    pre_sleep2 = asyncio.Event()  # set when second sleep is entered

    call_count = 0

    async def _mock_sleep(seconds: float) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First sleep: signal test that pre-body pause happened, return immediately
            pre_body.set()
        else:
            # Second+ sleep: signal that body completed, then block until cancelled
            pre_sleep2.set()
            # Park here until task.cancel() fires
            await asyncio.Event().wait()

    with patch("gubbi.extraction.orphan_cleanup.asyncio.sleep", side_effect=_mock_sleep):
        task = asyncio.create_task(
            run_orphan_cleanup(
                pool,
                threshold_minutes=threshold_minutes,
                sleep_seconds=1,
                budget_helper=budget_helper,
            )
        )
        # Wait until the loop body has completed and the second sleep was entered
        await pre_sleep2.wait()
        # Cancel while task is parked in the indefinite second sleep
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


def _empty_pool(swept_count: int = 0) -> AsyncMock:
    """Build a pool whose pending sweep returns ``swept_count`` and running sweep returns []."""
    pool = AsyncMock(spec=asyncpg.Pool)
    pool.fetchval = AsyncMock(return_value=swept_count)
    pool.fetch = AsyncMock(return_value=[])
    return pool


@pytest.mark.asyncio
async def test_stale_rows_updated() -> None:
    """run_orphan_cleanup updates pending rows older than the threshold."""
    pool = _empty_pool(swept_count=3)

    await _run_one_cycle(pool, threshold_minutes=30)

    assert pool.fetchval.called
    # Verify the SQL contains the right clauses
    sql_arg = pool.fetchval.call_args[0][0]
    assert "enqueue_lost" in sql_arg
    assert "pending" in sql_arg
    assert "failed" in sql_arg


@pytest.mark.asyncio
async def test_young_rows_untouched() -> None:
    """Rows newer than threshold must not be affected (checked via threshold parameter pass-through)."""
    pool = _empty_pool(swept_count=0)

    await _run_one_cycle(pool, threshold_minutes=30)

    # Threshold passed as parameter to the DB call
    call_args = pool.fetchval.call_args
    assert call_args is not None
    assert call_args[0][1] == 30


@pytest.mark.asyncio
async def test_postgres_error_caught_loop_continues() -> None:
    """PostgresError must be caught and the loop must continue (not propagate)."""
    pool = AsyncMock(spec=asyncpg.Pool)
    # First call raises, second returns 0
    pool.fetchval = AsyncMock(side_effect=[asyncpg.PostgresError("db gone"), 0])
    pool.fetch = AsyncMock(return_value=[])

    call_count = 0

    async def _sleep_then_count(seconds: float) -> None:
        nonlocal call_count
        call_count += 1
        if call_count >= 3:
            raise asyncio.CancelledError
        # Return immediately

    with patch("gubbi.extraction.orphan_cleanup.asyncio.sleep", side_effect=_sleep_then_count):
        task = asyncio.create_task(run_orphan_cleanup(pool, threshold_minutes=30, sleep_seconds=1))
        with suppress(asyncio.CancelledError):
            await task

    # Should have called fetchval twice (once raised, once succeeded)
    assert pool.fetchval.call_count == 2


@pytest.mark.asyncio
async def test_os_error_caught_loop_continues() -> None:
    """OSError must be caught, logged, and the loop must continue."""
    pool = AsyncMock(spec=asyncpg.Pool)
    pool.fetchval = AsyncMock(side_effect=[OSError("socket reset"), 0])
    pool.fetch = AsyncMock(return_value=[])
    mock_log = AsyncMock()
    call_count = 0

    async def _sleep_then_count(seconds: float) -> None:
        nonlocal call_count
        call_count += 1
        if call_count >= 3:
            raise asyncio.CancelledError

    with (
        patch("gubbi.extraction.orphan_cleanup.asyncio.sleep", side_effect=_sleep_then_count),
        patch("gubbi.extraction.orphan_cleanup.logger.bind", return_value=mock_log),
    ):
        task = asyncio.create_task(run_orphan_cleanup(pool, threshold_minutes=30, sleep_seconds=1))
        with suppress(asyncio.CancelledError):
            await task

    assert pool.fetchval.call_count == 2
    mock_log.warning.assert_awaited_once_with("orphan_cleanup_failed", exc_info=True)


@pytest.mark.asyncio
async def test_otel_counter_swept_attribute() -> None:
    """Counter increments with result='swept' state='pending' when pending rows are flipped."""
    pool = _empty_pool(swept_count=5)

    add_calls: list[tuple[int, dict[str, str]]] = []

    def _capture_add(amount: int, attributes: dict[str, str] | None = None) -> None:
        add_calls.append((amount, attributes or {}))

    with patch.object(ORPHAN_CLEANUP_SWEPT, "add", side_effect=_capture_add):
        await _run_one_cycle(pool)

    assert any(
        attrs.get("result") == "swept" and attrs.get("state") == "pending" for _, attrs in add_calls
    )


@pytest.mark.asyncio
async def test_otel_counter_none_attribute() -> None:
    """Counter increments with result='none' for the pending lane when nothing swept."""
    pool = _empty_pool(swept_count=0)

    add_calls: list[tuple[int, dict[str, str]]] = []

    def _capture_add(amount: int, attributes: dict[str, str] | None = None) -> None:
        add_calls.append((amount, attributes or {}))

    with patch.object(ORPHAN_CLEANUP_SWEPT, "add", side_effect=_capture_add):
        await _run_one_cycle(pool)

    assert any(
        attrs.get("result") == "none" and attrs.get("state") == "pending" for _, attrs in add_calls
    )


# ---------------------------------------------------------------------------
# Stuck-running reaper (B2 / Q3)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_running_sweep_uses_2x_arq_timeout_threshold() -> None:
    """Running sweep threshold == 2 * ARQ_JOB_TIMEOUT_SECS (1200 seconds)."""
    from gubbi.constants import ARQ_JOB_TIMEOUT_SECS

    assert _RUNNING_THRESHOLD_SECS == 2 * ARQ_JOB_TIMEOUT_SECS == 1200


@pytest.mark.asyncio
async def test_running_sweep_calls_correct_query() -> None:
    """Running-sweep UPDATE matches the locked B2 contract (status, error_code, threshold)."""
    pool = _empty_pool(swept_count=0)

    await _run_one_cycle(pool)

    pool.fetch.assert_awaited()
    sql, threshold = pool.fetch.call_args[0]
    assert "worker_lost" in sql
    assert "running" in sql
    assert "started_at" in sql
    assert "RETURNING id, user_id, period_start" in sql
    assert threshold == _RUNNING_THRESHOLD_SECS


@pytest.mark.asyncio
async def test_running_sweep_emits_state_running_counter() -> None:
    """Counter has state=running for the second lane and result=swept when rows flip."""
    pool = AsyncMock(spec=asyncpg.Pool)
    pool.fetchval = AsyncMock(return_value=0)  # no pending sweep
    # Two stuck running rows
    pool.fetch = AsyncMock(
        return_value=[
            {"id": "row-1", "user_id": "u1", "period_start": "2026-05-01"},
            {"id": "row-2", "user_id": "u2", "period_start": "2026-05-01"},
        ]
    )

    add_calls: list[tuple[int, dict[str, str]]] = []

    def _capture_add(amount: int, attributes: dict[str, str] | None = None) -> None:
        add_calls.append((amount, attributes or {}))

    with patch.object(ORPHAN_CLEANUP_SWEPT, "add", side_effect=_capture_add):
        await _run_one_cycle(pool)

    swept_running = [
        a for a in add_calls if a[1].get("state") == "running" and a[1].get("result") == "swept"
    ]
    assert len(swept_running) == 1
    assert swept_running[0][0] == 2


@pytest.mark.asyncio
async def test_running_sweep_refunds_each_row() -> None:
    """For every running row swept, record_actual_cost is called with actual=0 and PRE_CHARGE_CENTS."""
    from gubbi_common.budget import PRE_CHARGE_CENTS

    pool = AsyncMock(spec=asyncpg.Pool)
    pool.fetchval = AsyncMock(return_value=0)
    pool.fetch = AsyncMock(
        return_value=[
            {"id": "r1", "user_id": "u1", "period_start": "2026-05-01"},
            {"id": "r2", "user_id": "u2", "period_start": "2026-05-01"},
        ]
    )

    helper = MagicMock()
    helper.record_actual_cost = AsyncMock()

    await _run_one_cycle(pool, budget_helper=helper)

    assert helper.record_actual_cost.await_count == 2
    for call in helper.record_actual_cost.await_args_list:
        kwargs = call.kwargs
        assert kwargs["actual_cents"] == 0
        assert kwargs["estimated_cents"] == PRE_CHARGE_CENTS


@pytest.mark.asyncio
async def test_running_sweep_refund_failure_logged_and_swallowed() -> None:
    """A Redis failure in the refund path must not abort the rest of the sweep loop."""
    pool = AsyncMock(spec=asyncpg.Pool)
    pool.fetchval = AsyncMock(return_value=0)
    pool.fetch = AsyncMock(
        return_value=[
            {"id": "r1", "user_id": "u1", "period_start": "2026-05-01"},
            {"id": "r2", "user_id": "u2", "period_start": "2026-05-01"},
        ]
    )

    helper = MagicMock()
    helper.record_actual_cost = AsyncMock(side_effect=ConnectionError("redis dead"))

    # Must NOT raise -- the loop continues.
    await _run_one_cycle(pool, budget_helper=helper)

    # Both rows attempted (refund failure on row1 didn't stop row2).
    assert helper.record_actual_cost.await_count == 2


@pytest.mark.asyncio
async def test_no_refund_when_helper_missing() -> None:
    """When budget_helper=None (self-host) the sweep still flips rows but does not refund."""
    pool = AsyncMock(spec=asyncpg.Pool)
    pool.fetchval = AsyncMock(return_value=0)
    pool.fetch = AsyncMock(
        return_value=[{"id": "r1", "user_id": "u1", "period_start": "2026-05-01"}]
    )

    # No exception expected.
    await _run_one_cycle(pool, budget_helper=None)
    pool.fetch.assert_awaited()
