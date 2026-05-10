"""Unit tests for gubbi.extraction.orphan_cleanup.run_orphan_cleanup."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from unittest.mock import AsyncMock, patch

import asyncpg
import pytest

from gubbi.extraction.orphan_cleanup import ORPHAN_CLEANUP_SWEPT, run_orphan_cleanup


async def _run_one_cycle(pool: asyncpg.Pool, threshold_minutes: int = 30) -> None:
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
            run_orphan_cleanup(pool, threshold_minutes=threshold_minutes, sleep_seconds=1)
        )
        # Wait until the loop body has completed and the second sleep was entered
        await pre_sleep2.wait()
        # Cancel while task is parked in the indefinite second sleep
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_stale_rows_updated() -> None:
    """run_orphan_cleanup updates rows older than the threshold."""
    pool = AsyncMock(spec=asyncpg.Pool)
    pool.fetchval = AsyncMock(return_value=3)

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
    pool = AsyncMock(spec=asyncpg.Pool)
    pool.fetchval = AsyncMock(return_value=0)  # nothing swept

    await _run_one_cycle(pool, threshold_minutes=30)

    # Threshold passed as parameter to the DB call
    call_args = pool.fetchval.call_args
    assert call_args is not None
    assert call_args[0][1] == "30"


@pytest.mark.asyncio
async def test_postgres_error_caught_loop_continues() -> None:
    """PostgresError must be caught and the loop must continue (not propagate)."""
    pool = AsyncMock(spec=asyncpg.Pool)
    # First call raises, second returns 0
    pool.fetchval = AsyncMock(side_effect=[asyncpg.PostgresError("db gone"), 0])

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
async def test_otel_counter_swept_attribute() -> None:
    """Counter increments with result='swept' when rows are flipped."""
    pool = AsyncMock(spec=asyncpg.Pool)
    pool.fetchval = AsyncMock(return_value=5)

    add_calls: list[tuple[int, dict[str, str]]] = []

    def _capture_add(amount: int, attributes: dict[str, str] | None = None) -> None:
        add_calls.append((amount, attributes or {}))

    with patch.object(ORPHAN_CLEANUP_SWEPT, "add", side_effect=_capture_add):
        await _run_one_cycle(pool)

    assert any(attrs.get("result") == "swept" for _, attrs in add_calls)


@pytest.mark.asyncio
async def test_otel_counter_none_attribute() -> None:
    """Counter increments with result='none' when no rows are swept."""
    pool = AsyncMock(spec=asyncpg.Pool)
    pool.fetchval = AsyncMock(return_value=0)

    add_calls: list[tuple[int, dict[str, str]]] = []

    def _capture_add(amount: int, attributes: dict[str, str] | None = None) -> None:
        add_calls.append((amount, attributes or {}))

    with patch.object(ORPHAN_CLEANUP_SWEPT, "add", side_effect=_capture_add):
        await _run_one_cycle(pool)

    assert any(attrs.get("result") == "none" for _, attrs in add_calls)
