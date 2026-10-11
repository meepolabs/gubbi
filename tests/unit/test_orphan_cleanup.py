"""Unit tests for the run_orphan_cleanup loop; the sweep's SQL is covered by integration tests."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import asyncpg
import pytest

from gubbi.constants import ARQ_JOB_TIMEOUT_SECS, EXTRACTION_JOB_TIMEOUT_SECS
from gubbi.extraction.orphan_cleanup import (
    _RUNNING_THRESHOLD_SECS,
    _get_orphan_cleanup_swept_counter,
    run_orphan_cleanup,
)

pytestmark = pytest.mark.unit


async def _run_cycles(sweep: AsyncMock, cycles: int) -> None:
    """Run the loop for ``cycles`` sweeps, then cancel it while it sleeps."""
    sleeps = 0
    parked = asyncio.Event()

    async def _sleep(_seconds: float) -> None:
        nonlocal sleeps
        sleeps += 1
        if sleeps > cycles:
            parked.set()
            await asyncio.Event().wait()

    with (
        patch("gubbi.extraction.supervised_loop.asyncio.sleep", side_effect=_sleep),
        patch("gubbi.extraction.orphan_cleanup._sweep_running", sweep),
    ):
        task = asyncio.create_task(run_orphan_cleanup(MagicMock(), sleep_seconds=1))
        await asyncio.wait_for(parked.wait(), 5)
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


def test_running_threshold_derives_from_the_extraction_timeout() -> None:
    assert _RUNNING_THRESHOLD_SECS == EXTRACTION_JOB_TIMEOUT_SECS + ARQ_JOB_TIMEOUT_SECS == 2400


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(asyncpg.PostgresError("db gone"), id="postgres"),
        pytest.param(OSError("socket reset"), id="os"),
    ],
)
async def test_a_failed_sweep_is_logged_and_the_loop_continues(error: Exception) -> None:
    sweep = AsyncMock(side_effect=[error, 0])
    log = AsyncMock()

    with patch("gubbi.extraction.orphan_cleanup.logger.bind", return_value=log):
        await _run_cycles(sweep, cycles=2)

    assert sweep.await_count == 2
    log.warning.assert_awaited_once_with("orphan_cleanup_failed", exc_info=True)


@pytest.mark.parametrize(
    ("swept", "expected"),
    [
        pytest.param(2, (2, {"result": "swept", "state": "running"}), id="swept"),
        pytest.param(0, (1, {"result": "none", "state": "running"}), id="none"),
    ],
)
async def test_each_cycle_counts_its_outcome(swept: int, expected: tuple[int, Any]) -> None:
    calls: list[tuple[int, dict[str, str]]] = []

    def _add(amount: int, attributes: dict[str, str] | None = None) -> None:
        calls.append((amount, attributes or {}))

    with patch.object(_get_orphan_cleanup_swept_counter(), "add", side_effect=_add):
        await _run_cycles(AsyncMock(return_value=swept), cycles=1)

    assert calls == [expected]
