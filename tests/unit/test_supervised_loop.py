"""Unit tests for the supervised background-loop helper."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from unittest.mock import AsyncMock, patch

import pytest

from gubbi.extraction import supervised_loop
from gubbi.extraction.supervised_loop import (
    run_every,
    start_supervised,
    stop_supervised,
)

pytestmark = pytest.mark.unit


async def test_a_raising_tick_is_counted_and_the_loop_keeps_ticking() -> None:
    ticks = 0
    third_tick = asyncio.Event()

    async def _tick() -> None:
        nonlocal ticks
        ticks += 1
        if ticks == 3:
            third_tick.set()
        if ticks == 1:
            raise RuntimeError("tick crashed")

    with patch.object(supervised_loop, "record_background_task_died") as died:
        task = asyncio.create_task(run_every("demo", _tick, interval_secs=0))
        await asyncio.wait_for(third_tick.wait(), 5)
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    died.assert_called_once_with(task="demo", stage="tick")


@pytest.mark.parametrize(
    "outcome",
    [
        pytest.param(RuntimeError("loop crashed"), id="raises"),
        pytest.param(None, id="returns"),
    ],
)
async def test_a_loop_that_ends_on_its_own_is_counted_once(outcome: Exception | None) -> None:
    async def _loop() -> None:
        if outcome is not None:
            raise outcome

    with patch.object(supervised_loop, "record_background_task_died") as died:
        await start_supervised("demo", _loop())

    died.assert_called_once_with(task="demo", stage="loop")


async def _park() -> None:
    await asyncio.Event().wait()


async def test_cancelling_a_supervised_loop_is_not_counted() -> None:
    with patch.object(supervised_loop, "record_background_task_died") as died:
        task = start_supervised("demo", _park())
        await asyncio.sleep(0)
        await stop_supervised([task])

    assert task.cancelled()
    died.assert_not_called()


async def test_stop_returns_by_its_deadline_when_a_loop_ignores_cancellation() -> None:
    release = asyncio.Event()

    async def _stubborn() -> None:
        while not release.is_set():
            with suppress(asyncio.CancelledError):
                await asyncio.sleep(10)

    task = asyncio.create_task(_stubborn())
    await asyncio.sleep(0)
    with patch.object(supervised_loop.logger, "warning", AsyncMock()) as warning:
        await asyncio.wait_for(stop_supervised([task], timeout_secs=0.05), 2)

    assert not task.done()
    warning.assert_awaited_once()
    release.set()
    task.cancel()
    await asyncio.wait([task], timeout=1)
