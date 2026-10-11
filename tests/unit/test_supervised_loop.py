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
    stop_supervised_before_teardown,
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


def _ignores_cancellation(release: asyncio.Event) -> asyncio.Task[None]:
    async def _stubborn() -> None:
        while not release.is_set():
            with suppress(asyncio.CancelledError):
                await asyncio.sleep(10)

    return asyncio.create_task(_stubborn(), name="stubborn")


async def test_stop_reports_logs_and_counts_a_loop_that_ignores_cancellation() -> None:
    release = asyncio.Event()
    task = _ignores_cancellation(release)
    await asyncio.sleep(0)

    try:
        with (
            patch.object(supervised_loop.logger, "error", AsyncMock()) as error,
            patch.object(supervised_loop, "record_background_task_died") as died,
        ):
            stuck = await asyncio.wait_for(stop_supervised([task], timeout_secs=0.05), 2)

        assert stuck == {task}
        assert not task.done()
        error.assert_awaited_once_with("background_task_stop_timeout", task="stubborn")
        died.assert_called_once_with(task="stubborn", stage="stop_timeout")
    finally:
        release.set()
        task.cancel()
        await asyncio.wait([task], timeout=1)


async def test_stop_reports_nothing_when_every_loop_unwinds() -> None:
    task = start_supervised("demo", _park())
    await asyncio.sleep(0)

    with patch.object(supervised_loop, "record_background_task_died") as died:
        stuck = await stop_supervised([task], timeout_secs=1)

    assert stuck == set()
    died.assert_not_called()


async def test_teardown_stop_waits_out_its_grace_for_a_slow_unwinding_loop() -> None:
    events: list[str] = []

    async def _slow_unwind() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await asyncio.sleep(0.1)
            events.append("unwound")

    task = asyncio.create_task(_slow_unwind(), name="slow")
    await asyncio.sleep(0)

    with patch.object(supervised_loop.logger, "error", AsyncMock()) as error:
        await stop_supervised_before_teardown([task], timeout_secs=0.02, grace_secs=2)
    events.append("returned")

    assert events == ["unwound", "returned"]
    error.assert_awaited_once_with("background_task_stop_timeout", task="slow")


async def test_teardown_stop_logs_a_loop_alive_after_its_grace_and_returns() -> None:
    release = asyncio.Event()
    task = _ignores_cancellation(release)
    await asyncio.sleep(0)

    try:
        with patch.object(supervised_loop.logger, "error", AsyncMock()) as error:
            await asyncio.wait_for(
                stop_supervised_before_teardown([task], timeout_secs=0.02, grace_secs=0.02), 2
            )

        assert not task.done()
        assert [c.args[0] for c in error.await_args_list] == [
            "background_task_stop_timeout",
            "background_task_alive_at_teardown",
        ]
    finally:
        release.set()
        task.cancel()
        await asyncio.wait([task], timeout=1)
