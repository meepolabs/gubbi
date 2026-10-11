"""Unit tests for the extraction scheduler loop that need no database."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from unittest.mock import MagicMock, patch

import pytest

from gubbi.config import get_settings
from gubbi.extraction import supervised_loop
from gubbi.extraction.scheduler import (
    ExtractionScheduler,
    SchedulerConfig,
    run_extraction_scheduler,
)
from gubbi.extraction.worker import WorkerSettings

pytestmark = pytest.mark.unit


async def test_a_crashing_tick_increments_the_task_died_counter() -> None:
    ticks = 0
    third_tick = asyncio.Event()

    async def _crash(_self: ExtractionScheduler) -> None:
        nonlocal ticks
        ticks += 1
        if ticks == 3:
            third_tick.set()
        raise RuntimeError("tick crashed")

    config = SchedulerConfig(
        paused=False, budget_enabled=False, dispatch_bound=1, interval_secs=0.001
    )
    with (
        patch.object(ExtractionScheduler, "tick", _crash),
        patch.object(supervised_loop, "record_background_task_died") as died,
    ):
        task = asyncio.create_task(run_extraction_scheduler(MagicMock(), MagicMock(), config))
        await asyncio.wait_for(third_tick.wait(), 5)
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    assert died.call_count >= 2
    assert {(c.kwargs["task"], c.kwargs["stage"]) for c in died.call_args_list} == {
        ("extraction_scheduler", "tick")
    }


def test_scheduler_settings_default_to_running_every_30s_with_twice_worker_concurrency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "JOURNAL_EXTRACTION_SCHEDULER_PAUSED",
        "JOURNAL_EXTRACTION_SCHEDULER_INTERVAL_SECS",
        "JOURNAL_EXTRACTION_DISPATCH_BOUND",
    ):
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()

    config = SchedulerConfig.from_settings(get_settings())

    assert (config.paused, config.interval_secs, config.dispatch_bound) == (
        False,
        30,
        2 * WorkerSettings.max_jobs,
    )
