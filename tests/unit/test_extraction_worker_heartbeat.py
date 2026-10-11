"""The extraction worker's heartbeat counter: one beat per interval, stopped by shutdown."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

import pytest
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from gubbi.config import get_settings
from gubbi.extraction import heartbeat, supervised_loop
from gubbi.extraction import worker as worker_module
from gubbi.extraction.heartbeat import WORKER_HEARTBEAT_TASK_NAME, run_worker_heartbeat
from gubbi.extraction.supervised_loop import start_supervised, stop_supervised
from gubbi.telemetry.attrs import MetricNames
from tests.extraction.test_worker_startup_pg_probe import (
    _patch_worker_dependencies,
    _stub_runner_with_capture,
    _stub_telemetry,
)
from tests.fixtures.metrics import collect_gauges, metric_reader  # noqa: F401

pytestmark = pytest.mark.unit

_INTERVAL_SECS = 0.01
_real_sleep = asyncio.sleep


def _beats(reader: InMemoryMetricReader) -> float:
    return collect_gauges(reader).get(MetricNames.EXTRACTION_WORKER_HEARTBEAT, {}).get((), 0.0)


async def _wait_for_beats(reader: InMemoryMetricReader, count: int) -> None:
    while _beats(reader) < count:
        await _real_sleep(_INTERVAL_SECS)


async def test_the_heartbeat_beats_at_once_then_once_per_interval(
    metric_reader: InMemoryMetricReader,  # noqa: F811
) -> None:
    sleeps: list[float] = []
    parked = asyncio.Event()

    async def _two_intervals_then_park(seconds: float, *args: Any, **kwargs: Any) -> Any:
        if seconds != _INTERVAL_SECS:
            return await _real_sleep(seconds, *args, **kwargs)
        sleeps.append(seconds)
        if len(sleeps) > 2:
            parked.set()
            await asyncio.Event().wait()
        return None

    with patch.object(supervised_loop.asyncio, "sleep", _two_intervals_then_park):
        task = start_supervised(
            WORKER_HEARTBEAT_TASK_NAME, run_worker_heartbeat(interval_secs=_INTERVAL_SECS)
        )
        await asyncio.wait_for(parked.wait(), 5)
        await stop_supervised([task])

    assert _beats(metric_reader) == 3
    assert task.done()


async def test_worker_startup_starts_the_heartbeat_and_shutdown_stops_it(
    monkeypatch: pytest.MonkeyPatch,
    metric_reader: InMemoryMetricReader,  # noqa: F811
) -> None:
    monkeypatch.setenv("JOURNAL_LLM_PROVIDER", "fake")
    monkeypatch.setenv("JOURNAL_EXTRACTION_WORKER_HEARTBEAT_SECS", str(_INTERVAL_SECS))
    _patch_worker_dependencies(monkeypatch)
    _stub_telemetry(monkeypatch)
    _stub_runner_with_capture(monkeypatch)
    monkeypatch.setattr(worker_module, "run_worker_heartbeat", heartbeat.run_worker_heartbeat)
    monkeypatch.setattr(worker_module, "start_supervised", supervised_loop.start_supervised)
    get_settings.cache_clear()
    ctx: dict[str, Any] = {}

    try:
        await worker_module.startup(ctx)  # type: ignore[arg-type]
        await asyncio.wait_for(_wait_for_beats(metric_reader, 3), 5)
        await asyncio.wait_for(worker_module.shutdown(ctx), 5)  # type: ignore[arg-type]
        stopped_at = _beats(metric_reader)
        await _real_sleep(_INTERVAL_SECS * 5)
    finally:
        task = ctx.get("heartbeat_task")
        if task is not None and not task.done():
            await stop_supervised([task])
        get_settings.cache_clear()

    assert ctx["heartbeat_task"].done()
    assert _beats(metric_reader) == stopped_at
