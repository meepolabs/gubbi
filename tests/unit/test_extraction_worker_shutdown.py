"""Worker shutdown waits for cancellation cleanups before closing its clients.

A cancelled job settles its budget and writes its failed state in a cleanup
task that can outlive the job. Shutdown must let those finish (or cancel them
after a bound) before the Redis client and the PostgreSQL pool they use close.
"""

from __future__ import annotations

import asyncio
from functools import partial
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gubbi.extraction import supervised_loop
from gubbi.extraction import worker as worker_module

pytestmark = pytest.mark.unit


def _ctx(events: list[str]) -> dict[str, Any]:
    def closer(name: str) -> AsyncMock:
        return AsyncMock(side_effect=lambda *_a, **_k: events.append(name))

    redis = MagicMock()
    redis.aclose = closer("redis_closed")
    redis_pool = MagicMock()
    redis_pool.aclose = closer("redis_pool_closed")
    pool = MagicMock()
    pool.close = closer("pg_pool_closed")
    return {"redis": redis, "redis_pool": redis_pool, "pool": pool, "cancel_cleanups": set()}


async def test_shutdown_waits_for_a_running_cleanup_before_closing_clients() -> None:
    events: list[str] = []
    ctx = _ctx(events)

    async def cleanup() -> None:
        await asyncio.sleep(0.05)
        events.append("cleanup_done")

    task = asyncio.create_task(cleanup())
    ctx["cancel_cleanups"].add(task)

    await worker_module.shutdown(ctx)  # type: ignore[arg-type]

    assert events == ["cleanup_done", "redis_closed", "redis_pool_closed", "pg_pool_closed"]


async def test_shutdown_cancels_a_stuck_cleanup_after_its_bound_then_closes() -> None:
    events: list[str] = []
    ctx = _ctx(events)

    async def stuck() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            events.append("cleanup_cancelled")
            raise

    task = asyncio.create_task(stuck())
    ctx["cancel_cleanups"].add(task)

    with patch.object(worker_module, "_CANCEL_CLEANUP_DRAIN_SECS", 0.05):
        await asyncio.wait_for(worker_module.shutdown(ctx), 2)  # type: ignore[arg-type]

    assert task.cancelled()
    assert events == ["cleanup_cancelled", "redis_closed", "redis_pool_closed", "pg_pool_closed"]


async def test_shutdown_leaves_a_cleanup_that_ignores_cancellation_and_closes() -> None:
    events: list[str] = []
    ctx = _ctx(events)
    release = asyncio.Event()

    async def stubborn() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            events.append("cancel_ignored")
            await release.wait()
            events.append("cleanup_released")

    task = asyncio.create_task(stubborn())
    ctx["cancel_cleanups"].add(task)

    try:
        with (
            patch.object(worker_module, "_CANCEL_CLEANUP_DRAIN_SECS", 0.05),
            patch.object(worker_module, "_CANCEL_CLEANUP_UNWIND_SECS", 0.05),
        ):
            await asyncio.wait_for(worker_module.shutdown(ctx), 2)  # type: ignore[arg-type]

        assert not task.done()
        assert events == ["cancel_ignored", "redis_closed", "redis_pool_closed", "pg_pool_closed"]
    finally:
        release.set()
        await asyncio.wait_for(task, 2)


async def test_shutdown_without_cleanups_closes_clients() -> None:
    events: list[str] = []

    await worker_module.shutdown(_ctx(events))  # type: ignore[arg-type]

    assert events == ["redis_closed", "redis_pool_closed", "pg_pool_closed"]


def _short_heartbeat_stop(timeout_secs: float, grace_secs: float) -> Any:
    return patch.object(
        worker_module,
        "stop_supervised_before_teardown",
        partial(
            supervised_loop.stop_supervised_before_teardown,
            timeout_secs=timeout_secs,
            grace_secs=grace_secs,
        ),
    )


async def test_shutdown_closes_clients_only_after_a_slow_heartbeat_unwinds() -> None:
    events: list[str] = []
    ctx = _ctx(events)

    async def slow_heartbeat() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await asyncio.sleep(0.1)
            events.append("heartbeat_unwound")

    ctx["heartbeat_task"] = asyncio.create_task(slow_heartbeat())
    await asyncio.sleep(0)

    with _short_heartbeat_stop(timeout_secs=0.02, grace_secs=2):
        await asyncio.wait_for(worker_module.shutdown(ctx), 5)  # type: ignore[arg-type]

    assert events == ["heartbeat_unwound", "redis_closed", "redis_pool_closed", "pg_pool_closed"]


async def test_shutdown_closes_clients_under_a_live_heartbeat_after_both_waits() -> None:
    events: list[str] = []
    ctx = _ctx(events)
    release = asyncio.Event()

    async def stubborn_heartbeat() -> None:
        while not release.is_set():
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                events.append("cancel_ignored")

    async def log_error(event: str, **_kw: Any) -> None:
        events.append(event)

    task = asyncio.create_task(stubborn_heartbeat())
    ctx["heartbeat_task"] = task
    await asyncio.sleep(0)

    try:
        with (
            _short_heartbeat_stop(timeout_secs=0.02, grace_secs=0.02),
            patch.object(supervised_loop.logger, "error", AsyncMock(side_effect=log_error)),
            patch.object(supervised_loop, "record_background_task_died"),
        ):
            await asyncio.wait_for(worker_module.shutdown(ctx), 2)  # type: ignore[arg-type]

        assert not task.done()
        assert events == [
            "cancel_ignored",
            "background_task_stop_timeout",
            "background_task_alive_at_teardown",
            "redis_closed",
            "redis_pool_closed",
            "pg_pool_closed",
        ]
    finally:
        release.set()
        task.cancel()
        await asyncio.wait([task], timeout=1)
