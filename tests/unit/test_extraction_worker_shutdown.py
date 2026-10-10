"""Worker shutdown waits for cancellation cleanups before closing its clients.

A cancelled job settles its budget and writes its failed state in a cleanup
task that can outlive the job. Shutdown must let those finish (or cancel them
after a bound) before the Redis client and the PostgreSQL pool they use close.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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


async def test_shutdown_without_cleanups_closes_clients() -> None:
    events: list[str] = []

    await worker_module.shutdown(_ctx(events))  # type: ignore[arg-type]

    assert events == ["redis_closed", "redis_pool_closed", "pg_pool_closed"]
