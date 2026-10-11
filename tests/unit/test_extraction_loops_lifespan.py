"""The lifespan owns the orphan sweep and the queue scheduler loops.

Driven through the mocked lifespan harness in ``tests/unit/test_lifespan_boot.py``;
both loops are replaced with coroutines that record their arguments and park
until cancelled, so no database or Redis is needed.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from asgi_lifespan import LifespanManager
from fastapi import FastAPI

import gubbi.main
from gubbi.config import get_settings
from gubbi.extraction.scheduler import SchedulerConfig
from tests.unit.test_lifespan_boot import (
    _drop_optional_env,
    _patch_lifespan_dependencies,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _fresh_settings() -> Iterator[None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _parking_loops(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace both loops with coroutines that record their call and wait to be cancelled."""
    seen: dict[str, Any] = {"calls": {}, "cancelled": set()}

    def _parker(name: str) -> Any:
        async def _loop(*args: Any, **kwargs: Any) -> None:
            seen["calls"][name] = (args, kwargs)
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                seen["cancelled"].add(name)
                raise

        return _loop

    monkeypatch.setattr("gubbi.main.run_orphan_cleanup", _parker("orphan_cleanup"))
    monkeypatch.setattr("gubbi.main.run_extraction_scheduler", _parker("scheduler"))
    return seen


def _with_admin_pool(handles: dict[str, Any]) -> MagicMock:
    admin_pool = MagicMock()
    admin_pool.get_max_size = MagicMock(return_value=2)
    admin_pool.close = AsyncMock()
    _, _, _, mcp = handles["build_app_ctx"].return_value
    handles["build_app_ctx"].return_value = (handles["app_ctx"], handles["pool"], admin_pool, mcp)
    return admin_pool


async def test_both_loops_run_against_the_admin_pool_during_lifespan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _drop_optional_env(monkeypatch)
    monkeypatch.setenv("JOURNAL_EXTRACTION_SCHEDULER_PAUSED", "true")
    handles = _patch_lifespan_dependencies(monkeypatch)
    admin_pool = _with_admin_pool(handles)
    seen = _parking_loops(monkeypatch)
    app = FastAPI(lifespan=gubbi.main.lifespan)

    async with LifespanManager(app):
        await asyncio.sleep(0)
        tasks = list(app.state.background_tasks)
        assert len(tasks) == 2
        assert not any(task.done() for task in tasks)

    arq_pool = handles["arq_create_pool"].return_value
    assert seen["calls"]["orphan_cleanup"] == ((admin_pool,), {})
    scheduler_args, _ = seen["calls"]["scheduler"]
    assert scheduler_args[:2] == (admin_pool, arq_pool)
    assert isinstance(scheduler_args[2], SchedulerConfig)
    assert scheduler_args[2].paused is True


async def test_loops_are_cancelled_on_shutdown(monkeypatch: pytest.MonkeyPatch) -> None:
    _drop_optional_env(monkeypatch)
    handles = _patch_lifespan_dependencies(monkeypatch)
    _with_admin_pool(handles)
    seen = _parking_loops(monkeypatch)
    app = FastAPI(lifespan=gubbi.main.lifespan)

    async with LifespanManager(app):
        await asyncio.sleep(0)
        tasks = set(app.state.background_tasks)

    assert all(task.done() for task in tasks)
    assert seen["cancelled"] == {"orphan_cleanup", "scheduler"}


async def test_without_an_admin_pool_each_loop_is_reported_not_started(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _drop_optional_env(monkeypatch)
    _patch_lifespan_dependencies(monkeypatch)
    seen = _parking_loops(monkeypatch)
    app = FastAPI(lifespan=gubbi.main.lifespan)

    with patch("gubbi.extraction.supervised_loop.record_background_task_died") as died:
        async with LifespanManager(app):
            await asyncio.sleep(0)
            assert app.state.background_tasks == set()

    assert seen["calls"] == {}
    assert sorted(call.kwargs["task"] for call in died.call_args_list) == [
        "extraction_scheduler",
        "orphan_cleanup",
    ]
    assert {call.kwargs["stage"] for call in died.call_args_list} == {"not_started"}
