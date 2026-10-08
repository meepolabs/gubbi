"""The lifespan starts the orphan cleanup cron with the admin pool and cancels it on shutdown.

Driven through the mocked lifespan harness in ``tests/unit/test_lifespan_boot.py``;
``run_orphan_cleanup`` is replaced with a coroutine that records its arguments
and parks until cancelled, so no database or Redis is needed.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from asgi_lifespan import LifespanManager
from fastapi import FastAPI

import gubbi.main
from gubbi.config import get_settings
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


def _parking_cron(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace the cron with a coroutine that records its call and waits to be cancelled."""
    seen: dict[str, Any] = {"calls": [], "cancelled": False}

    async def _cron(admin_pool: Any, **kwargs: Any) -> None:
        seen["calls"].append((admin_pool, kwargs))
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            seen["cancelled"] = True
            raise

    monkeypatch.setattr("gubbi.main.run_orphan_cleanup", _cron)
    return seen


def _with_admin_pool(handles: dict[str, Any]) -> MagicMock:
    admin_pool = MagicMock()
    admin_pool.get_max_size = MagicMock(return_value=2)
    admin_pool.close = AsyncMock()
    _, _, _, mcp = handles["build_app_ctx"].return_value
    handles["build_app_ctx"].return_value = (handles["app_ctx"], handles["pool"], admin_pool, mcp)
    return admin_pool


async def test_cron_task_runs_against_admin_pool_during_lifespan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With an admin pool, one live cron task is registered and gets the configured threshold."""
    _drop_optional_env(monkeypatch)
    handles = _patch_lifespan_dependencies(monkeypatch)
    admin_pool = _with_admin_pool(handles)
    seen = _parking_cron(monkeypatch)
    app = FastAPI(lifespan=gubbi.main.lifespan)

    async with LifespanManager(app):
        await asyncio.sleep(0)
        tasks = list(app.state.background_tasks)
        assert len(tasks) == 1
        assert not tasks[0].done()
        assert seen["calls"] == [
            (
                admin_pool,
                {
                    "threshold_minutes": get_settings().llm.orphan_cleanup_threshold_minutes,
                    "budget_helper": app.state.budget_helper,
                },
            )
        ]


async def test_cron_task_cancelled_on_shutdown(monkeypatch: pytest.MonkeyPatch) -> None:
    """Lifespan exit cancels the cron task and swallows its CancelledError."""
    _drop_optional_env(monkeypatch)
    handles = _patch_lifespan_dependencies(monkeypatch)
    _with_admin_pool(handles)
    seen = _parking_cron(monkeypatch)
    app = FastAPI(lifespan=gubbi.main.lifespan)

    async with LifespanManager(app):
        await asyncio.sleep(0)
        (task,) = app.state.background_tasks

    assert task.cancelled()
    assert seen["cancelled"] is True


async def test_no_cron_task_without_admin_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without an admin pool the cross-tenant sweep is not started."""
    _drop_optional_env(monkeypatch)
    _patch_lifespan_dependencies(monkeypatch)
    seen = _parking_cron(monkeypatch)
    app = FastAPI(lifespan=gubbi.main.lifespan)

    async with LifespanManager(app):
        await asyncio.sleep(0)
        assert app.state.background_tasks == set()

    assert seen["calls"] == []
