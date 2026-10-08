"""Startup teardown swallows a cancelled close() so the probe failure propagates.

When a required startup probe fails, both ``gubbi.main.lifespan`` (through
``teardown_lifespan_resources``) and ``gubbi.extraction.worker.startup``
close what they opened, best-effort, before re-raising. ``CancelledError``
is a ``BaseException``, so a teardown guarded only by ``suppress(Exception)``
would let a cancellation arriving inside ``close()`` escape and replace the
``ProbeFailure`` the caller is meant to see.

Each test drives the real ``StartupRunner`` into a required-probe failure,
makes one ``close()`` raise ``CancelledError``, and asserts that the
``ProbeFailure`` propagates and that the remaining teardown still ran.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import redis.exceptions as redis_exc
from asgi_lifespan import LifespanManager
from fastapi import FastAPI
from gubbi_common.bootstrap import PgLogProbeError, ProbeFailure, StartupRunner

import gubbi.main
from gubbi.config import get_settings
from gubbi.extraction import worker as worker_module
from tests.extraction.test_worker_startup_pg_probe import _patch_worker_dependencies
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


def _fail_lifespan_redis_probe(monkeypatch: pytest.MonkeyPatch, handles: dict[str, Any]) -> None:
    """Make the lifespan's required Redis probe fail under the real runner."""
    handles["redis_client"].ping = AsyncMock(
        side_effect=redis_exc.ConnectionError("simulated_redis_unreachable")
    )
    monkeypatch.setattr("gubbi.main.StartupRunner", StartupRunner)


async def test_lifespan_pool_close_cancelled_error_does_not_mask_probe_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancelled ``pool.close()`` leaves the lifespan's ``ProbeFailure`` intact."""
    _drop_optional_env(monkeypatch)
    handles = _patch_lifespan_dependencies(monkeypatch)
    _fail_lifespan_redis_probe(monkeypatch, handles)
    handles["pool"].close = AsyncMock(side_effect=asyncio.CancelledError())

    app = FastAPI(lifespan=gubbi.main.lifespan)
    with pytest.raises(ProbeFailure):
        async with LifespanManager(app):
            pass

    handles["redis_client"].aclose.assert_awaited_once()
    handles["oauth_storage"].close.assert_awaited_once()
    handles["pool"].close.assert_awaited_once()


async def test_lifespan_cancelled_pool_close_still_closes_admin_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Teardown continues past a cancelled ``pool.close()`` to a cancelled admin pool close."""
    _drop_optional_env(monkeypatch)
    handles = _patch_lifespan_dependencies(monkeypatch)
    _fail_lifespan_redis_probe(monkeypatch, handles)

    admin_pool = MagicMock()
    admin_pool.get_max_size = MagicMock(return_value=2)
    admin_pool.close = AsyncMock(side_effect=asyncio.CancelledError())
    _, _, _, mcp = handles["build_app_ctx"].return_value
    handles["build_app_ctx"].return_value = (handles["app_ctx"], handles["pool"], admin_pool, mcp)
    handles["pool"].close = AsyncMock(side_effect=asyncio.CancelledError())

    app = FastAPI(lifespan=gubbi.main.lifespan)
    with pytest.raises(ProbeFailure):
        async with LifespanManager(app):
            pass

    handles["pool"].close.assert_awaited_once()
    admin_pool.close.assert_awaited_once()


async def test_worker_startup_pool_close_cancelled_error_does_not_mask_probe_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Worker variant: a cancelled ``pool.close()`` leaves ``ProbeFailure`` intact."""
    handles = _patch_worker_dependencies(monkeypatch)
    monkeypatch.setattr(worker_module, "_configure_worker_telemetry", MagicMock())
    monkeypatch.setattr(
        "gubbi_common.bootstrap.probes.pg_log._probe_pg_log_settings",
        AsyncMock(side_effect=PgLogProbeError("unsafe log_statement=all")),
    )
    handles["pool"].close = AsyncMock(side_effect=asyncio.CancelledError())

    ctx: dict[str, Any] = {}
    with pytest.raises(ProbeFailure):
        await worker_module.startup(ctx)  # type: ignore[arg-type]

    handles["redis_client"].aclose.assert_awaited_once()
    handles["redis_pool"].aclose.assert_awaited_once()
    handles["pool"].close.assert_awaited_once()
