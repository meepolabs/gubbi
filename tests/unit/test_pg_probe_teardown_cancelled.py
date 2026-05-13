"""Lifespan / worker startup teardown propagates CancelledError intact.

The pg_log_probe handlers in both ``gubbi.main`` (HTTP lifespan) and
``gubbi.extraction.worker`` (Arq worker startup) catch ``BaseException``
so a probe-time cancellation still trips pool teardown before unwinding.
The teardown itself (``pool.close()``) is best-effort and was previously
guarded by ``with suppress(Exception)``.

That guard was wrong: under Python 3.8+, ``asyncio.CancelledError`` is
a ``BaseException``, not an ``Exception``. A second cancellation
arriving INSIDE ``pool.close()`` would therefore escape the suppress
block and clobber the original cancellation that the surrounding
``raise`` is meant to re-raise.

These tests pin the new behaviour: if ``pool.close()`` raises
``CancelledError`` mid-teardown, the suppress block swallows it, the
pool's ``close()`` is still awaited, and the original probe error
propagates.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from asgi_lifespan import LifespanManager
from fastapi import FastAPI

import gubbi.main
from gubbi.extraction import worker as worker_module
from tests.extraction.test_worker_startup_pg_probe import _patch_worker_dependencies
from tests.unit.test_lifespan_boot import (
    _drop_optional_env,
    _patch_lifespan_dependencies,
)


class _ProbeBoom(Exception):
    """Distinct exception class so we can assert it is the propagated error."""


@pytest.mark.unit
async def test_lifespan_pool_close_cancelled_error_does_not_mask_probe_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A CancelledError from ``pool.close()`` must NOT escape the suppress block.

    Wire the probe to raise a plain ``Exception`` ("ProbeBoom"); wire
    ``pool.close`` to raise ``CancelledError``. The original probe error
    must still propagate -- not the cancellation.
    """
    _drop_optional_env(monkeypatch)
    handles = _patch_lifespan_dependencies(monkeypatch)

    # Probe raises a sentinel error.
    probe_mock = AsyncMock(side_effect=_ProbeBoom("simulated unsafe log_statement"))
    monkeypatch.setattr("gubbi.main.probe_pg_log_settings", probe_mock)

    # Pool close raises CancelledError mid-teardown. The suppress block
    # in main.py must catch it (BaseException-aware suppress) so the
    # original ProbeBoom propagates.
    handles["pool"].close = AsyncMock(side_effect=asyncio.CancelledError())

    app = FastAPI(lifespan=gubbi.main.lifespan)

    with pytest.raises(_ProbeBoom):
        async with LifespanManager(app):
            pass

    # Pool close was attempted (best-effort teardown ran) before the
    # original error propagated.
    handles["pool"].close.assert_awaited_once()


@pytest.mark.unit
async def test_worker_startup_pool_close_cancelled_error_does_not_mask_probe_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Worker variant: ``CancelledError`` from ``pool.close`` is suppressed."""
    handles = _patch_worker_dependencies(monkeypatch)

    probe_mock = AsyncMock(side_effect=_ProbeBoom("simulated unsafe log_statement"))
    monkeypatch.setattr(worker_module, "probe_pg_log_settings", probe_mock)

    handles["pool"].close = AsyncMock(side_effect=asyncio.CancelledError())

    ctx: dict[str, Any] = {}
    with pytest.raises(_ProbeBoom):
        await worker_module.startup(ctx)  # type: ignore[arg-type]

    handles["pool"].close.assert_awaited_once()


@pytest.mark.unit
async def test_lifespan_admin_pool_close_cancelled_error_is_suppressed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Admin pool close raising ``CancelledError`` must not mask the probe error.

    The lifespan path checks ``admin_pool is not None`` before attempting
    its close; this test wires both pools so both close() calls run.
    """
    _drop_optional_env(monkeypatch)
    handles = _patch_lifespan_dependencies(monkeypatch)

    # Inject an admin_pool into the build_app_ctx return so the optional
    # close() branch fires. The original mock returned (app_ctx, pool, None, mcp);
    # rebuild a tuple with admin_pool populated. Pull the original return
    # value off the existing AsyncMock (call_count is 0 here).
    original_app_ctx = handles["app_ctx"]
    original_pool = handles["pool"]
    # Reuse a fresh MagicMock for mcp since handles dict doesn't expose it.
    from contextlib import asynccontextmanager

    mcp_app = MagicMock()
    mcp = MagicMock()
    mcp.streamable_http_app = MagicMock(return_value=mcp_app)

    @asynccontextmanager
    async def _session_run() -> Any:
        yield None

    mcp.session_manager = MagicMock()
    mcp.session_manager.run = _session_run

    admin_pool = MagicMock()
    admin_pool.close = AsyncMock(side_effect=asyncio.CancelledError())

    build_app_ctx_mock = AsyncMock(return_value=(original_app_ctx, original_pool, admin_pool, mcp))
    monkeypatch.setattr("gubbi.main._build_app_ctx", build_app_ctx_mock)

    probe_mock = AsyncMock(side_effect=_ProbeBoom("unsafe"))
    monkeypatch.setattr("gubbi.main.probe_pg_log_settings", probe_mock)

    # Make the regular pool's close cleanly succeed so we isolate the
    # admin-pool branch.
    original_pool.close = AsyncMock()

    app = FastAPI(lifespan=gubbi.main.lifespan)

    with pytest.raises(_ProbeBoom):
        async with LifespanManager(app):
            pass

    original_pool.close.assert_awaited_once()
    admin_pool.close.assert_awaited_once()
