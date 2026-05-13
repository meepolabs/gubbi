"""Lifespan integration: pg_log_probe wired after pool init.

Two scenarios:

* WARN/STRICT-with-safe-settings -- probe is invoked exactly once with
  ``app_pool`` and the resolved ``mode`` keyword; lifespan continues
  through to the yield point.
* STRICT with the probe raising -- lifespan tears down both pools and
  propagates the original error; no resource leak.

Reuses the dependency-stub helpers from ``test_lifespan_boot.py`` so
the test only differs in the probe wiring assertion.
"""

from __future__ import annotations

import importlib
from unittest.mock import AsyncMock

import pytest
from asgi_lifespan import LifespanManager
from fastapi import FastAPI
from gubbi_common.bootstrap.pg_log_probe import PgLogProbeError

import gubbi.main
from tests.unit.test_lifespan_boot import (
    _drop_optional_env,
    _patch_lifespan_dependencies,
)


@pytest.mark.unit
async def test_lifespan_calls_pg_log_probe_after_pool_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The lifespan invokes ``probe_pg_log_settings`` exactly once with the app pool.

    Default ``mode`` resolves to ``strict`` from
    ``JOURNAL_PG_LOG_PROBE_MODE`` (env var unset -> strict). The probe
    is async; ``AsyncMock`` ensures the await is well-formed.
    """
    _drop_optional_env(monkeypatch)
    monkeypatch.delenv("JOURNAL_PG_LOG_PROBE_MODE", raising=False)
    handles = _patch_lifespan_dependencies(monkeypatch)

    probe_mock = AsyncMock(return_value=None)
    monkeypatch.setattr("gubbi.main.probe_pg_log_settings", probe_mock)

    app = FastAPI(lifespan=gubbi.main.lifespan)

    async with LifespanManager(app):
        assert app.state.app_ctx is handles["app_ctx"]

    # Probe called exactly once with the app pool stub and a mode= keyword.
    probe_mock.assert_awaited_once()
    args, kwargs = probe_mock.call_args
    assert args == (handles["pool"],), f"probe called with unexpected args: {args}"
    assert (
        kwargs.get("mode") == "strict"
    ), f"default JOURNAL_PG_LOG_PROBE_MODE should resolve to 'strict'; got {kwargs}"


@pytest.mark.unit
async def test_lifespan_calls_pg_log_probe_with_warn_mode_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``JOURNAL_PG_LOG_PROBE_MODE=warn`` is forwarded as the ``mode`` kwarg."""
    _drop_optional_env(monkeypatch)
    monkeypatch.setenv("JOURNAL_PG_LOG_PROBE_MODE", "warn")
    _patch_lifespan_dependencies(monkeypatch)

    probe_mock = AsyncMock(return_value=None)
    monkeypatch.setattr("gubbi.main.probe_pg_log_settings", probe_mock)

    app = FastAPI(lifespan=gubbi.main.lifespan)

    async with LifespanManager(app):
        pass

    _, kwargs = probe_mock.call_args
    assert kwargs.get("mode") == "warn"


@pytest.mark.unit
async def test_lifespan_aborts_when_pg_probe_raises_strict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the probe raises ``PgLogProbeError`` the lifespan tears down the pool.

    We only require that the exception propagates (the ASGI lifespan
    fails) and the app pool's ``close`` was awaited; the rest of the
    teardown chain is exercised by ``test_lifespan_boot``.
    """
    _drop_optional_env(monkeypatch)
    handles = _patch_lifespan_dependencies(monkeypatch)

    probe_mock = AsyncMock(side_effect=PgLogProbeError("unsafe log_statement=all"))
    monkeypatch.setattr("gubbi.main.probe_pg_log_settings", probe_mock)

    app = FastAPI(lifespan=gubbi.main.lifespan)

    with pytest.raises(PgLogProbeError):
        async with LifespanManager(app):
            pass

    handles["pool"].close.assert_awaited_once()


@pytest.mark.unit
def test_main_module_exposes_probe_pg_log_settings_symbol() -> None:
    """``probe_pg_log_settings`` is imported at module scope so monkeypatch works.

    The two scenarios above patch ``gubbi.main.probe_pg_log_settings``;
    that path only resolves when the symbol is bound on the module --
    a deferred ``import`` inside the lifespan body would skip the patch
    and silently bypass these tests.
    """
    importlib.reload(gubbi.main)
    assert hasattr(gubbi.main, "probe_pg_log_settings"), (
        "gubbi.main must import probe_pg_log_settings at module scope so "
        "test monkeypatch can intercept the call."
    )
