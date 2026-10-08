"""Lifespan integration: PgLogProbe wired into StartupRunner.

The legacy probe was a bare ``probe_pg_log_settings`` call directly in
the lifespan body. After T1 it is composed as :class:`PgLogProbe` and
sequenced by :class:`StartupRunner`. This test pins the new shape:

* SUCCESS path -- the lifespan reaches the yield point, probe ran with
  ``settings.pg_log_probe_mode`` (default STRICT).
* FAILURE path -- when the probe raises (STRICT mode + unsafe GUC),
  ``StartupRunner`` raises ``ProbeFailure`` and the lifespan's outer
  teardown helper closes the app pool exactly once.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass

import pytest
from asgi_lifespan import LifespanManager
from fastapi import FastAPI
from gubbi_common.bootstrap import (
    ProbeFailure,
    ProbeResult,
    ProbeStatus,
    StartupProbe,
    StartupRunner,
)

import gubbi.main
from tests.unit.test_lifespan_boot import (
    _drop_optional_env,
    _patch_lifespan_dependencies,
)


@dataclass
class _AlwaysFailPgLogProbe:
    """Stand-in :class:`StartupProbe` that always fails as ``pg_log``."""

    name: str = "pg_log"
    required: bool = True
    timeout_s: float = 5.0

    async def run(self) -> ProbeResult:
        return ProbeResult(
            ProbeStatus.FAIL,
            diagnostic={"mode": "strict", "findings": "unsafe log_statement=all"},
        )


@pytest.mark.unit
async def test_lifespan_uses_settings_pg_log_probe_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lifespan composes ``PgLogProbe`` from ``settings.pg_log_probe_mode``.

    Verified by patching :class:`StartupRunner` to capture the probes
    handed to ``run()`` -- no real Postgres GUC fetch happens. The test
    asserts that:

    * a probe named ``pg_log`` is in the sequence;
    * its ``mode`` matches the default ``STRICT`` resolved from the
      Settings field (env var unset).
    """
    _drop_optional_env(monkeypatch)
    monkeypatch.delenv("JOURNAL_PG_LOG_PROBE_MODE", raising=False)
    _patch_lifespan_dependencies(monkeypatch)

    captured: list[StartupProbe] = []

    class _CaptureRunner:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        async def run(self, probes: list[StartupProbe], **kwargs: object) -> tuple[()]:
            del kwargs
            captured.extend(probes)
            return ()

    monkeypatch.setattr("gubbi.main.StartupRunner", _CaptureRunner)

    app = FastAPI(lifespan=gubbi.main.lifespan)
    async with LifespanManager(app):
        pass

    pg_log_probes = [p for p in captured if p.name == "pg_log"]
    assert len(pg_log_probes) == 1, f"expected exactly one pg_log probe, got {pg_log_probes}"
    # Default JOURNAL_PG_LOG_PROBE_MODE resolves to STRICT via Settings.
    assert pg_log_probes[0].mode.value == "strict"  # type: ignore[attr-defined]


@pytest.mark.unit
async def test_lifespan_aborts_when_pg_probe_fails_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Required ``pg_log`` FAIL escalates to :class:`ProbeFailure`; pool tears down.

    We replace the runner with a real :class:`StartupRunner` instance
    but inject an ``_AlwaysFailPgLogProbe`` ahead of the real probe
    list by stubbing the constructor. Simpler path: monkeypatch
    StartupRunner.run to receive a probe list whose pg_log probe
    fails. The teardown helper closes the app pool exactly once via
    the lifespan's outer ``finally``.
    """
    _drop_optional_env(monkeypatch)
    handles = _patch_lifespan_dependencies(monkeypatch)

    failing_probe = _AlwaysFailPgLogProbe()

    class _InjectFailingPgLog:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            self._inner = StartupRunner(app_env="dev")

        async def run(self, probes: list[StartupProbe], **kwargs: object) -> tuple[()]:
            # Replace the real pg_log probe with the failing stand-in;
            # leave the rest in place so the runner's required-fail
            # escalation still happens at the right ordinal.
            replaced = [failing_probe if p.name == "pg_log" else p for p in probes]
            return await self._inner.run(replaced, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("gubbi.main.StartupRunner", _InjectFailingPgLog)

    app = FastAPI(lifespan=gubbi.main.lifespan)

    with pytest.raises(ProbeFailure):
        async with LifespanManager(app):
            pass

    handles["pool"].close.assert_awaited_once()


@pytest.mark.unit
def test_main_module_exposes_startup_runner_symbol() -> None:
    """``StartupRunner`` is bound on ``gubbi.main`` so monkeypatch can intercept.

    The two scenarios above patch ``gubbi.main.StartupRunner``; that
    path only resolves when the symbol is bound on the module --
    a deferred ``import`` inside the lifespan body would skip the
    patch and silently bypass these tests.
    """
    importlib.reload(gubbi.main)
    assert hasattr(gubbi.main, "StartupRunner"), (
        "gubbi.main must import StartupRunner at module scope so test "
        "monkeypatch can intercept the construct + run."
    )
