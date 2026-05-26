"""Worker startup wires PgLogProbe + WorkerReplicaCountWarnProbe through StartupRunner.

After T3 the Arq worker mirrors the gubbi HTTP lifespan: probes are
constructed and handed to ``StartupRunner.run()``. The runner emits
structured ``startup.probe.<outcome>`` events; required-probe FAIL
surfaces as ``ProbeFailure``.

Tests cover:

* Probes are constructed with the right shape (pool + Settings-derived
  mode for PgLog; replica_count + live pool max for WorkerReplicaCount).
* PgLog probe FAIL escalates to ``ProbeFailure``; the worker's outer
  ``except BaseException`` closes the pool best-effort before the error
  propagates.
* Settings rejects invalid ``JOURNAL_REPLICA_COUNT`` at construction
  (one level UP from the worker, after T3 deleted ``_validate_replica_count``).
* WorkerReplicaCountWarnProbe emits the alertable counter + structured
  WARN when ``replica_count > 1``; default 1 stays silent.
* OTel wiring stays one-shot per process; a configure-time failure leaves
  the latch unset so a retry can re-try.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import structlog
from gubbi_common.bootstrap import (
    PgLogProbeError,
    PgLogProbeMode,
    ProbeFailure,
)
from gubbi_common.bootstrap.probes import PgLogProbe

from gubbi.bootstrap.probes.worker_replica_count import WorkerReplicaCountWarnProbe
from gubbi.extraction import worker as worker_module

# Stub app-pool max size returned by ``pool.get_max_size()``. The replica
# probe reads the LIVE pool max via ``get_max_size()`` (not a constant), so
# the stub pool must answer with a real int. Kept as a local test constant
# so the assertion pins the live-read contract rather than a particular
# production-configured size.
_STUB_APP_POOL_MAX = 12


@pytest.fixture(autouse=True)
def _reset_worker_telemetry_guard_between_tests() -> Iterator[None]:
    """Re-arm the one-shot telemetry guard before and after each test.

    ``_configure_worker_telemetry`` latches ``_WORKER_TELEMETRY_CONFIGURED``
    on first success so a second ``startup()`` in the SAME process does not
    rebuild exporters/readers (production intent). Without resetting it
    between tests, the first test that drives the real
    ``_configure_worker_telemetry`` would latch the guard for the rest of
    the session and silently skip wiring in later tests -- masking
    regressions. Reset on both sides so order does not matter.
    """
    worker_module._reset_worker_telemetry_guard()
    yield
    worker_module._reset_worker_telemetry_guard()


def _patch_worker_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    """Stub the heavy dependencies the worker startup hook would normally hit.

    Does NOT stub ``_configure_worker_telemetry``: tests that exercise the
    one-shot guard need the real function. Tests that don't care about
    telemetry wiring add a local stub. ``OTEL_ENABLED=false`` in the
    suite's env (see ``tests/conftest.py``) means the real wiring is
    a benign no-op (SDK providers with no exporter).

    Critically, the ``threading.Thread`` constructor used by the worker
    is patched on the **module's** ``threading`` reference -- NOT on the
    stdlib ``threading`` module -- so global Thread creation (asyncio
    executors, etc.) stays untouched. The health-server thread the
    worker creates is replaced with a no-op stand-in whose ``start`` is
    a sync ``MagicMock``.
    """
    pool = MagicMock()
    pool.close = AsyncMock()
    # The replica probe reads the live per-pod pool max via get_max_size();
    # a bare MagicMock would return a MagicMock (not an int) and break the
    # arithmetic. Answer with a real int.
    pool.get_max_size = MagicMock(return_value=_STUB_APP_POOL_MAX)

    # PgLogProbe (when not stubbed at the runner level) reaches into the
    # pool via ``async with pool.acquire() as conn: await
    # conn.fetchval(...)``. Stub the connection so fetchval returns None
    # (the safe default -- the probe's STRICT predicate treats ``None`` as
    # "GUC not set" and returns OK).
    conn_stub = MagicMock()
    conn_stub.fetchval = AsyncMock(return_value=None)
    acquire_cm = MagicMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn_stub)
    acquire_cm.__aexit__ = AsyncMock(return_value=None)
    pool.acquire = MagicMock(return_value=acquire_cm)

    monkeypatch.setattr(worker_module, "init_pool", AsyncMock(return_value=pool))

    redis_pool_stub = MagicMock()
    redis_pool_stub.aclose = AsyncMock()
    redis_client_stub = MagicMock()
    redis_client_stub.aclose = AsyncMock()
    # T2 follow-up: the worker now PINGs Redis at startup via
    # ``RedisPingProbe`` BEFORE the BudgetHelper wiring runs. Default to
    # a successful PONG so the smoke fixture's happy path works; tests
    # that exercise the probe failure path override ``ping`` on the
    # returned handle.
    redis_client_stub.ping = AsyncMock(return_value=b"PONG")
    redis_client_stub.register_script = MagicMock(return_value=MagicMock())

    monkeypatch.setattr(
        "redis.asyncio.ConnectionPool.from_url",
        MagicMock(return_value=redis_pool_stub),
    )
    monkeypatch.setattr(
        "redis.asyncio.Redis",
        MagicMock(return_value=redis_client_stub),
    )

    # Replace the worker's ``threading`` reference so the health-server
    # thread is a no-op. Don't patch stdlib ``threading`` -- that would
    # break asyncio's internal Thread usage.
    fake_threading = MagicMock()
    fake_thread = MagicMock()
    fake_thread.start = MagicMock()
    fake_threading.Thread = MagicMock(return_value=fake_thread)
    monkeypatch.setattr(worker_module, "threading", fake_threading)

    monkeypatch.setattr(worker_module, "initialize_logger", MagicMock())
    monkeypatch.setattr(worker_module, "_build_content_cipher", MagicMock(return_value=None))

    # The Anthropic provider is constructed inside ``startup`` -- patch
    # it to a no-op so no real API client is built.
    monkeypatch.setattr(worker_module, "AnthropicProvider", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(worker_module, "ExtractionService", MagicMock(return_value=MagicMock()))

    # The probe outcome counter helper is bound at import time; tests
    # that don't care about counter emission need a no-op so the runner
    # callback path is inert.
    monkeypatch.setattr(worker_module, "record_startup_probe_outcome", MagicMock())

    return {
        "pool": pool,
        "redis_client": redis_client_stub,
        "redis_pool": redis_pool_stub,
    }


def _stub_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub ``_configure_worker_telemetry`` for tests that don't care about it."""
    monkeypatch.setattr(worker_module, "_configure_worker_telemetry", MagicMock())


def _stub_runner_with_capture(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace ``StartupRunner`` with a no-op that captures the probes list.

    Returns a handle whose ``probes`` key is populated after ``runner.run``
    is awaited. Tests use this to assert which probes the worker
    constructs and with which keyword arguments.
    """
    captured: dict[str, Any] = {"probes": None, "constructor_kwargs": None}

    class _CapturingRunner:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args
            captured["constructor_kwargs"] = kwargs

        async def run(self, probes: object, **kwargs: object) -> tuple[()]:
            del kwargs
            captured["probes"] = probes
            return ()

    monkeypatch.setattr(worker_module, "StartupRunner", _CapturingRunner)
    return captured


# ---------------------------------------------------------------------------
# Probe wiring: the worker constructs PgLogProbe + WorkerReplicaCountWarnProbe
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_worker_startup_constructs_pg_log_probe_with_strict_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``startup`` builds a PgLogProbe carrying the worker pool + STRICT mode."""
    monkeypatch.delenv("JOURNAL_PG_LOG_PROBE_MODE", raising=False)
    handles = _patch_worker_dependencies(monkeypatch)
    _stub_telemetry(monkeypatch)
    captured = _stub_runner_with_capture(monkeypatch)

    from gubbi.config import get_settings

    get_settings.cache_clear()

    ctx: dict[str, Any] = {}
    await worker_module.startup(ctx)  # type: ignore[arg-type]

    probes = captured["probes"]
    assert probes is not None, "runner.run was never invoked"
    pg_probes = [p for p in probes if isinstance(p, PgLogProbe)]
    assert len(pg_probes) == 1
    assert pg_probes[0].pool is handles["pool"]
    assert pg_probes[0].mode is PgLogProbeMode.STRICT

    get_settings.cache_clear()


@pytest.mark.unit
async def test_worker_startup_forwards_warn_mode_from_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``JOURNAL_PG_LOG_PROBE_MODE=warn`` flows through Settings into the probe."""
    monkeypatch.setenv("JOURNAL_PG_LOG_PROBE_MODE", "warn")
    _patch_worker_dependencies(monkeypatch)
    _stub_telemetry(monkeypatch)
    captured = _stub_runner_with_capture(monkeypatch)

    from gubbi.config import get_settings

    get_settings.cache_clear()

    ctx: dict[str, Any] = {}
    await worker_module.startup(ctx)  # type: ignore[arg-type]

    probes = captured["probes"]
    pg_probes = [p for p in probes if isinstance(p, PgLogProbe)]
    assert len(pg_probes) == 1
    assert pg_probes[0].mode is PgLogProbeMode.WARN

    get_settings.cache_clear()


@pytest.mark.unit
async def test_worker_startup_probe_order_is_pg_log_redis_ping_then_replica_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Worker probe order is structurally pinned: PgLog -> RedisPing -> WorkerReplicaCount.

    Mirrors the gubbi HTTP lifespan probe trace (PgLog -> RedisPing ->
    ReplicaCount). Adding/reordering worker probes will force this test
    to update so the order is reviewed deliberately, not silently
    shuffled. RedisPingProbe was added in T2 follow-up so a dead Redis
    surfaces at boot rather than at first arq poll.
    """
    from gubbi_common.bootstrap.probes import RedisPingProbe

    _patch_worker_dependencies(monkeypatch)
    _stub_telemetry(monkeypatch)
    captured = _stub_runner_with_capture(monkeypatch)

    from gubbi.config import get_settings

    get_settings.cache_clear()

    ctx: dict[str, Any] = {}
    await worker_module.startup(ctx)  # type: ignore[arg-type]

    probes = captured["probes"]
    assert probes is not None
    assert len(probes) == 3
    assert isinstance(probes[0], PgLogProbe)
    assert isinstance(probes[1], RedisPingProbe)
    assert isinstance(probes[2], WorkerReplicaCountWarnProbe)

    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# PgLog probe FAIL: ProbeFailure propagates; pool closed by outer except.
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_worker_startup_aborts_when_pg_probe_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unsafe pg_log GUC raises ProbeFailure and closes the worker pool.

    After T3 the runner is the failure-translation layer:
    ``probe_pg_log_settings`` raising ``PgLogProbeError`` inside the
    PgLogProbe surfaces as ``ProbeFailure`` from ``runner.run()``. The
    outer ``except BaseException`` block in the worker's startup body
    still closes the pool best-effort before the error propagates to
    the Arq runtime.
    """
    handles = _patch_worker_dependencies(monkeypatch)
    _stub_telemetry(monkeypatch)
    # Use the real StartupRunner so the PgLogProbe actually executes; stub
    # the underlying ``probe_pg_log_settings`` at the probe's import path
    # to raise the canonical PgLogProbeError.
    monkeypatch.setattr(
        "gubbi_common.bootstrap.probes.pg_log.probe_pg_log_settings",
        AsyncMock(side_effect=PgLogProbeError("unsafe log_statement=all")),
    )

    from gubbi.config import get_settings

    get_settings.cache_clear()

    ctx: dict[str, Any] = {}
    with pytest.raises(ProbeFailure):
        await worker_module.startup(ctx)  # type: ignore[arg-type]

    handles["pool"].close.assert_awaited_once()

    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Replica-count over-provisioning WARN + alertable counter (#138 worker variant).
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_worker_startup_warns_and_increments_counter_when_replicas_gt_1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """JOURNAL_REPLICA_COUNT=2 -> startup.probe.warn + record_replica_count_warning.

    The worker is fixed at a single replica by deploy convention. Replica
    count > 1 multiplies the per-pod extraction-budget logic + DB pool
    N-fold AND violates the single-worker policy. After T3 the
    WorkerReplicaCountWarnProbe emits both signals through the runner.

    Emitter distinction across services (gubbi HTTP / worker / cloud-api)
    is via the ``service.name`` RESOURCE attribute set at OTel init, not
    via a metric attribute -- the helper takes only ``replica_count``.
    """
    _patch_worker_dependencies(monkeypatch)
    _stub_telemetry(monkeypatch)
    monkeypatch.setenv("JOURNAL_REPLICA_COUNT", "2")

    from gubbi.config import get_settings

    get_settings.cache_clear()

    # Patch the counter helper at the probe's import path so the alertable
    # signal can be asserted without standing up a real OTel meter.
    record_mock = MagicMock()
    monkeypatch.setattr(
        "gubbi.bootstrap.probes.worker_replica_count.record_replica_count_warning",
        record_mock,
    )

    ctx: dict[str, Any] = {}
    with structlog.testing.capture_logs() as logs:
        await worker_module.startup(ctx)  # type: ignore[arg-type]

    record_mock.assert_called_once_with(replica_count=2)

    # The runner's structured WARN event carries the policy-violation
    # diagnostic the WorkerReplicaCountWarnProbe builds.
    warnings = [
        log
        for log in logs
        if log.get("event") == "startup.probe.warn"
        and log.get("name") == "worker_replica_count"
        and log.get("log_level") == "warning"
    ]
    assert len(warnings) == 1, f"expected one worker_replica_count WARN, got {warnings}"
    diagnostic = warnings[0]["diagnostic"]
    assert diagnostic["replica_count"] == 2
    # Worker opens ONLY the app pool (no admin pool), so per-pod footprint
    # is the live app pool max via get_max_size().
    assert diagnostic["pool_max_per_pod"] == _STUB_APP_POOL_MAX
    assert diagnostic["pool_max_total"] == _STUB_APP_POOL_MAX * 2
    # The note must flag the single-worker deploy-policy violation.
    assert "single" in diagnostic["note"].lower()

    get_settings.cache_clear()


@pytest.mark.unit
async def test_worker_startup_silent_at_default_replica_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default (unset) JOURNAL_REPLICA_COUNT=1 -> no WARN, no counter."""
    _patch_worker_dependencies(monkeypatch)
    _stub_telemetry(monkeypatch)
    monkeypatch.delenv("JOURNAL_REPLICA_COUNT", raising=False)

    from gubbi.config import get_settings

    get_settings.cache_clear()

    record_mock = MagicMock()
    monkeypatch.setattr(
        "gubbi.bootstrap.probes.worker_replica_count.record_replica_count_warning",
        record_mock,
    )

    ctx: dict[str, Any] = {}
    with structlog.testing.capture_logs() as logs:
        await worker_module.startup(ctx)  # type: ignore[arg-type]

    record_mock.assert_not_called()
    assert not [
        log
        for log in logs
        if log.get("event") == "startup.probe.warn" and log.get("name") == "worker_replica_count"
    ], "default single-worker deploy must stay silent"

    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Replica count validation moved one level up: Settings construction.
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("value", "expected_match"),
    [
        ("abc", "Input should be a valid integer"),
        ("0", "greater than or equal to 1"),
        ("-2", "greater than or equal to 1"),
    ],
)
def test_settings_rejects_invalid_replica_count(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
    expected_match: str,
) -> None:
    """Invalid ``JOURNAL_REPLICA_COUNT`` fails Settings construction.

    Pre-T3 the worker's local ``_validate_replica_count`` raised
    ``RuntimeError`` from inside ``startup()``. After T3 the worker no
    longer re-validates -- ``Settings.replica_count`` (default 1, ge=1)
    catches the bad value at Settings construction, BEFORE
    ``worker.startup()`` runs. The test pins the validation layer at
    Settings, not at the worker.
    """
    import pydantic

    from gubbi.config import Settings, get_settings

    monkeypatch.setenv("JOURNAL_REPLICA_COUNT", value)
    get_settings.cache_clear()

    with pytest.raises(pydantic.ValidationError, match=expected_match):
        Settings()

    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Worker OTel wiring is one-shot per process (#138 re-entry guard).
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_worker_telemetry_wiring_is_one_shot_across_two_startups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second ``startup()`` in the same process re-uses the first wiring.

    OTel's ``set_*_provider`` Once-guards reject a second registration AFTER
    the new providers + their exporter threads are built, so re-wiring on
    every startup would leak threads and log "Overriding ... Provider is not
    allowed". The module-level guard makes ``configure_otel`` run at most
    once; the contract is observable as ``_WORKER_TELEMETRY_CONFIGURED``
    transitioning False -> True on the first call and STAYING True across
    every subsequent startup.

    Asserting the flag transitions decouples this test from any specific
    import path inside ``_configure_worker_telemetry``: a refactor that
    promotes the lazy ``from ... import ...`` to a top-level import will
    not break this test as long as the latch contract is preserved.
    """
    _patch_worker_dependencies(monkeypatch)
    # Note: do NOT call _stub_telemetry() -- this test exercises the real
    # _configure_worker_telemetry function.
    monkeypatch.setattr(
        worker_module,
        "StartupRunner",
        MagicMock(return_value=MagicMock(run=AsyncMock(return_value=()))),
    )
    monkeypatch.delenv("JOURNAL_REPLICA_COUNT", raising=False)

    # Stub the SDK init the wiring would otherwise run -- no real exporter
    # threads or providers in the test process. The test does NOT assert on
    # the call count of these stubs; the flag transitions are the contract.
    monkeypatch.setattr("gubbi_common.telemetry.otel.configure_otel", MagicMock())
    monkeypatch.setattr(worker_module, "rebind_metrics_after_configure", MagicMock())

    from gubbi.config import get_settings

    get_settings.cache_clear()

    ctx: dict[str, Any] = {}
    assert worker_module._WORKER_TELEMETRY_CONFIGURED is False, (
        "precondition: the autouse guard fixture must reset the flag "
        "before each test so the latch transition is observable"
    )

    await worker_module.startup(ctx)  # type: ignore[arg-type]
    assert worker_module._WORKER_TELEMETRY_CONFIGURED is True, (
        "first startup must latch the guard so subsequent startups "
        "short-circuit before re-installing exporters/readers"
    )

    # Second startup must not raise (the guard short-circuits inside
    # _configure_worker_telemetry) and must leave the flag latched.
    await worker_module.startup(ctx)  # type: ignore[arg-type]
    assert worker_module._WORKER_TELEMETRY_CONFIGURED is True, (
        "second startup must leave the guard latched -- a False reading "
        "here would mean the wiring re-ran and the guard failed"
    )

    get_settings.cache_clear()


@pytest.mark.unit
async def test_worker_telemetry_guard_unlatched_on_configure_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A configure-time failure leaves the guard unset so a retry can re-try.

    The guard latches only after a clean wiring. If ``configure_otel`` raises
    (configure/import-time failure -- swallowed, logged at ERROR), the guard
    stays unset so a subsequent ``startup`` attempts the wiring again rather
    than being permanently skipped.

    Asserting on the flag (rather than on call counts via the lazy-imported
    SDK function) keeps the test independent of where the import is resolved
    inside ``_configure_worker_telemetry``.
    """
    _patch_worker_dependencies(monkeypatch)
    # Note: do NOT call _stub_telemetry() -- this test exercises the real
    # _configure_worker_telemetry function (specifically its swallow-and-log
    # branch).
    monkeypatch.setattr(
        worker_module,
        "StartupRunner",
        MagicMock(return_value=MagicMock(run=AsyncMock(return_value=()))),
    )
    monkeypatch.delenv("JOURNAL_REPLICA_COUNT", raising=False)

    # Stub the SDK init to raise -- exercises the swallow-and-log branch
    # inside _configure_worker_telemetry that intentionally leaves the
    # guard unset on failure.
    monkeypatch.setattr(
        "gubbi_common.telemetry.otel.configure_otel",
        MagicMock(side_effect=RuntimeError("boom")),
    )
    monkeypatch.setattr(worker_module, "rebind_metrics_after_configure", MagicMock())

    from gubbi.config import get_settings

    get_settings.cache_clear()

    ctx: dict[str, Any] = {}
    # First startup: failure is swallowed (best-effort), flag stays False.
    await worker_module.startup(ctx)  # type: ignore[arg-type]
    assert worker_module._WORKER_TELEMETRY_CONFIGURED is False, (
        "a configure-time failure must leave the guard unset so a "
        "subsequent startup can retry the wiring; latching here would "
        "permanently skip telemetry for the rest of the process lifetime"
    )

    # Second startup: also fails (since the stub still raises), flag stays
    # False. The fact that startup() did not propagate the RuntimeError
    # implicitly verifies the swallow path; the flag staying False
    # verifies the latch was not set.
    await worker_module.startup(ctx)  # type: ignore[arg-type]
    assert (
        worker_module._WORKER_TELEMETRY_CONFIGURED is False
    ), "a second failed configure must still leave the guard unset"

    get_settings.cache_clear()
