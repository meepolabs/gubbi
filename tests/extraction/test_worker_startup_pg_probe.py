"""Worker startup wires ``probe_pg_log_settings`` after the PG pool opens.

The Arq worker hits the same encrypted INSERT path as the HTTP API
(via ``extract_conversation``), so the probe runs in both surfaces.
This test stubs every heavy dependency the ``startup`` hook touches and
asserts the probe is invoked exactly once with the worker's pool and
the env-resolved ``mode`` keyword.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import structlog
from gubbi_common.bootstrap.pg_log_probe import PgLogProbeError

from gubbi.extraction import worker as worker_module

# Stub app-pool max size returned by ``pool.get_max_size()``. The worker's
# #138 WARNING reads the LIVE pool max via ``get_max_size()`` (not a
# constant), so the stub pool must answer with a real int. Kept as a local
# test constant so the assertion pins the live-read contract rather than a
# particular production-configured size.
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
    """Stub out everything the worker startup hook would normally hit.

    Critically, the ``threading.Thread`` constructor used by the worker
    is patched on the **module's** ``threading`` reference -- NOT on the
    stdlib ``threading`` module -- so global Thread creation (asyncio
    executors, etc.) stays untouched. The health-server thread the
    worker creates is replaced with a no-op stand-in whose ``start`` is
    a sync ``MagicMock``.
    """
    pool = MagicMock()
    pool.close = AsyncMock()
    # The #138 WARNING reads the live per-pod pool max via get_max_size();
    # a bare MagicMock would return a MagicMock (not an int) and break the
    # arithmetic. Answer with a real int.
    pool.get_max_size = MagicMock(return_value=_STUB_APP_POOL_MAX)

    monkeypatch.setattr(worker_module, "init_pool", AsyncMock(return_value=pool))

    redis_pool_stub = MagicMock()
    redis_pool_stub.aclose = AsyncMock()
    redis_client_stub = MagicMock()
    redis_client_stub.aclose = AsyncMock()
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

    return {
        "pool": pool,
        "redis_client": redis_client_stub,
        "redis_pool": redis_pool_stub,
    }


@pytest.mark.unit
async def test_worker_startup_calls_pg_log_probe_with_default_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``startup`` invokes ``probe_pg_log_settings`` with the worker pool + strict mode."""
    monkeypatch.delenv("JOURNAL_PG_LOG_PROBE_MODE", raising=False)
    handles = _patch_worker_dependencies(monkeypatch)

    probe_mock = AsyncMock(return_value=None)
    monkeypatch.setattr(worker_module, "probe_pg_log_settings", probe_mock)

    ctx: dict[str, Any] = {}
    await worker_module.startup(ctx)  # type: ignore[arg-type]

    probe_mock.assert_awaited_once()
    args, kwargs = probe_mock.call_args
    assert args == (handles["pool"],)
    assert kwargs.get("mode") == "strict"


@pytest.mark.unit
async def test_worker_startup_forwards_warn_mode_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``JOURNAL_PG_LOG_PROBE_MODE=warn`` is forwarded to the probe."""
    monkeypatch.setenv("JOURNAL_PG_LOG_PROBE_MODE", "warn")
    _patch_worker_dependencies(monkeypatch)

    probe_mock = AsyncMock(return_value=None)
    monkeypatch.setattr(worker_module, "probe_pg_log_settings", probe_mock)

    ctx: dict[str, Any] = {}
    await worker_module.startup(ctx)  # type: ignore[arg-type]

    _, kwargs = probe_mock.call_args
    assert kwargs.get("mode") == "warn"


@pytest.mark.unit
async def test_worker_startup_aborts_when_pg_probe_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the probe raises, the worker pool is closed and the error propagates."""
    handles = _patch_worker_dependencies(monkeypatch)

    probe_mock = AsyncMock(side_effect=PgLogProbeError("unsafe log_statement=all"))
    monkeypatch.setattr(worker_module, "probe_pg_log_settings", probe_mock)

    ctx: dict[str, Any] = {}
    with pytest.raises(PgLogProbeError):
        await worker_module.startup(ctx)  # type: ignore[arg-type]

    handles["pool"].close.assert_awaited_once()


@pytest.mark.unit
def test_worker_module_exposes_probe_pg_log_settings_symbol() -> None:
    """``probe_pg_log_settings`` is bound on ``gubbi.extraction.worker`` for monkeypatch."""
    assert hasattr(worker_module, "probe_pg_log_settings"), (
        "gubbi.extraction.worker must import probe_pg_log_settings at module "
        "scope so test monkeypatch can intercept the call."
    )


# ---------------------------------------------------------------------------
# Replica-count over-provisioning WARNING + alertable counter (#138).
# ---------------------------------------------------------------------------


def _patch_worker_for_replica_tests(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Stub the worker startup deps + return the patched counter helper mock.

    Stubs the pg-log probe (no real GUCs), the OTel meter wiring (no real
    SDK in a unit test), and the ``record_replica_count_warning`` counter
    helper so the alertable signal can be asserted without a live meter.
    """
    _patch_worker_dependencies(monkeypatch)
    monkeypatch.setattr(worker_module, "probe_pg_log_settings", AsyncMock(return_value=None))
    # The worker wires the OTel meter at startup; in a unit test we stub it
    # so no real exporter/provider is installed. The counter helper itself
    # is patched separately so the alertable contract is observable.
    monkeypatch.setattr(worker_module, "_configure_worker_telemetry", MagicMock())
    record_mock = MagicMock()
    monkeypatch.setattr(worker_module, "record_replica_count_warning", record_mock)
    return record_mock


@pytest.mark.unit
async def test_worker_startup_warns_and_increments_counter_when_replicas_gt_1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """JOURNAL_REPLICA_COUNT=2 -> WARNING + counter increment.

    The worker is fixed at a single replica by deploy convention, so a
    replica count > 1 multiplies the per-pod extraction-budget logic + DB
    pool N-fold AND violates that single-worker policy. Both must surface.
    The emitter is distinguished by the ``service.name`` RESOURCE
    attribute set at OTel init (here ``"gubbi-extraction-worker"`` from
    ``_configure_worker_telemetry``), not by a metric attribute -- the
    helper takes only ``replica_count``.
    """
    record_mock = _patch_worker_for_replica_tests(monkeypatch)
    monkeypatch.setenv("JOURNAL_REPLICA_COUNT", "2")

    ctx: dict[str, Any] = {}
    with structlog.testing.capture_logs() as logs:
        await worker_module.startup(ctx)  # type: ignore[arg-type]

    record_mock.assert_called_once_with(replica_count=2)

    warnings = [
        log
        for log in logs
        if log.get("event") == "extraction_worker_replica_policy_violation"
        and log.get("log_level") == "warning"
    ]
    assert len(warnings) == 1, f"expected one over-provisioning WARNING, got {warnings}"
    emitted = warnings[0]
    assert emitted["replica_count"] == 2
    # Worker opens ONLY the app pool (admin pool not opened), so the per-pod
    # footprint is the live app pool max (read via get_max_size(), here the
    # stub's _STUB_APP_POOL_MAX).
    assert emitted["db_pool_max_per_pod"] == _STUB_APP_POOL_MAX
    assert emitted["effective_db_connections"] == _STUB_APP_POOL_MAX * 2
    # The note must flag the single-worker deploy-policy violation.
    assert "single" in emitted["note"].lower()


@pytest.mark.unit
async def test_worker_startup_silent_at_default_replica_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default (unset) JOURNAL_REPLICA_COUNT=1 -> no WARNING, no counter."""
    record_mock = _patch_worker_for_replica_tests(monkeypatch)
    monkeypatch.delenv("JOURNAL_REPLICA_COUNT", raising=False)

    ctx: dict[str, Any] = {}
    with structlog.testing.capture_logs() as logs:
        await worker_module.startup(ctx)  # type: ignore[arg-type]

    record_mock.assert_not_called()
    assert not [
        log for log in logs if log.get("event") == "extraction_worker_replica_policy_violation"
    ], "default single-worker deploy must stay silent"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("value", "match"),
    [
        ("abc", "not an integer"),
        ("0", ">= 1"),
        ("-2", ">= 1"),
    ],
)
async def test_worker_startup_rejects_invalid_replica_count(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
    match: str,
) -> None:
    """A non-integer or < 1 JOURNAL_REPLICA_COUNT aborts worker startup."""
    _patch_worker_for_replica_tests(monkeypatch)
    monkeypatch.setenv("JOURNAL_REPLICA_COUNT", value)

    ctx: dict[str, Any] = {}
    with pytest.raises(RuntimeError, match=match):
        await worker_module.startup(ctx)  # type: ignore[arg-type]


@pytest.mark.unit
def test_worker_validate_replica_count_helper_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unit-cover the worker's ``_validate_replica_count`` in isolation."""
    monkeypatch.delenv("JOURNAL_REPLICA_COUNT", raising=False)
    assert worker_module._validate_replica_count() == 1

    monkeypatch.setenv("JOURNAL_REPLICA_COUNT", "4")
    assert worker_module._validate_replica_count() == 4

    monkeypatch.setenv("JOURNAL_REPLICA_COUNT", "nope")
    with pytest.raises(RuntimeError, match="not an integer"):
        worker_module._validate_replica_count()

    monkeypatch.setenv("JOURNAL_REPLICA_COUNT", "0")
    with pytest.raises(RuntimeError, match=">= 1"):
        worker_module._validate_replica_count()


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
    monkeypatch.setattr(worker_module, "probe_pg_log_settings", AsyncMock(return_value=None))
    monkeypatch.setattr(worker_module, "record_replica_count_warning", MagicMock())
    monkeypatch.delenv("JOURNAL_REPLICA_COUNT", raising=False)

    # Stub the SDK init the wiring would otherwise run -- no real exporter
    # threads or providers in the test process. The test does NOT assert on
    # the call count of these stubs; the flag transitions are the contract.
    monkeypatch.setattr("gubbi_common.telemetry.otel.configure_otel", MagicMock())
    monkeypatch.setattr(worker_module, "initialize_metrics", MagicMock())

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
    monkeypatch.setattr(worker_module, "probe_pg_log_settings", AsyncMock(return_value=None))
    monkeypatch.setattr(worker_module, "record_replica_count_warning", MagicMock())
    monkeypatch.delenv("JOURNAL_REPLICA_COUNT", raising=False)

    # Stub the SDK init to raise -- exercises the swallow-and-log branch
    # inside _configure_worker_telemetry that intentionally leaves the
    # guard unset on failure.
    monkeypatch.setattr(
        "gubbi_common.telemetry.otel.configure_otel",
        MagicMock(side_effect=RuntimeError("boom")),
    )
    monkeypatch.setattr(worker_module, "initialize_metrics", MagicMock())

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
