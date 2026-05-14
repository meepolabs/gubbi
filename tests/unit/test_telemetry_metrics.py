"""Unit tests for ``gubbi.telemetry.metrics`` (CRIT-5 / M4 e2e review).

Pre-A1 the four instruments in ``gubbi/telemetry/metrics.py`` were built
at module import, which runs before ``configure_otel()`` in the gubbi
lifespan. They bound to the NoOp meter provider and silently discarded
every ``.add(...)`` / ``.record(...)`` call -- which made the
``audit.persistence_failure`` alarm sensor the DEC-098 contract depends
on non-functional.

These tests verify the post-A1 ``initialize_metrics()`` shape:

1. With a NoOp meter provider in place, ``initialize_metrics()`` returns
   NoOp instruments.
2. After swapping the provider and clearing the lru_cache,
   ``initialize_metrics()`` returns the real instruments built against
   the new provider -- proving the lifespan ordering is correct.
3. ``record_audit_persistence_failure`` smoke-call does not raise.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from opentelemetry.metrics import Counter, NoOpMeterProvider, set_meter_provider
from opentelemetry.metrics import _internal as otel_metrics_internal
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from gubbi.telemetry import metrics as gubbi_metrics
from gubbi.telemetry.attrs import MetricNames

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _restore_otel_globals() -> Iterator[None]:
    """Snapshot + restore the OTel private meter-provider globals.

    H-1 (R2 fix-pass 2026-05-13): each test in this module mutates the
    ``opentelemetry.metrics._internal._METER_PROVIDER`` and
    ``_METER_PROVIDER_SET_ONCE`` private slots via ``_set_provider``.
    Without an explicit restore, the last test's provider leaks into
    every subsequent test in the session (including tests in OTHER
    modules). That makes failures order-dependent and obscures the
    real owner of a leaked state.

    This autouse fixture snapshots both slots before the test runs and
    restores them in ``finally`` regardless of test outcome. It also
    clears the gubbi ``initialize_metrics`` lru_cache so the next test
    starts from a clean cache state.
    """
    saved_provider = otel_metrics_internal._METER_PROVIDER  # type: ignore[attr-defined]
    saved_once = otel_metrics_internal._METER_PROVIDER_SET_ONCE  # type: ignore[attr-defined]
    try:
        yield
    finally:
        otel_metrics_internal._METER_PROVIDER = saved_provider  # type: ignore[attr-defined]
        otel_metrics_internal._METER_PROVIDER_SET_ONCE = saved_once  # type: ignore[attr-defined]
        gubbi_metrics.initialize_metrics.cache_clear()


def _set_provider(provider: object) -> None:
    """Force ``opentelemetry`` to use ``provider`` for new ``get_meter`` calls.

    ``set_meter_provider`` is one-shot per process in the SDK -- it is
    guarded by a ``Once`` latch. To enable test isolation we reset the
    latch (private slot in ``opentelemetry.metrics._internal``) before
    each rebind. The proxy meter provider's reference to the previous
    real provider is also cleared so that get_meter resolves freshly.

    The ``_restore_otel_globals`` autouse fixture above puts these slots
    back to their pre-test values, so per-test mutation here is safe.
    """
    otel_metrics_internal._METER_PROVIDER_SET_ONCE = (  # type: ignore[attr-defined]
        otel_metrics_internal.Once()  # type: ignore[attr-defined]
    )
    otel_metrics_internal._METER_PROVIDER = None  # type: ignore[attr-defined]
    set_meter_provider(provider)  # type: ignore[arg-type]


def test_initialize_metrics_returns_noop_when_provider_is_noop() -> None:
    """NoOp meter provider -> NoOp counter returned for audit.persistence_failure."""
    gubbi_metrics.initialize_metrics.cache_clear()
    _set_provider(NoOpMeterProvider())

    try:
        inst = gubbi_metrics.initialize_metrics()
        counter = inst[MetricNames.AUDIT_PERSISTENCE_FAILURE]

        # NoOp counters live in the ``opentelemetry.metrics._internal``
        # tree; checking the class name keeps the test resilient to
        # cross-version path tweaks.
        assert type(counter).__name__.startswith("NoOp")
    finally:
        gubbi_metrics.initialize_metrics.cache_clear()


def test_initialize_metrics_rebinds_to_real_provider_after_cache_clear() -> None:
    """Lifespan ordering: cache_clear + real provider -> real instruments.

    This is the regression test for CRIT-5. Before the fix the
    instruments were captured at module import (NoOp), and a later
    ``set_meter_provider`` call had no effect. After the fix the first
    ``initialize_metrics()`` call post-clear binds to whatever provider
    is current at that moment -- which in production is the SDK
    provider configured during ``configure_otel()``.
    """
    gubbi_metrics.initialize_metrics.cache_clear()
    _set_provider(NoOpMeterProvider())
    inst_noop = gubbi_metrics.initialize_metrics()
    assert type(inst_noop[MetricNames.AUDIT_PERSISTENCE_FAILURE]).__name__.startswith("NoOp")

    # Now: configure a real SDK provider and rebind. This mirrors the
    # gubbi lifespan flow where ``configure_otel()`` runs after the
    # module-level imports have already happened.
    reader = InMemoryMetricReader()
    real_provider = MeterProvider(metric_readers=[reader])
    _set_provider(real_provider)
    gubbi_metrics.initialize_metrics.cache_clear()

    try:
        inst_real = gubbi_metrics.initialize_metrics()
        counter = inst_real[MetricNames.AUDIT_PERSISTENCE_FAILURE]

        # Real SDK counter implements the Counter Protocol and is NOT a
        # NoOp shell.
        assert isinstance(counter, Counter)
        assert not type(counter).__name__.startswith("NoOp")
    finally:
        gubbi_metrics.initialize_metrics.cache_clear()


def test_record_audit_persistence_failure_smoke() -> None:
    """``record_audit_persistence_failure`` does not raise on a real provider."""
    gubbi_metrics.initialize_metrics.cache_clear()
    reader = InMemoryMetricReader()
    real_provider = MeterProvider(metric_readers=[reader])
    _set_provider(real_provider)

    try:
        # Must not raise -- this is the DEC-098 alarm sensor.
        gubbi_metrics.record_audit_persistence_failure("test_event")
        gubbi_metrics.record_audit_persistence_failure("test_event")
    finally:
        gubbi_metrics.initialize_metrics.cache_clear()


def test_record_audit_persistence_failure_increments_counter() -> None:
    """Counter sum reaches 2 after two ``record_audit_persistence_failure`` calls.

    Regression for H-2 (M4 e2e review). The previous smoke test only
    confirmed the call did not raise -- a key-mismatch bug in
    ``record_audit_persistence_failure`` (wrong instrument name lookup,
    no attribute set, etc.) would still have silently passed because no
    test ever read the exported metric value. This test walks the
    ``InMemoryMetricReader``'s captured data and asserts the
    ``audit.persistence_failure`` counter sum equals the number of
    record calls. If the helper ever stops incrementing the counter,
    this fails loud -- which is what the DEC-098 alarm contract needs.
    """
    gubbi_metrics.initialize_metrics.cache_clear()
    reader = InMemoryMetricReader()
    real_provider = MeterProvider(metric_readers=[reader])
    _set_provider(real_provider)

    try:
        gubbi_metrics.record_audit_persistence_failure("test_event")
        gubbi_metrics.record_audit_persistence_failure("test_event")

        metrics_data = reader.get_metrics_data()
        assert metrics_data is not None, "reader returned no metrics data"

        # Walk: ResourceMetrics -> ScopeMetrics -> Metric -> data points.
        total = 0
        found = False
        for resource_metrics in metrics_data.resource_metrics:
            for scope_metrics in resource_metrics.scope_metrics:
                for metric in scope_metrics.metrics:
                    if metric.name != MetricNames.AUDIT_PERSISTENCE_FAILURE:
                        continue
                    found = True
                    for point in metric.data.data_points:
                        total += int(point.value)

        assert found, (
            f"counter {MetricNames.AUDIT_PERSISTENCE_FAILURE!r} not found in "
            "exported metrics data -- helper bound to wrong instrument or "
            "failed to register against the configured provider"
        )
        assert total == 2, f"expected counter sum == 2 after two record calls, got {total}"
    finally:
        gubbi_metrics.initialize_metrics.cache_clear()


def test_record_tool_call_and_response_size_smoke() -> None:
    """Convenience helpers do not raise on a real provider."""
    gubbi_metrics.initialize_metrics.cache_clear()
    reader = InMemoryMetricReader()
    real_provider = MeterProvider(metric_readers=[reader])
    _set_provider(real_provider)

    try:
        gubbi_metrics.record_tool_call(12.5, tool_name="journal_append_entry")
        gubbi_metrics.record_tool_response_size(1024, tool_name="journal_append_entry")
    finally:
        gubbi_metrics.initialize_metrics.cache_clear()


def test_rebind_metrics_after_configure_rebinds_lifespan_cache() -> None:
    """``rebind_metrics_after_configure`` rebinds NoOp -> real provider.

    M-1 (R2 fix-pass 2026-05-13): the previous tests verified the
    ``initialize_metrics`` lru_cache shape directly by clearing it
    inline, but the production lifespan wire-up
    (``configure_otel -> rebind_metrics_after_configure``) had no
    regression coverage. A future refactor that deletes the
    ``cache_clear()`` + warm-call lines from the lifespan path would
    leave the DEC-098 alarm sensor permanently bound to NoOp without
    a single failing test.

    This test pins the contract by:

      1. Binding the lru_cache against a NoOp provider via a real
         ``record_audit_persistence_failure`` call.
      2. Swapping the active meter provider to a real SDK provider.
      3. Calling ``rebind_metrics_after_configure`` (the helper that
         the lifespan wire-up invokes) -- NOT clearing the lru_cache
         manually.
      4. Asserting the cached instrument is no longer NoOp.
    """
    # 1. Seal cache against NoOp provider via the real call path.
    _set_provider(NoOpMeterProvider())
    gubbi_metrics.initialize_metrics.cache_clear()
    gubbi_metrics.record_audit_persistence_failure("test_event")
    sealed = gubbi_metrics.initialize_metrics()[MetricNames.AUDIT_PERSISTENCE_FAILURE]
    assert type(sealed).__name__.startswith(
        "NoOp"
    ), "precondition: cache must be sealed against NoOp instrument"

    # 2. Configure a real SDK provider (mirror of what configure_otel does).
    reader = InMemoryMetricReader()
    real_provider = MeterProvider(metric_readers=[reader])
    _set_provider(real_provider)

    # 3. Invoke the production rebind helper -- NOT initialize_metrics.cache_clear() directly.
    from gubbi.telemetry import rebind_metrics_after_configure

    rebind_metrics_after_configure()

    # 4. Cache is now bound to the real provider's instruments.
    rebound = gubbi_metrics.initialize_metrics()[MetricNames.AUDIT_PERSISTENCE_FAILURE]
    assert isinstance(rebound, Counter)
    assert not type(rebound).__name__.startswith("NoOp"), (
        "rebind_metrics_after_configure did not re-bind the lru_cache; "
        "the DEC-098 alarm sensor would be dead in production"
    )
