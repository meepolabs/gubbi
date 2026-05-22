"""Unit tests for ``gubbi.telemetry.metrics`` (M4 e2e review).

Covers two M4 fixes that touch this module:

CRIT-5 / A1: pre-A1 the four instruments in ``gubbi/telemetry/metrics.py``
were built at module import, which runs before ``configure_otel()`` in the
gubbi lifespan. They bound to the NoOp meter provider and silently discarded
every ``.add(...)`` / ``.record(...)`` call -- which made the
``audit.persistence_failure`` alarm sensor the DEC-098 contract depends on
non-functional. These tests verify the post-A1 ``initialize_metrics()`` shape:

1. With a NoOp meter provider in place, ``initialize_metrics()`` returns
   NoOp instruments.
2. After swapping the provider and clearing the lru_cache,
   ``initialize_metrics()`` returns the real instruments built against
   the new provider -- proving the lifespan ordering is correct.
3. ``record_audit_persistence_failure`` smoke-call does not raise.

S8 H2 / A7: the filter delegates to
``gubbi_common.telemetry.allowlist.is_banned_key`` so it honours
``DERIVATIVE_MODIFIERS`` (safe suffixes like ``_hash``, ``_size``,
``_len``) and ``NEVER_EXEMPT_BASES`` (credential-shaped roots like
``password``, ``api_key``).
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
from gubbi.telemetry.metrics import _validate_metric_attrs

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


def test_record_replica_count_warning_smoke() -> None:
    """``record_replica_count_warning`` does not raise on a real provider."""
    gubbi_metrics.initialize_metrics.cache_clear()
    reader = InMemoryMetricReader()
    real_provider = MeterProvider(metric_readers=[reader])
    _set_provider(real_provider)

    try:
        gubbi_metrics.record_replica_count_warning(replica_count=2)
        gubbi_metrics.record_replica_count_warning(replica_count=3)
    finally:
        gubbi_metrics.initialize_metrics.cache_clear()


def test_record_replica_count_warning_is_noop_safe_when_instrument_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing instrument must degrade to a no-op, not raise.

    Unlike the other ``record_*`` helpers (direct ``inst[key]`` access),
    this helper guards with ``.get()`` because the Arq worker may call it
    before the OTel meter is wired. Simulate the missing-instrument shape
    by clearing the cache and monkeypatching ``initialize_metrics`` to
    return a dict WITHOUT the replica-count key -- the call must stay
    silent rather than KeyError out and crash worker boot.
    """
    gubbi_metrics.initialize_metrics.cache_clear()

    # ``record_replica_count_warning`` calls ``initialize_metrics()`` then
    # ``.get(...)``; an empty dict exercises the missing-instrument guard.
    # ``monkeypatch.setattr`` restores the original at teardown automatically;
    # the autouse ``_restore_otel_globals`` fixture then runs cache_clear()
    # on the restored real ``initialize_metrics``.
    monkeypatch.setattr(gubbi_metrics, "initialize_metrics", lambda: {})
    # Must not raise.
    gubbi_metrics.record_replica_count_warning(replica_count=2)


def test_record_replica_count_warning_increments_with_attributes() -> None:
    """Counter sum + attributes are exported correctly.

    Walks the ``InMemoryMetricReader``'s captured data and asserts:
      * the ``gateway.replica_count_warning`` counter exists,
      * total sum equals the number of record calls,
      * each data point carries the ``replica_count`` attribute the
        alerting rule splits on.

    The instrument name is REUSED from cloud-api on purpose so a single
    HyperDX rule aggregates across cloud-api + gubbi + worker; emitter
    distinction is via the ``service.name`` RESOURCE attribute (set at
    OTel init, the canonical cross-service identifier), not a metric
    attribute -- the helper deliberately does not emit a redundant
    ``service`` metric attribute (cloud-api's helper does not either,
    so a metric-attribute group-by rule would silently miss it).
    """
    gubbi_metrics.initialize_metrics.cache_clear()
    reader = InMemoryMetricReader()
    real_provider = MeterProvider(metric_readers=[reader])
    _set_provider(real_provider)

    try:
        gubbi_metrics.record_replica_count_warning(replica_count=2)
        gubbi_metrics.record_replica_count_warning(replica_count=4)

        metrics_data = reader.get_metrics_data()
        assert metrics_data is not None, "reader returned no metrics data"

        total = 0
        found = False
        seen_replica_counts: set[str] = set()
        seen_attribute_keys: set[str] = set()
        for resource_metrics in metrics_data.resource_metrics:
            for scope_metrics in resource_metrics.scope_metrics:
                for metric in scope_metrics.metrics:
                    if metric.name != MetricNames.REPLICA_COUNT_WARNING:
                        continue
                    found = True
                    for point in metric.data.data_points:
                        total += int(point.value)
                        seen_replica_counts.add(str(point.attributes.get("replica_count")))
                        seen_attribute_keys.update(point.attributes.keys())

        assert found, (
            f"counter {MetricNames.REPLICA_COUNT_WARNING!r} not found in exported "
            "metrics data -- helper bound to wrong instrument or failed to register"
        )
        assert total == 2, f"expected counter sum == 2 after two record calls, got {total}"
        # replica_count is stringified on the attribute (label cardinality).
        assert seen_replica_counts == {
            "2",
            "4",
        }, f"replica_count attribute must carry the value as a string; got {seen_replica_counts}"
        # No ``service`` metric attribute: emitters are distinguished by
        # the service.name RESOURCE attribute, matching cloud-api's helper.
        # A regression that re-introduced a service metric attribute would
        # silently break cross-service aggregation; pin it here.
        assert "service" not in seen_attribute_keys, (
            "record_replica_count_warning must NOT emit a 'service' metric "
            "attribute -- emitter distinction is via the service.name RESOURCE "
            "attribute set at OTel init, mirroring cloud-api's helper. "
            f"got attribute keys: {seen_attribute_keys}"
        )
    finally:
        gubbi_metrics.initialize_metrics.cache_clear()


def test_replica_count_warning_instrument_registered() -> None:
    """``initialize_metrics`` always registers the replica-count counter.

    Mirrors the contract the other instruments rely on: the key is always
    present after ``initialize_metrics()``, so the HTTP-path ``inst.get``
    lookup succeeds and the counter exports.
    """
    gubbi_metrics.initialize_metrics.cache_clear()
    reader = InMemoryMetricReader()
    real_provider = MeterProvider(metric_readers=[reader])
    _set_provider(real_provider)

    try:
        inst = gubbi_metrics.initialize_metrics()
        assert MetricNames.REPLICA_COUNT_WARNING in inst
        assert isinstance(inst[MetricNames.REPLICA_COUNT_WARNING], Counter)
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


def test_validate_metric_attrs_keeps_derivative_modifiers() -> None:
    """Derivative-suffixed keys (``_hash``, ``_size``, ``_len``) must survive.

    The pre-S8 H2 substring loop dropped these because it matched ``agent``,
    ``text`` and ``query`` as substrings of the safe suffix forms. The
    canonical ``is_banned_key`` consults ``DERIVATIVE_MODIFIERS`` so these
    keys pass through.
    """
    # Arrange
    attrs = {
        "user_agent_hash": "abc123",
        "text_hash": "def456",
        "query_size": "42",
        "text_len": "128",
        "tool.name": "journal_append_entry",
    }

    # Act
    cleaned = _validate_metric_attrs(attrs)

    # Assert
    assert cleaned == attrs


def test_validate_metric_attrs_drops_banned_keys() -> None:
    """Credential-shaped keys must be dropped even with a derivative suffix.

    ``NEVER_EXEMPT_BASES`` overrides ``DERIVATIVE_MODIFIERS`` so
    ``password_hash`` and ``api_key_hash`` are still banned even though they
    end in ``_hash``.
    """
    # Arrange
    attrs = {
        "password": "secret",
        "password_hash": "should-still-drop",
        "api_key": "sk-xxx",
        "tool.name": "safe",
    }

    # Act
    cleaned = _validate_metric_attrs(attrs)

    # Assert
    assert "password" not in cleaned
    assert "password_hash" not in cleaned
    assert "api_key" not in cleaned
    assert cleaned.get("tool.name") == "safe"


def test_validate_metric_attrs_drops_substring_content() -> None:
    """Keys containing banned substrings without derivative exemption drop.

    The banned substrings here are ``content`` (inside ``request_content``)
    and ``email`` (inside ``user_email``); the leading ``request_`` /
    ``user_`` prefixes are not what triggers the drop.
    """
    # Arrange
    attrs = {
        # "content" is the banned substring inside "request_content"
        "request_content": "raw body",
        # "email" is the banned substring inside "user_email"
        "user_email": "alice@example.com",
        "tool.name": "ok",
    }

    # Act
    cleaned = _validate_metric_attrs(attrs)

    # Assert
    assert "request_content" not in cleaned
    assert "user_email" not in cleaned
    assert cleaned.get("tool.name") == "ok"


def test_validate_metric_attrs_logs_warning_on_drop(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Operational signal: dropped keys must produce a WARNING log line."""
    # Arrange
    import logging

    caplog.set_level(logging.WARNING, logger="gubbi.telemetry.metrics")

    # Act
    _validate_metric_attrs({"password": "leak"})

    # Assert
    assert any(
        "password" in rec.getMessage() and rec.levelno == logging.WARNING for rec in caplog.records
    )


def test_validate_metric_attrs_empty_input() -> None:
    """Empty input maps to empty output without raising."""
    assert _validate_metric_attrs({}) == {}
