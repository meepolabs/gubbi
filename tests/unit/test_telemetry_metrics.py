"""Unit tests for ``gubbi.telemetry.metrics``.

Covers two fixes that touch this module:

Previously the four instruments in ``gubbi/telemetry/metrics.py``
were built at module import, which runs before ``configure_otel()`` in the
gubbi lifespan. They bound to the NoOp meter provider and silently discarded
every ``.add(...)`` / ``.record(...)`` call -- which made the
``audit.persistence_failure`` alarm sensor non-functional. These tests
verify the new ``initialize_metrics()`` shape:

1. With a NoOp meter provider in place, ``initialize_metrics()`` returns
   NoOp instruments.
2. After swapping the provider and clearing the lru_cache,
   ``initialize_metrics()`` returns the real instruments built against
   the new provider -- proving the lifespan ordering is correct.
3. ``record_audit_persistence_failure`` smoke-call does not raise.

The filter delegates to
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

    Each test in this module mutates the
    ``opentelemetry.metrics._internal._METER_PROVIDER`` and
    ``_METER_PROVIDER_SET_ONCE`` private slots via ``_set_provider``.
    Without an explicit restore, the last test's provider leaks into
    every subsequent test in the session (including tests in OTHER
    modules). That makes failures order-dependent and obscures the
    real owner of a leaked state.

    This autouse fixture snapshots both slots before the test runs and
    restores them in ``finally`` regardless of test outcome. It also
    clears the gubbi ``initialize_metrics`` lru_cache and the three
    orphan-counter factories so the next test starts from a clean
    cache state.
    """
    saved_provider = otel_metrics_internal._METER_PROVIDER  # type: ignore[attr-defined]
    saved_once = otel_metrics_internal._METER_PROVIDER_SET_ONCE  # type: ignore[attr-defined]
    try:
        yield
    finally:
        otel_metrics_internal._METER_PROVIDER = saved_provider  # type: ignore[attr-defined]
        otel_metrics_internal._METER_PROVIDER_SET_ONCE = saved_once  # type: ignore[attr-defined]
        gubbi_metrics.initialize_metrics.cache_clear()
        # clear the three orphan-counter factory caches so each test
        # observes a fresh meter resolution rather than the last test's
        # cached counter object.
        #
        # CONVENTION: this list MUST stay in sync with the rebind hook
        # at ``gubbi.telemetry.rebind_metrics_after_configure``. When
        # adding a new ``@lru_cache``-deferred metric factory, add a
        # ``cache_clear()`` here AND in the rebind hook. Drift between
        # the two means tests will leak state across the suite OR
        # production will leak state across lifespan re-runs.
        from gubbi.extraction.jobs.extract_conversation import (
            _get_extraction_refund_skipped_counter,
        )
        from gubbi.extraction.llm.anthropic_provider import _get_anthropic_retry_counter
        from gubbi.extraction.orphan_cleanup import _get_orphan_cleanup_swept_counter

        _get_orphan_cleanup_swept_counter.cache_clear()
        _get_extraction_refund_skipped_counter.cache_clear()
        _get_anthropic_retry_counter.cache_clear()


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

    This is the regression test. Before the fix the
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
        # Must not raise -- this is the alarm sensor.
        gubbi_metrics.record_audit_persistence_failure("test_event")
        gubbi_metrics.record_audit_persistence_failure("test_event")
    finally:
        gubbi_metrics.initialize_metrics.cache_clear()


def test_record_audit_persistence_failure_increments_counter() -> None:
    """Counter sum reaches 2 after two ``record_audit_persistence_failure`` calls.

    Regression test. The previous smoke test only
    confirmed the call did not raise -- a key-mismatch bug in
    ``record_audit_persistence_failure`` (wrong instrument name lookup,
    no attribute set, etc.) would still have silently passed because no
    test ever read the exported metric value. This test walks the
    ``InMemoryMetricReader``'s captured data and asserts the
    ``audit.persistence_failure`` counter sum equals the number of
    record calls. If the helper ever stops incrementing the counter,
    this fails loud -- which is what the alarm contract needs.
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

    The previous tests verified the
    ``initialize_metrics`` lru_cache shape directly by clearing it
    inline, but the production lifespan wire-up
    (``configure_otel -> rebind_metrics_after_configure``) had no
    regression coverage. A future refactor that deletes the
    ``cache_clear()`` + warm-call lines from the lifespan path would
    leave the alarm sensor permanently bound to NoOp without
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
    assert type(sealed).__name__.startswith("NoOp"), (
        "precondition: cache must be sealed against NoOp instrument"
    )

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
        "the alarm sensor would be dead in production"
    )


def test_validate_metric_attrs_keeps_derivative_modifiers() -> None:
    """Derivative-suffixed keys (``_hash``, ``_size``, ``_len``) must survive.

    The previous substring loop dropped these because it matched ``agent``,
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


# ---------------------------------------------------------------------------
# Orphan counter rebind regression tests
# ---------------------------------------------------------------------------
#
# Three counters in gubbi previously created their meter instruments at
# module import time, well before ``configure_otel`` ran during the FastAPI
# lifespan. They permanently bound to the NoOp meter provider and silently
# discarded every ``.add(...)`` -- the exact shape that the
# canonical ``initialize_metrics`` rebind already addressed for the
# audit/MCP/replica counters but had NOT been propagated to:
#
#   1. extraction_jobs.orphan_cleanup_swept_total
#   2. extraction.refund_skipped_total
#   3. anthropic.retry_count_total
#
# Each test below pins the contract: after ``configure_otel`` runs (modeled
# here by a ``_set_provider`` swap + ``rebind_metrics_after_configure``
# call), the counter is bound to the SDK provider and a ``.add(1)`` lands
# in the in-memory metric reader's exported data. A regression that drops
# any factory from the rebind hook -- or re-introduces a module-scope
# ``meter.create_counter`` import-time emit -- fails this test loud.


def _walk_counter_total(reader: InMemoryMetricReader, metric_name: str) -> tuple[bool, int]:
    """Walk reader's ResourceMetrics tree and sum data points for ``metric_name``.

    Returns ``(found, total)``. Mirrors the walk in
    ``test_record_audit_persistence_failure_increments_counter`` so the
    regression tests below stay close to the existing pattern.
    """
    metrics_data = reader.get_metrics_data()
    assert metrics_data is not None, "reader returned no metrics data"
    total = 0
    found = False
    for resource_metrics in metrics_data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                if metric.name != metric_name:
                    continue
                found = True
                for point in metric.data.data_points:
                    # Counter metrics export NumberDataPoint with .value;
                    # the union also includes Histogram* points (no .value).
                    # The metric_name filter above guarantees Counter shape.
                    total += int(point.value)  # type: ignore[union-attr]
    return found, total


def test_orphan_cleanup_swept_counter_rebinds_to_real_provider() -> None:
    """``extraction_jobs.orphan_cleanup_swept_total`` lands in real reader after rebind.

    The previous module-scope ``ORPHAN_CLEANUP_SWEPT = _meter.create_counter(...)``
    bound at import time and silently dropped every ``.add(...)`` because
    ``configure_otel`` had not run yet. Pin the rebind contract: after
    ``rebind_metrics_after_configure`` clears + re-primes the lru_cache
    factory, the counter binds to the SDK provider and the increment
    actually exports.
    """
    from gubbi.extraction.orphan_cleanup import _get_orphan_cleanup_swept_counter
    from gubbi.telemetry import rebind_metrics_after_configure

    # 1. Seal cache against NoOp provider (mirrors module import time).
    _set_provider(NoOpMeterProvider())
    _get_orphan_cleanup_swept_counter.cache_clear()
    _get_orphan_cleanup_swept_counter().add(1, attributes={"result": "swept", "state": "pending"})
    sealed = _get_orphan_cleanup_swept_counter()
    assert type(sealed).__name__.startswith("NoOp"), (
        "precondition: factory must be sealed against NoOp"
    )

    # 2. Configure a real SDK provider (mirror of what configure_otel does).
    reader = InMemoryMetricReader()
    real_provider = MeterProvider(metric_readers=[reader])
    _set_provider(real_provider)

    # 3. Re-bind via the production hook -- NOT cache_clear directly.
    rebind_metrics_after_configure()

    # 4. Increment now lands in the SDK reader.
    _get_orphan_cleanup_swept_counter().add(1, attributes={"result": "swept", "state": "running"})
    found, total = _walk_counter_total(reader, "extraction_jobs.orphan_cleanup_swept_total")

    assert found, (
        "extraction_jobs.orphan_cleanup_swept_total not found in exported metrics -- "
        "factory still bound to NoOp; rebind hook is not wiring this counter"
    )
    assert total == 1, f"expected counter sum == 1 after rebind+add, got {total}"
    # Verify it is a real SDK counter, not a NoOp shell.
    rebound = _get_orphan_cleanup_swept_counter()
    assert isinstance(rebound, Counter)
    assert not type(rebound).__name__.startswith("NoOp")
    # Cleanup runs in the autouse ``_restore_otel_globals`` fixture --
    # no need for a manual try/finally here.


def test_extraction_refund_skipped_counter_rebinds_to_real_provider() -> None:
    """``extraction.refund_skipped_total`` lands in real reader after rebind.

    Protects the refund-path observability for the worker. Without
    this, a refund-skipped event during a worker crash would be invisible
    at HyperDX even though the structured WARNING fires.
    """
    from gubbi.extraction.jobs.extract_conversation import (
        _get_extraction_refund_skipped_counter,
    )
    from gubbi.telemetry import rebind_metrics_after_configure

    _set_provider(NoOpMeterProvider())
    _get_extraction_refund_skipped_counter.cache_clear()
    _get_extraction_refund_skipped_counter().add(1, attributes={"reason": "unknown_period"})
    sealed = _get_extraction_refund_skipped_counter()
    assert type(sealed).__name__.startswith("NoOp"), (
        "precondition: factory must be sealed against NoOp"
    )

    reader = InMemoryMetricReader()
    real_provider = MeterProvider(metric_readers=[reader])
    _set_provider(real_provider)

    rebind_metrics_after_configure()

    _get_extraction_refund_skipped_counter().add(1, attributes={"reason": "unknown_period"})
    found, total = _walk_counter_total(reader, "extraction.refund_skipped_total")

    assert found, (
        "extraction.refund_skipped_total not found in exported metrics -- "
        "factory still bound to NoOp; rebind hook is not wiring this counter"
    )
    assert total == 1, f"expected counter sum == 1 after rebind+add, got {total}"
    rebound = _get_extraction_refund_skipped_counter()
    assert isinstance(rebound, Counter)
    assert not type(rebound).__name__.startswith("NoOp")
    # Cleanup runs in the autouse ``_restore_otel_globals`` fixture.


def test_anthropic_retry_counter_rebinds_to_real_provider() -> None:
    """``anthropic.retry_count_total`` lands in real reader after rebind.

    Most beta-relevant of the three -- LLM retry storms are invisible
    at HyperDX without this rebind. The Anthropic provider is on the hot
    path of every extraction job; a rate-limit cascade would drive the
    counter but the alarm rule would never fire.
    """
    from gubbi.extraction.llm.anthropic_provider import _get_anthropic_retry_counter
    from gubbi.telemetry import rebind_metrics_after_configure

    _set_provider(NoOpMeterProvider())
    _get_anthropic_retry_counter.cache_clear()
    _get_anthropic_retry_counter().add(
        1, attributes={"result": "retried", "error_class": "LLMTransientError"}
    )
    sealed = _get_anthropic_retry_counter()
    assert type(sealed).__name__.startswith("NoOp"), (
        "precondition: factory must be sealed against NoOp"
    )

    reader = InMemoryMetricReader()
    real_provider = MeterProvider(metric_readers=[reader])
    _set_provider(real_provider)

    rebind_metrics_after_configure()

    _get_anthropic_retry_counter().add(
        1, attributes={"result": "exhausted", "error_class": "LLMTransientError"}
    )
    found, total = _walk_counter_total(reader, "anthropic.retry_count_total")

    assert found, (
        "anthropic.retry_count_total not found in exported metrics -- "
        "factory still bound to NoOp; rebind hook is not wiring this counter"
    )
    assert total == 1, f"expected counter sum == 1 after rebind+add, got {total}"
    rebound = _get_anthropic_retry_counter()
    assert isinstance(rebound, Counter)
    assert not type(rebound).__name__.startswith("NoOp")
    # Cleanup runs in the autouse ``_restore_otel_globals`` fixture.


def test_orphan_counter_modules_have_no_module_scope_counter_names() -> None:
    """Pin: orphan-counter modules expose factories, NOT module-scope counters.

    A regression that re-introduces ``COUNTER_NAME = _meter.create_counter(...)``
    at module scope -- the original bug shape -- would leave the factory
    rebind path silently parallel to a stale module-scope reference: any
    caller that imports the old name binds to the import-time NoOp meter
    permanently. The `_get_*_counter()` factory tests above only catch
    the factory-path regression; this test catches the module-scope-shape
    regression.

    Add new entries here when adding a new orphan-counter factory.
    """
    from gubbi.extraction import orphan_cleanup
    from gubbi.extraction.jobs import extract_conversation
    from gubbi.extraction.llm import anthropic_provider

    assert not hasattr(orphan_cleanup, "ORPHAN_CLEANUP_SWEPT"), (
        "orphan_cleanup.ORPHAN_CLEANUP_SWEPT was re-introduced as a "
        "module-scope counter; it must live behind "
        "_get_orphan_cleanup_swept_counter() so the rebind hook can re-prime it"
    )
    assert not hasattr(extract_conversation, "EXTRACTION_REFUND_SKIPPED"), (
        "extract_conversation.EXTRACTION_REFUND_SKIPPED was re-introduced as a "
        "module-scope counter; it must live behind "
        "_get_extraction_refund_skipped_counter()"
    )
    assert not hasattr(anthropic_provider, "ANTHROPIC_RETRY_COUNT"), (
        "anthropic_provider.ANTHROPIC_RETRY_COUNT was re-introduced as a "
        "module-scope counter; it must live behind "
        "_get_anthropic_retry_counter()"
    )
