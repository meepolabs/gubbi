"""Metric instruments for gubbi.

Defines and initializes the gubbi-side metric instruments:
    - mcp.tool_call_duration (histogram)
    - mcp.tool_call_count (counter)
    - mcp.tool_response_size_chars (histogram)
    - audit.persistence_failure (counter)

The instruments are built lazily through ``initialize_metrics()`` so that
they bind to the meter provider configured by ``configure_otel()`` during
``main.py`` lifespan, not the NoOp provider that exists at module import
time. This mirrors the cloud pattern at
``gubbi-cloud/gubbi_cloud/telemetry/metrics.py``.

Prior to this refactor the four
instruments were created at module import, which runs before
``configure_otel()`` -- they bound to the NoOp provider and silently
discarded every ``.add(...)`` / ``.record(...)``. That made the
``audit.persistence_failure`` alarm contract non-functional. The fix is
to defer instrument creation to
``initialize_metrics()`` and look up the cached instruments on every
record call.

Usage::

    from gubbi.telemetry.metrics import record_tool_call

    record_tool_call(latency_ms, tool_name="journal_append_entry")

For direct instrument access (rare):

    from gubbi.telemetry.metrics import initialize_metrics

    initialize_metrics()["mcp.tool_call_count"].add(1, {"tool.name": name})
"""

from __future__ import annotations

import functools
import logging
from typing import cast

from gubbi_common.telemetry.allowlist import is_banned_key
from opentelemetry.metrics import Counter, Histogram, get_meter

from gubbi.telemetry.attrs import MetricNames

__all__: list[str] = [
    "initialize_metrics",
    "record_audit_persistence_failure",
    "record_replica_count_warning",
    "record_startup_probe_outcome",
    "record_tool_call",
    "record_tool_response_size",
]

logger = logging.getLogger(__name__)

_METER_NAME = "gubbi"


@functools.lru_cache
def initialize_metrics() -> dict[str, object]:
    """Build and cache all gubbi metric instruments.

    Returns a dict keyed by instrument name. Subsequent calls return the
    same cached dict. Call ``initialize_metrics.cache_clear()`` in tests
    that need to re-bind to a freshly configured meter provider.
    """
    meter = get_meter(_METER_NAME)

    inst: dict[str, object] = {}
    inst[MetricNames.MCP_TOOL_CALL_DURATION] = meter.create_histogram(
        name=MetricNames.MCP_TOOL_CALL_DURATION,
        description="Per-tool call latency in milliseconds",
        unit="ms",
    )
    inst[MetricNames.MCP_TOOL_CALL_COUNT] = meter.create_counter(
        name=MetricNames.MCP_TOOL_CALL_COUNT,
        description="Per-tool invocation count",
        unit="1",
    )
    inst[MetricNames.MCP_TOOL_RESPONSE_SIZE_CHARS] = meter.create_histogram(
        name=MetricNames.MCP_TOOL_RESPONSE_SIZE_CHARS,
        description="Per-tool response size in characters",
        unit="chars",
    )
    inst[MetricNames.AUDIT_PERSISTENCE_FAILURE] = meter.create_counter(
        name=MetricNames.AUDIT_PERSISTENCE_FAILURE,
        description="Audit log persistence failure count by event type",
        unit="1",
    )
    inst[MetricNames.REPLICA_COUNT_WARNING] = meter.create_counter(
        name=MetricNames.REPLICA_COUNT_WARNING,
        description=(
            "Incremented once at lifespan/worker startup when "
            "JOURNAL_REPLICA_COUNT > 1. gubbi's per-pod resources (DB "
            "connection pool; worker extraction-budget logic) are "
            "in-process, so the effective per-pod cap becomes cap * "
            "REPLICA_COUNT. Counter is the alertable counterpart to the "
            "structured WARNING log events (db_pool_over_provisioned for "
            "gubbi; extraction_worker_replica_policy_violation for the "
            "worker). Emitters are distinguished by the service.name "
            "RESOURCE attribute set at OTel init (the canonical OTel "
            "cross-service identifier), not by a metric attribute."
        ),
        unit="1",
    )
    inst[MetricNames.STARTUP_PROBE_OUTCOME] = meter.create_counter(
        name=MetricNames.STARTUP_PROBE_OUTCOME,
        description=(
            "Incremented once per startup probe invocation by "
            "StartupRunner via the OutcomeCounterCallback hook. Carries "
            "{name, outcome, app_env} attributes; emitter split via "
            "service.name. Cardinality bounded by the small product of "
            "probe-set x {ok, warn, fail} x {dev, ci, staging, "
            "production}."
        ),
        unit="1",
    )
    return inst


def _validate_metric_attrs(attrs: dict[str, str]) -> dict[str, str]:
    """Strip banned keys from metric attributes.

    Delegates to :func:`gubbi_common.telemetry.allowlist.is_banned_key`, the
    canonical single source of truth used by ``safe_set_attributes`` on the
    span path. This honours ``DERIVATIVE_MODIFIERS`` (so keys like
    ``user_agent_hash`` or ``text_hash`` survive) and ``NEVER_EXEMPT_BASES``
    (so credential-shaped keys like ``password_hash`` are still dropped).
    """
    cleaned: dict[str, str] = {}
    for key, value in attrs.items():
        if is_banned_key(key):
            logger.warning("Dropping banned metric attribute %r", key)
            continue
        cleaned[key] = value
    return cleaned


def record_tool_call(latency_ms: float, tool_name: str) -> None:
    """Record a tool call duration and increment the call counter.

    Convenience function that records both histogram and counter with
    the tool name attribute. Safe to call when metrics are disabled
    (instrument lookup will still return a NoOp instrument).

    Note: ``initialize_metrics()`` always populates all four keys.
    Direct ``inst[key]`` access lets a missing key surface as a
    KeyError -- a None-guard would silently hide that regression.
    """
    attrs = _validate_metric_attrs({"tool.name": tool_name})
    inst = initialize_metrics()
    cast(Histogram, inst[MetricNames.MCP_TOOL_CALL_DURATION]).record(latency_ms, attributes=attrs)
    cast(Counter, inst[MetricNames.MCP_TOOL_CALL_COUNT]).add(1, attributes=attrs)


def record_tool_response_size(size_chars: int, tool_name: str) -> None:
    """Record a tool response size in the histogram.

    Direct ``inst[key]`` access: see ``record_tool_call`` docstring.
    """
    attrs = _validate_metric_attrs({"tool.name": tool_name})
    inst = initialize_metrics()
    cast(Histogram, inst[MetricNames.MCP_TOOL_RESPONSE_SIZE_CHARS]).record(
        size_chars, attributes=attrs
    )


def record_audit_persistence_failure(event_type: str) -> None:
    """Increment the audit persistence failure counter.

    This is the sensor for the fail-open compensating
    alarm. Must bind to the real meter provider (post-lifespan) for the
    HyperDX alarm contract to fire. ``initialize_metrics()`` is
    lru_cached and primed at the tail of ``configure_otel()`` so the
    cached instruments are bound to the configured provider.

    Direct ``inst[key]`` access: a missing key here would silently
    disable the alarm sensor. Let KeyError surface.
    """
    attrs = _validate_metric_attrs({"event_type": event_type})
    inst = initialize_metrics()
    cast(Counter, inst[MetricNames.AUDIT_PERSISTENCE_FAILURE]).add(1, attributes=attrs)


def record_replica_count_warning(*, replica_count: int) -> None:
    """Increment the gateway.replica_count_warning counter at startup.

    Emitted once per process startup when
    ``JOURNAL_REPLICA_COUNT > 1`` -- from the FastAPI lifespan and from
    the Arq extraction worker's startup hook. Pairs with the structured
    ``db_pool_over_provisioned`` (gubbi) /
    ``extraction_worker_replica_policy_violation`` (worker) WARNING so
    HyperDX (or any backend reading the OTel metric) has an alertable
    signal even when log-stream sampling drops the WARNING line.

    Emitter distinction is via the ``service.name`` RESOURCE attribute
    (gubbi-common's ``configure_otel`` sets it from its ``service_name``
    arg: ``"gubbi"`` for the HTTP service, ``"gubbi-extraction-worker"``
    for the Arq worker). That is the canonical OTel cross-service
    identifier and is what alert rules in HyperDX should split / group
    by; this helper deliberately does NOT emit a redundant ``service``
    METRIC attribute (cloud-api's helper has the same shape, so a
    cross-service alert rule that grouped by metric attribute would
    silently miss cloud-api).

    Unlike the other ``record_*`` helpers in this module, this one guards
    against a missing instrument with ``.get()`` rather than letting a
    KeyError surface: the worker does not run the FastAPI lifespan and may
    not have wired the OTel meter when this fires, so a missing instrument
    must degrade to a no-op startup-time signal -- never crash worker
    boot. ``initialize_metrics()`` always populates the key, so on the
    HTTP path the lookup succeeds; on an unwired worker meter the counter
    is a NoOp and the ``.add`` is silently dropped (the paired WARNING log
    still fires). Label cardinality is bounded by the small integer range
    of plausible replica counts.
    """
    inst = initialize_metrics()
    raw = inst.get(MetricNames.REPLICA_COUNT_WARNING)
    if raw is not None:
        cast(Counter, raw).add(1, {"replica_count": str(replica_count)})


def record_startup_probe_outcome(*, name: str, outcome: str, app_env: str) -> None:
    """Increment the startup.probe.outcome_total counter.

    Bound to the ``OutcomeCounterCallback`` slot on
    ``StartupRunner``: invoked once per probe with the OBSERVABILITY
    outcome (after the runner's non-required FAIL -> WARN downgrade).

    Like ``record_replica_count_warning``, this guards against a
    missing instrument with ``.get()`` rather than raising: when the
    OTel meter is unwired (e.g. a worker that never reached
    ``configure_otel``), the lookup returns None and the call is a
    no-op. The runner catches probe-author-side counter exceptions
    via its own try/except, so this helper raising would not abort
    boot -- but a no-op is the cleaner contract.
    """
    inst = initialize_metrics()
    raw = inst.get(MetricNames.STARTUP_PROBE_OUTCOME)
    if raw is not None:
        cast(Counter, raw).add(
            1,
            {"name": name, "outcome": outcome, "app_env": app_env},
        )
