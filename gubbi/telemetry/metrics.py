"""Metric instruments for gubbi (TASK-03.19).

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

CRIT-5 (M4 e2e review, 2026-05-13): prior to this refactor the four
instruments were created at module import, which runs before
``configure_otel()`` -- they bound to the NoOp provider and silently
discarded every ``.add(...)`` / ``.record(...)``. That made the
``audit.persistence_failure`` alarm contract that DEC-098 depends on
non-functional. The fix is to defer instrument creation to
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

from opentelemetry.metrics import Counter, Histogram, get_meter

from gubbi.telemetry.attrs import BANNED_KEYS, MetricNames

__all__: list[str] = [
    "initialize_metrics",
    "record_audit_persistence_failure",
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
    return inst


def _validate_metric_attrs(attrs: dict[str, str]) -> dict[str, str]:
    """Strip banned keys from metric attributes.

    Same privacy rules as span attributes: no content, email, etc.
    Returns a new dict with only allowed keys.
    """
    cleaned: dict[str, str] = {}
    for key, value in attrs.items():
        if key in BANNED_KEYS:
            logger.warning("Dropping banned metric attribute %r", key)
            continue
        skip = False
        for banned in BANNED_KEYS:
            if banned in key:
                logger.warning("Dropping metric attribute %r (contains banned key %r)", key, banned)
                skip = True
                break
        if not skip:
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

    CRIT-5: this is the sensor for the DEC-098 fail-open compensating
    alarm. Must bind to the real meter provider (post-lifespan) for the
    HyperDX alarm contract to fire. ``initialize_metrics()`` is
    lru_cached and primed at the tail of ``configure_otel()`` so the
    cached instruments are bound to the configured provider.

    Direct ``inst[key]`` access: a missing key here would silently
    disable the DEC-098 alarm sensor. Let KeyError surface.
    """
    attrs = _validate_metric_attrs({"event_type": event_type})
    inst = initialize_metrics()
    cast(Counter, inst[MetricNames.AUDIT_PERSISTENCE_FAILURE]).add(1, attributes=attrs)
