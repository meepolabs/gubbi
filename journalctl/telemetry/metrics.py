"""Metric instruments for journalctl (TASK-03.19).

Defines and initializes the journalctl-side metric instruments:
    - mcp.tool_call_duration (histogram)
    - mcp.tool_call_count (counter)
    - mcp.tool_response_size_chars (histogram)
    - audit.persistence_failure (counter)

Usage::

    from journalctl.telemetry.metrics import (
        TOOL_CALL_DURATION,
        TOOL_CALL_COUNT,
    )

    TOOL_CALL_COUNT.add(1, {"tool.name": "journal_append_entry"})
    TOOL_CALL_DURATION.record(latency_ms, {"tool.name": name})
"""

from __future__ import annotations

import logging
from typing import Any

from opentelemetry import metrics

from journalctl.telemetry.attrs import BANNED_KEYS, MetricNames

logger = logging.getLogger(__name__)

_METER_NAME = "journalctl"
_meter: Any = None  # Lazy-initialized meter


def _get_meter() -> Any:
    """Return the journalctl meter, creating it on first access."""
    global _meter  # noqa: PLW0603
    if _meter is None:
        _meter = metrics.get_meter(_METER_NAME)
    return _meter


# -- Histogram: mcp.tool_call_duration (latency per tool in ms) --
TOOL_CALL_DURATION = _get_meter().create_histogram(
    name=MetricNames.MCP_TOOL_CALL_DURATION,
    description="Per-tool call latency in milliseconds",
    unit="ms",
)

# -- Counter: mcp.tool_call_count (invocations per tool) --
TOOL_CALL_COUNT = _get_meter().create_counter(
    name=MetricNames.MCP_TOOL_CALL_COUNT,
    description="Per-tool invocation count",
    unit="1",
)

# -- Histogram: mcp.tool_response_size_chars (size per tool) --
TOOL_RESPONSE_SIZE_CHARS = _get_meter().create_histogram(
    name=MetricNames.MCP_TOOL_RESPONSE_SIZE_CHARS,
    description="Per-tool response size in characters",
    unit="chars",
)

# -- Counter: audit.persistence_failure (by event_type) --
AUDIT_PERSISTENCE_FAILURE = _get_meter().create_counter(
    name=MetricNames.AUDIT_PERSISTENCE_FAILURE,
    description="Audit log persistence failure count by event type",
    unit="1",
)


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
    the tool name attribute.
    """
    attrs = _validate_metric_attrs({"tool.name": tool_name})
    TOOL_CALL_DURATION.record(latency_ms, attributes=attrs)
    TOOL_CALL_COUNT.add(1, attributes=attrs)


def record_tool_response_size(size_chars: int, tool_name: str) -> None:
    """Record a tool response size in the histogram."""
    attrs = _validate_metric_attrs({"tool.name": tool_name})
    TOOL_RESPONSE_SIZE_CHARS.record(size_chars, attributes=attrs)


def record_audit_persistence_failure(event_type: str) -> None:
    """Increment the audit persistence failure counter."""
    attrs = _validate_metric_attrs({"event_type": event_type})
    AUDIT_PERSISTENCE_FAILURE.add(1, attributes=attrs)
