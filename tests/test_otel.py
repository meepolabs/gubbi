"""Tests for OpenTelemetry instrumentation (TASK-03.19).

Coverage:
    - Allowlist verification: banned attributes are dropped at the exporter layer.
    - mcp.tool_call golden path: required attributes present.
    - correlation_id propagation: inbound header reaches span attribute + log entry.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Generator
from typing import Any

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

from journalctl.telemetry.attrs import (
    BANNED_KEYS,
    MCP_TOOL_CALL_ATTRS,
    SpanNames,
    safe_set_attributes,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# In-memory span exporter for test assertions
# ---------------------------------------------------------------------------


class InMemoryExporter(SpanExporter):
    """Stores exported spans in a list for test assertions."""

    def __init__(self) -> None:
        self.spans: list[ReadableSpan] = []

    def export(self, spans: list[ReadableSpan]) -> SpanExportResult:  # type: ignore[override]
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        self.spans.clear()


@pytest.fixture
def in_memory_tracer() -> Generator[tuple[Any, InMemoryExporter], None, None]:
    """Yield (tracer, exporter) with in-memory span export."""
    exporter = InMemoryExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    old_provider = trace.get_tracer_provider()
    trace.set_tracer_provider(provider)
    try:
        yield tracer, exporter
    finally:
        trace.set_tracer_provider(old_provider)


# ---------------------------------------------------------------------------
# Allowlist verification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("banned_key", sorted(BANNED_KEYS))
def test_banned_attribute_dropped(
    in_memory_tracer: tuple[Any, InMemoryExporter],
    banned_key: str,
) -> None:
    """Banned attribute keys must not appear in exported span attributes.

    For each banned key, build a synthetic span, set it via
    safe_set_attributes, and assert it is absent from the exporter.
    """
    tracer, exporter = in_memory_tracer
    span_name = SpanNames.MCP_TOOL_CALL

    with tracer.start_as_current_span(span_name) as span:
        safe_set_attributes(span_name, span, {banned_key: "should-not-leak"})

    # Flush
    assert len(exporter.spans) == 1
    exported = exporter.spans[0]
    attrs = dict(exported.attributes) if exported.attributes else {}
    assert banned_key not in attrs, f"Banned key {banned_key!r} found in exported span attributes"


def test_banned_substring_in_key_dropped(in_memory_tracer: tuple[Any, InMemoryExporter]) -> None:
    """Attribute keys containing a banned substring are also dropped."""
    tracer, exporter = in_memory_tracer
    span_name = SpanNames.MCP_TOOL_CALL

    with tracer.start_as_current_span(span_name) as span:
        # "content_hash" contains "content" -- should be dropped
        safe_set_attributes(span_name, span, {"content_hash": "abc123", "tool.name": "good"})

    assert len(exporter.spans) == 1
    attrs = dict(exporter.spans[0].attributes) if exporter.spans[0].attributes else {}
    assert "content_hash" not in attrs
    assert attrs.get("tool.name") == "good"


# ---------------------------------------------------------------------------
# mcp.tool_call golden path
# ---------------------------------------------------------------------------


def test_mcp_tool_call_span_golden_path(in_memory_tracer: tuple[Any, InMemoryExporter]) -> None:
    """mcp.tool_call span must have all golden-path attributes."""
    tracer, exporter = in_memory_tracer
    span_name = SpanNames.MCP_TOOL_CALL

    attrs: dict[str, Any] = {
        "tool.name": "journal_append_entry",
        "user_id": str(uuid.uuid4()),
        "tool.scope_required": "write",
        "result": "success",
        "result.size_chars": 128,
        "latency_ms": 42.0,
    }
    with tracer.start_as_current_span(span_name) as span:
        safe_set_attributes(span_name, span, attrs)

    assert len(exporter.spans) == 1
    exported = exporter.spans[0]
    exported_attrs = dict(exported.attributes) if exported.attributes else {}

    # All MCP_TOOL_CALL_ATTRS must be present
    for attr_key in MCP_TOOL_CALL_ATTRS:
        assert (
            attr_key in exported_attrs
        ), f"Missing required attribute {attr_key!r} in mcp.tool_call span"
        assert exported_attrs[attr_key] == attrs[attr_key], (
            f"Attribute {attr_key!r} value mismatch: "
            f"expected {attrs[attr_key]!r}, got {exported_attrs[attr_key]!r}"
        )


def test_mcp_tool_call_span_name(in_memory_tracer: tuple[Any, InMemoryExporter]) -> None:
    """Span name must be 'mcp.tool_call'."""
    tracer, exporter = in_memory_tracer

    with tracer.start_as_current_span(SpanNames.MCP_TOOL_CALL) as span:
        safe_set_attributes(SpanNames.MCP_TOOL_CALL, span, {"tool.name": "test"})

    assert len(exporter.spans) == 1
    assert exporter.spans[0].name == "mcp.tool_call"


# ---------------------------------------------------------------------------
# Correlation ID propagation
# ---------------------------------------------------------------------------


def test_correlation_id_in_span_attributes(in_memory_tracer: tuple[Any, InMemoryExporter]) -> None:
    """Correlation ID must appear as a span attribute when routed through safe_set_attributes."""
    tracer, exporter = in_memory_tracer

    from journalctl.telemetry.logging import set_correlation_id

    cid = str(uuid.uuid4())
    set_correlation_id(cid)

    span_name = SpanNames.MCP_TOOL_CALL
    with tracer.start_as_current_span(span_name) as span:
        # Route correlation_id through safe_set_attributes (not set_attribute)
        safe_set_attributes(span_name, span, {"tool.name": "test", "correlation_id": cid})

    assert len(exporter.spans) == 1
    attrs = dict(exporter.spans[0].attributes) if exporter.spans[0].attributes else {}
    assert (
        attrs.get("correlation_id") == cid
    ), f"Expected correlation_id={cid!r} in span attributes, got {attrs.get('correlation_id')!r}"


def test_correlation_id_generated_when_missing() -> None:
    """CorrelationIDMiddleware must generate a UUID when no header is present."""
    from journalctl.middleware.correlation import CorrelationIDMiddleware

    # Test the extract method directly
    scope: dict[str, Any] = {"headers": []}
    cid = CorrelationIDMiddleware._extract_or_generate(scope)
    assert cid is not None
    assert len(cid) > 0
    # Verify it's a valid UUID
    parsed = uuid.UUID(cid)
    assert str(parsed) == cid


def test_correlation_id_extracted_from_header() -> None:
    """CorrelationIDMiddleware must extract correlation_id from request header."""
    from journalctl.middleware.correlation import CorrelationIDMiddleware

    cid_in = str(uuid.uuid4())
    scope: dict[str, Any] = {
        "headers": [
            (b"x-correlation-id", cid_in.encode("latin-1")),
        ]
    }
    cid_out = CorrelationIDMiddleware._extract_or_generate(scope)
    assert cid_out == cid_in


# ---------------------------------------------------------------------------
# safe_set_attributes edge cases
# ---------------------------------------------------------------------------


def test_safe_set_attributes_unknown_span_name(
    in_memory_tracer: tuple[Any, InMemoryExporter],
) -> None:
    """Unknown span names should not allow any attributes through."""
    tracer, exporter = in_memory_tracer

    with tracer.start_as_current_span("unknown.span") as span:
        safe_set_attributes("unknown.span", span, {"some_key": "value"})

    assert len(exporter.spans) == 1
    attrs = dict(exporter.spans[0].attributes) if exporter.spans[0].attributes else {}
    assert "some_key" not in attrs


# ---------------------------------------------------------------------------
# StructuredLogFormatter: JSON schema compliance
# ---------------------------------------------------------------------------


def test_structured_log_formatter_has_required_fields() -> None:
    """StructuredLogFormatter must emit JSON with required schema fields."""
    from journalctl.telemetry.logging import StructuredLogFormatter

    formatter = StructuredLogFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="test.event",
        args=None,
        exc_info=None,
    )
    output = formatter.format(record)
    import json

    parsed = json.loads(output)
    assert "timestamp" in parsed
    assert "level" in parsed
    assert parsed["level"] == "INFO"
    assert "service" in parsed
    assert parsed["service"] == "journalctl"
    assert "event" in parsed
    assert parsed["event"] == "test.event"
    # correlation_id, trace_id, span_id are optional (may be None)
    assert "correlation_id" in parsed
    assert "trace_id" in parsed
    assert "span_id" in parsed
