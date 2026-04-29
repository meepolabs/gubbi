"""Attribute allowlist for OpenTelemetry spans (DEC-070 / TASK-03.19).

Hard rule: NO journal content in spans, metrics, or structured logs.
This module defines the per-span attribute schema and enforces the
allowlist at the span-builder layer. Anything not in the schema is
dropped with a log warning.

Banned attribute keys (any occurrence):
    content, reasoning, summary, messages, body from any tool input or
    output; raw email addresses; raw user-agent strings; raw IP addresses
    outside audit_log; search query strings; tool error messages that
    quote offending data.
"""

from __future__ import annotations

import logging
from typing import Any, Final

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
_TRACER_NAME: Final[str] = "journalctl"
_NS_PER_MS: Final[int] = 1_000_000

# ---------------------------------------------------------------------------
# Banned attribute keys — frozenset for O(1) membership checks
# ---------------------------------------------------------------------------
# These keys are NEVER allowed in span attributes, metric attributes, or
# structured log fields originating from the journalctl process.
BANNED_KEYS: Final[frozenset[str]] = frozenset(
    {
        "content",
        "reasoning",
        "summary",
        "messages",
        "body",
        "email",
        "user_agent",
        "ip_address",
        "query",
        "search_query",
    }
)


# ---------------------------------------------------------------------------
# Span name constants — single source of truth for every critical-path span
# ---------------------------------------------------------------------------
class SpanNames:
    """Canonical span names for journalctl critical-path spans."""

    MCP_TOOL_CALL: Final[str] = "mcp.tool_call"
    MCP_TOOL_RESPONSE_SIZE_CHECK: Final[str] = "mcp.tool_response_size_check"
    DB_QUERY_USER_SCOPED: Final[str] = "db.query.user_scoped"
    EMBEDDING_ENCODE: Final[str] = "embedding.encode"
    CIPHER_ENCRYPT: Final[str] = "cipher.encrypt"
    CIPHER_DECRYPT: Final[str] = "cipher.decrypt"
    AUDIT_WRITE: Final[str] = "audit.write"
    HTTP_REQUEST: Final[str] = "http.request"


# ---------------------------------------------------------------------------
# Metric name constants
# ---------------------------------------------------------------------------
class MetricNames:
    """Canonical metric names for journalctl metrics."""

    MCP_TOOL_CALL_DURATION: Final[str] = "mcp.tool_call_duration"
    MCP_TOOL_CALL_COUNT: Final[str] = "mcp.tool_call_count"
    MCP_TOOL_RESPONSE_SIZE_CHARS: Final[str] = "mcp.tool_response_size_chars"
    AUDIT_PERSISTENCE_FAILURE: Final[str] = "audit.persistence_failure"


# ---------------------------------------------------------------------------
# Per-span attribute schema
# ---------------------------------------------------------------------------
# Each span kind has a frozen set of allowlisted attribute keys. Any
# attribute key not present in the corresponding set is dropped.

MCP_TOOL_CALL_ATTRS: Final[frozenset[str]] = frozenset(
    {
        "tool.name",
        "user_id",
        "tool.scope_required",
        "result",
        "result.size_chars",
        "latency_ms",
    }
)

MCP_TOOL_RESPONSE_SIZE_CHECK_ATTRS: Final[frozenset[str]] = frozenset(
    {
        "tool.name",
        "size_chars",
        "error_threshold_hit",
    }
)

DB_QUERY_USER_SCOPED_ATTRS: Final[frozenset[str]] = frozenset(
    {
        "query_kind",
        "user_id",
        "row_count",
        "latency_ms",
    }
)

EMBEDDING_ENCODE_ATTRS: Final[frozenset[str]] = frozenset(
    {
        "text_hash",
        "text_len",
        "latency_ms",
    }
)

CIPHER_OP_ATTRS: Final[frozenset[str]] = frozenset(
    {
        "version",
        "field_kind",
        "bytes_processed",
        "latency_ms",
    }
)

AUDIT_WRITE_ATTRS: Final[frozenset[str]] = frozenset(
    {
        "event_type",
        "target_id",
        "actor_type",
        "success",
        "latency_ms",
    }
)

CORRELATION_ID_ATTRS: Final[frozenset[str]] = frozenset(
    {
        "correlation_id",
    }
)

# Mapping from span name to its allowlisted attribute keys.
# Correlation ID attributes are merged into every span allowlist so
# safe_set_attributes can set correlation_id on any span.
_SPAN_ALLOWLISTS: dict[str, frozenset[str]] = {
    SpanNames.MCP_TOOL_CALL: MCP_TOOL_CALL_ATTRS | CORRELATION_ID_ATTRS,
    SpanNames.MCP_TOOL_RESPONSE_SIZE_CHECK: (
        MCP_TOOL_RESPONSE_SIZE_CHECK_ATTRS | CORRELATION_ID_ATTRS
    ),
    SpanNames.DB_QUERY_USER_SCOPED: DB_QUERY_USER_SCOPED_ATTRS | CORRELATION_ID_ATTRS,
    SpanNames.EMBEDDING_ENCODE: EMBEDDING_ENCODE_ATTRS | CORRELATION_ID_ATTRS,
    SpanNames.CIPHER_ENCRYPT: CIPHER_OP_ATTRS | CORRELATION_ID_ATTRS,
    SpanNames.CIPHER_DECRYPT: CIPHER_OP_ATTRS | CORRELATION_ID_ATTRS,
    SpanNames.AUDIT_WRITE: AUDIT_WRITE_ATTRS | CORRELATION_ID_ATTRS,
    SpanNames.HTTP_REQUEST: CORRELATION_ID_ATTRS,
}


def get_allowlisted_attrs(span_name: str) -> frozenset[str]:
    """Return the allowlisted attribute set for *span_name*.

    Falls back to an empty frozenset for unknown span names so callers
    always get a predictable value (no attributes pass through).
    """
    return _SPAN_ALLOWLISTS.get(span_name, frozenset())


def safe_set_attributes(
    span_name: str,
    span: Any,
    attrs: dict[str, Any],
) -> None:
    """Set attributes on *span*, dropping any key not in the allowlist.

    Banned keys (matching BANNED_KEYS by substring or exact match) are
    dropped with a warning. Non-allowlisted keys are also dropped.
    This is the sole entry point for setting span attributes in
    journalctl — never call ``span.set_attribute`` directly.

    Parameters
    ----------
    span_name:
        The canonical span name (e.g. ``SpanNames.MCP_TOOL_CALL``).
        Used to look up the allowlist.
    span:
        The OpenTelemetry span to set attributes on.
    attrs:
        Dictionary of {key: value} to set. Keys not in the per-span
        allowlist are silently dropped.
    """
    allowlist = get_allowlisted_attrs(span_name)
    for key, value in attrs.items():
        # Check banned keys first — exact match or substring match
        if key in BANNED_KEYS:
            logger.warning(
                "Dropping banned span attribute %r on span %s",
                key,
                span_name,
            )
            continue
        for banned in BANNED_KEYS:
            if banned in key:
                logger.warning(
                    "Dropping span attribute %r (contains banned key %r) on span %s",
                    key,
                    banned,
                    span_name,
                )
                break
        else:
            if key in allowlist:
                span.set_attribute(key, value)
            else:
                logger.debug(
                    "Dropping non-allowlisted span attribute %r on span %s",
                    key,
                    span_name,
                )
