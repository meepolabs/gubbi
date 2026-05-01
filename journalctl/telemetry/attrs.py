"""Span name constants + re-export of the unified allowlist (DEC-070 / TASK-03.19).

The privacy allowlist (BANNED_KEYS, SPAN_ALLOWLIST, safe_set_attributes)
now lives in :mod:`gubbi_common.telemetry.allowlist`. This module
keeps the journalctl-specific span and metric name constants and
re-exports the shared symbols so existing imports such as ::

    from journalctl.telemetry.attrs import safe_set_attributes, SpanNames

continue to work without touching every call site.

Substring matching is now applied uniformly to ``BANNED_KEYS``, so
attribute keys such as ``content_hash``, ``email_hash`` and
``client_user_agent_hash`` are dropped even when not enumerated. This
matches the historical journalctl behaviour and is the conservative
interpretation of DEC-070.
"""

from __future__ import annotations

from typing import Final

from gubbi_common.telemetry.allowlist import (
    BANNED_KEYS,
    SPAN_ALLOWLIST,
    safe_set_attributes,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
_TRACER_NAME: Final[str] = "journalctl"
_NS_PER_MS: Final[int] = 1_000_000


# ---------------------------------------------------------------------------
# Span name constants -- single source of truth for journalctl critical-path
# spans. Kept here (not in gubbi-common) because they are journalctl-specific
# string constants used by callers; the shared module only owns the
# allowlist tables keyed by span-name strings.
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


# Per-span attribute schemas referenced directly by tests and by callers
# that pre-validate keys before passing them to safe_set_attributes.
# These remain authoritative for the constants they name, but are
# narrower than the unified SPAN_ALLOWLIST (which adds correlation_id).
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


def get_allowlisted_attrs(span_name: str) -> frozenset[str]:
    """Return the allowlisted attribute set for *span_name*.

    Falls back to an empty frozenset for unknown span names so callers
    always get a predictable value (no attributes pass through).
    """
    allowed: frozenset[str] | None = SPAN_ALLOWLIST.get(span_name)
    if allowed is None:
        return frozenset()
    return allowed


__all__ = [
    "BANNED_KEYS",
    "MCP_TOOL_CALL_ATTRS",
    "MetricNames",
    "SPAN_ALLOWLIST",
    "SpanNames",
    "_NS_PER_MS",
    "_TRACER_NAME",
    "get_allowlisted_attrs",
    "safe_set_attributes",
]
