"""Span name constants + gubbi-specific span attribute allowlist.

The global SPAN_ALLOWLIST was removed in gubbi-common v0.2.0;
``safe_set_attributes`` now requires an ``allowlist`` kwarg. This module
maintains the gubbi-specific allowlist and a local wrapper so that
existing call sites do not need to change.

``BANNED_KEYS`` is re-exported from gubbi-common for convenience.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from gubbi_common.telemetry.allowlist import (
    BANNED_KEYS,
)
from gubbi_common.telemetry.allowlist import (
    safe_set_attributes as _gc_safe_set_attributes,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from opentelemetry.trace import Span

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
# Underscore-prefixed names indicate they are intentionally not part of
# the public ``__all__`` API for ``import *`` consumers, but they are
# explicitly imported by name across the package (`gubbi.telemetry.spans`,
# `gubbi.crypto.cipher`, `gubbi.tools.registry`,
# `gubbi.storage.embedding_service`, `gubbi.extraction.jobs.extract_conversation`).
# Treat as package-internal constants -- canonical home is here, do not
# duplicate elsewhere.
_TRACER_NAME: Final[str] = "gubbi"
_NS_PER_MS: Final[int] = 1_000_000


# ---------------------------------------------------------------------------
# Span name constants -- single source of truth for gubbi critical-path
# spans.
# ---------------------------------------------------------------------------
class SpanNames:
    """Canonical span names for gubbi critical-path spans."""

    MCP_TOOL_CALL: Final[str] = "mcp.tool_call"
    MCP_TOOL_RESPONSE_SIZE_CHECK: Final[str] = "mcp.tool_response_size_check"
    EMBEDDING_ENCODE: Final[str] = "embedding.encode"
    CIPHER_ENCRYPT: Final[str] = "cipher.encrypt"
    CIPHER_DECRYPT: Final[str] = "cipher.decrypt"
    AUDIT_WRITE: Final[str] = "audit.write"
    HTTP_REQUEST: Final[str] = "http.request"
    EXTRACTION_JOB: Final[str] = "extraction.job"
    EXTRACTION_LLM_CALL: Final[str] = "extraction.llm_call"


# ---------------------------------------------------------------------------
# Metric name constants
# ---------------------------------------------------------------------------
class MetricNames:
    """Canonical metric names for gubbi metrics."""

    MCP_TOOL_CALL_DURATION: Final[str] = "mcp.tool_call_duration"
    MCP_TOOL_CALL_COUNT: Final[str] = "mcp.tool_call_count"
    MCP_TOOL_RESPONSE_SIZE_CHARS: Final[str] = "mcp.tool_response_size_chars"
    AUDIT_PERSISTENCE_FAILURE: Final[str] = "audit.persistence_failure"
    # Instrument name is deliberately the SAME as cloud-api's
    # ``gateway.replica_count_warning`` so one HyperDX alerting rule rolls
    # up across cloud-api + gubbi + worker (OTel aggregates by instrument
    # name, with service.name as a resource dimension). The ``gateway.``
    # prefix is reused on purpose even though gubbi is the MCP server, not
    # the gateway -- it is the cross-service alert-rule join key, not a
    # claim about which component emits it. The ``service`` attribute on
    # the increment distinguishes emitters (gubbi vs gubbi-worker).
    REPLICA_COUNT_WARNING: Final[str] = "gateway.replica_count_warning"
    # Cross-service startup-probe outcome counter -- shared NAME with
    # cloud-api so a single HyperDX rule rolls up across both. Emitter
    # split via service.name; per-call attributes (probe name, outcome,
    # app_env) cap cardinality at the small product of {probe-set} *
    # {ok|warn|fail} * {dev|ci|staging|production}.
    STARTUP_PROBE_OUTCOME: Final[str] = "startup.probe.outcome_total"
    # A background loop lost a tick or stopped; ``task`` names the loop,
    # ``stage`` is tick, loop, not_started or stop_timeout.
    BACKGROUND_TASK_DIED: Final[str] = "background_task.died_total"
    # Extraction queue and budget gauges, recorded once per scheduler tick.
    EXTRACTION_QUEUE_OLDEST_RUNNABLE_AGE: Final[str] = (
        "extraction.queue.oldest_runnable_age_seconds"
    )
    EXTRACTION_QUEUE_DEPTH: Final[str] = "extraction.queue.depth"
    EXTRACTION_QUEUE_DEFERRED: Final[str] = "extraction.queue.deferred"
    EXTRACTION_BUDGET_GLOBAL_USED_RATIO: Final[str] = "extraction.budget.global_used_ratio"
    EXTRACTION_PROVIDER_TOKENS_MONTH: Final[str] = "extraction.provider.tokens_month"
    # Incremented by the extraction worker on a fixed interval while it runs.
    EXTRACTION_WORKER_HEARTBEAT: Final[str] = "extraction.worker.heartbeat"
    # One per extraction job failure written, by retry class and whether the
    # job will run again.
    EXTRACTION_JOBS_FAILED: Final[str] = "extraction.jobs.failed_total"
    # Weighted tokens charged to user wallets, by lane and by which balance
    # paid: ``monthly`` allowance or ``credit``.
    EXTRACTION_CHARGES_WEIGHTED: Final[str] = "extraction.charges.weighted_total"
    # LLM calls of finished job attempts whose usage the upstream never
    # reported (timed out), so their cost is missing from the charge; by lane.
    EXTRACTION_LLM_UNMETERED_CALLS: Final[str] = "extraction.llm.unmetered_calls_total"
    # Synced messages written by ingest whose text was cut to the length limit.
    INGEST_MESSAGES_TRUNCATED: Final[str] = "ingest.messages.truncated_total"


# Attribute keys each allowlisted metric may carry; any other key is dropped.
# Values must come from small fixed sets: never a user, job or conversation id.
GUBBI_METRIC_ALLOWLIST: Mapping[str, frozenset[str]] = {
    MetricNames.EXTRACTION_QUEUE_OLDEST_RUNNABLE_AGE: frozenset(),
    MetricNames.EXTRACTION_QUEUE_DEPTH: frozenset({"lane"}),
    MetricNames.EXTRACTION_QUEUE_DEFERRED: frozenset({"defer_reason"}),
    MetricNames.EXTRACTION_BUDGET_GLOBAL_USED_RATIO: frozenset(),
    MetricNames.EXTRACTION_PROVIDER_TOKENS_MONTH: frozenset({"kind"}),
    MetricNames.EXTRACTION_WORKER_HEARTBEAT: frozenset(),
    MetricNames.EXTRACTION_JOBS_FAILED: frozenset({"error_class", "outcome"}),
    MetricNames.EXTRACTION_CHARGES_WEIGHTED: frozenset({"lane", "part"}),
    MetricNames.EXTRACTION_LLM_UNMETERED_CALLS: frozenset({"lane"}),
    MetricNames.INGEST_MESSAGES_TRUNCATED: frozenset(),
}


# ---------------------------------------------------------------------------
# Journalctl-specific span attribute allowlist
# ---------------------------------------------------------------------------
# Built from the actual safe_set_attributes call sites in this repo.
# Every span name that gubbi uses gets a frozenset of the attribute
# keys actually passed to it.  Keys not listed here are dropped for
# known span names.
# ---------------------------------------------------------------------------
GUBBI_SPAN_ALLOWLIST: Mapping[str, frozenset[str]] = {
    SpanNames.MCP_TOOL_CALL: frozenset(
        {
            "tool.name",
            "user_id",
            "tool.scope_required",
            "result",
            "result.size_chars",
            "latency_ms",
            "correlation_id",
        }
    ),
    SpanNames.MCP_TOOL_RESPONSE_SIZE_CHECK: frozenset(
        {
            "tool.name",
            "size_chars",
            "error_threshold_hit",
            "correlation_id",
        }
    ),
    SpanNames.CIPHER_ENCRYPT: frozenset(
        {
            "version",
            "field_kind",
            "bytes_processed",
            "latency_ms",
            "correlation_id",
        }
    ),
    SpanNames.CIPHER_DECRYPT: frozenset(
        {
            "version",
            "field_kind",
            "bytes_processed",
            "latency_ms",
            "correlation_id",
        }
    ),
    SpanNames.AUDIT_WRITE: frozenset(
        {
            "event_type",
            "actor_type",
            "success",
            "latency_ms",
            "correlation_id",
        }
    ),
    SpanNames.EMBEDDING_ENCODE: frozenset(
        {
            "text_hash",
            "text_len",
            "latency_ms",
            "correlation_id",
        }
    ),
    SpanNames.HTTP_REQUEST: frozenset(
        {
            "correlation_id",
        }
    ),
    SpanNames.EXTRACTION_JOB: frozenset(
        {
            "job_id",
            "user_id",
            "conversation_id",
            "success",
            "failure_reason",
            "attempt_number",
            "latency_ms",
            "correlation_id",
        }
    ),
    SpanNames.EXTRACTION_LLM_CALL: frozenset(
        {
            "provider_name",
            "model_name",
            "prompt_size",
            "completion_size",
            "cached_prompt_size",
            "reasoning_size",
            "finish_reason",
            "attempts",
            "chunk_index",
            "stage",
            "latency_ms",
            "correlation_id",
        }
    ),
}


# Per-span attribute schemas referenced directly by tests and by callers
# that pre-validate keys before passing them to safe_set_attributes.
# These are a subset of the GUBBI_SPAN_ALLOWLIST entries (they omit
# correlation_id which is added by the allowlist).
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


def safe_set_attributes(
    span_name: str,
    span: Span,
    attrs: Mapping[str, Any],
) -> None:
    """Set span attributes filtered through the gubbi allowlist.

    Span names not present in GUBBI_SPAN_ALLOWLIST have ALL attributes
    dropped (only a DEBUG log is emitted). Register new span names in the
    allowlist before instrumenting.
    """
    _gc_safe_set_attributes(span_name, span, attrs, allowlist=GUBBI_SPAN_ALLOWLIST)


def get_allowlisted_attrs(span_name: str) -> frozenset[str]:
    """Return the allowlisted attribute set for *span_name*.

    Falls back to an empty frozenset for unknown span names so callers
    always get a predictable value (no attributes pass through).
    """
    allowed: frozenset[str] | None = GUBBI_SPAN_ALLOWLIST.get(span_name)
    if allowed is None:
        return frozenset()
    return allowed


__all__ = [
    "BANNED_KEYS",
    "GUBBI_METRIC_ALLOWLIST",
    "GUBBI_SPAN_ALLOWLIST",
    "MCP_TOOL_CALL_ATTRS",
    "MetricNames",
    "SpanNames",
    "get_allowlisted_attrs",
    "safe_set_attributes",
]
