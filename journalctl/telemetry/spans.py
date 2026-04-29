"""Span helpers for journalctl critical-path spans (TASK-03.19).

Provides async context managers for spans that are not instrumented
inline in the relevant modules. Currently that is the forward-prep
mcp.tool_response_size_check hook (TASK-03.23).

The primary critical-path spans (mcp.tool_call, db.query.user_scoped,
embedding.encode, cipher.encrypt/decrypt, audit.write) are instrumented
inline in their respective modules using safe_set_attributes directly.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from opentelemetry import trace

from journalctl.telemetry.attrs import _TRACER_NAME, SpanNames, safe_set_attributes

if TYPE_CHECKING:
    from opentelemetry.trace import Span

logger = logging.getLogger(__name__)

_tracer = trace.get_tracer(_TRACER_NAME)


# ---------------------------------------------------------------------------
# mcp.tool_response_size_check  (forward-prep for TASK-03.23)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def record_mcp_tool_response_size_check(
    tool_name: str,
    size_chars: int,
    error_threshold_hit: bool,
) -> AsyncIterator[Span]:
    """Async context manager for the safety-net size check span.

    This is a forward-prep hook for TASK-03.23. Once the 80KB safety
    net lands, callers fire this span when a tool response exceeds the
    threshold.

    Attributes:
        tool.name, size_chars, error_threshold_hit
    """
    span_name = SpanNames.MCP_TOOL_RESPONSE_SIZE_CHECK
    attrs: dict[str, Any] = {
        "tool.name": tool_name,
        "size_chars": size_chars,
        "error_threshold_hit": error_threshold_hit,
    }
    with _tracer.start_as_current_span(span_name) as span:
        safe_set_attributes(span_name, span, attrs)
        yield span
