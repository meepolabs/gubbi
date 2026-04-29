"""Register all MCP tools on the FastMCP server.

Defines ``READ_TOOLS`` and ``WRITE_TOOLS`` categorization lists.  These
live centrally in this module rather than per-tool-module because the
categorization is the single thing that has to change atomically when
the ``journal:read`` / ``journal:write`` split lands -- one diff in one
file beats five diffs across five tool modules.  The aggregation gives
``filter_tools_by_scope()`` and downstream consumers a single source of
truth.

Exposes ``filter_tools_by_scope()`` -- a data-driven filter hook for
``tools/list``.  In v1 a token with ``journal`` scope sees every tool;
tokens without it see nothing (default-deny defense in depth -- the
middleware already rejects no-scope tokens at the HTTP layer, but the
filter must not fail open if that ever changes).  After the split,
update the mapping below.

Also monkey-patches the MCP SDK's ToolManager.call_tool to add
OpenTelemetry ``mcp.tool_call`` spans around every tool dispatch
(TASK-03.19).
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.tools import ToolManager

# Imported for type info when available; but we monkey-patch via ToolManager
from opentelemetry import trace

from journalctl.core.context import AppContext
from journalctl.telemetry.attrs import _NS_PER_MS, _TRACER_NAME, SpanNames, safe_set_attributes
from journalctl.tools import (
    context,
    conversations,
    entries,
    search,
    topics,
)

# ---------------------------------------------------------------------------
# Tool categorization lists (used by future read/write scope split)
# ---------------------------------------------------------------------------

READ_TOOLS: frozenset[str] = frozenset(
    {
        "journal_search",
        "journal_list_topics",
        "journal_list_conversations",
        "journal_read_topic",
        "journal_read_conversation",
        "journal_briefing",
        "journal_timeline",
    }
)

WRITE_TOOLS: frozenset[str] = frozenset(
    {
        "journal_append_entry",
        "journal_update_entry",
        "journal_create_topic",
        "journal_save_conversation",
        "journal_delete_entry",
    }
)

ALL_TOOLS = READ_TOOLS | WRITE_TOOLS


def filter_tools_by_scope(
    tool_names: list[str],
    token_scopes: set[str],
) -> list[str]:
    """Filter a tool list by the token's scopes.

    v1: a token with ``journal`` scope sees every tool; tokens without it
    see nothing.  This is the second line of defense behind
    ``BearerAuthMiddleware``, which already rejects no-scope tokens at the
    HTTP layer; if that check is ever bypassed or relaxed, this filter
    must not fail open.

    After the ``journal:read`` / ``journal:write`` split, replace the
    short-circuit with the per-scope mapping (``READ_TOOLS`` for
    ``journal:read``, ``READ_TOOLS | WRITE_TOOLS`` for ``journal:write``).

    Parameters
    ----------
    tool_names :
        Full list of tool names (from ``tools/list``).
    token_scopes :
        The scopes granted to the current token.

    Returns
    -------
    list[str]
        The subset of ``tool_names`` the token is authorized to call.
    """
    if "journal" in token_scopes:
        return tool_names

    # Default-deny: tokens without journal scope see no tools.
    return []


logger = logging.getLogger(__name__)

_tracer = trace.get_tracer(_TRACER_NAME)


def _patch_tool_manager(tm: ToolManager) -> None:
    """Monkey-patch ToolManager.call_tool to emit mcp.tool_call spans.

    Every tool dispatch is wrapped with a span containing:
        tool.name, result, result.size_chars, latency_ms
    """
    original_call_tool = tm.call_tool

    @functools.wraps(original_call_tool)
    async def patched_call_tool(
        self: ToolManager,
        name: str,
        arguments: dict[str, Any] = None,  # type: ignore[assignment]
        **kwargs: Any,
    ) -> Any:
        span_name = SpanNames.MCP_TOOL_CALL
        start_ns = time.monotonic_ns()
        attrs: dict[str, Any] = {
            "tool.name": name,
            # TODO(TASK-03.19a): wire user_id and tool.scope_required from
            # request context when available.
            "user_id": "",
            "tool.scope_required": "",
        }

        with _tracer.start_as_current_span(span_name) as span:
            safe_set_attributes(span_name, span, attrs)
            try:
                result = await original_call_tool(name, arguments or {}, **kwargs)
                result_str = str(result)
                result_size = len(result_str)
                latency_ms = (time.monotonic_ns() - start_ns) / _NS_PER_MS
                safe_set_attributes(
                    span_name,
                    span,
                    {
                        "result": "success",
                        "result.size_chars": result_size,
                        "latency_ms": round(latency_ms, 2),
                    },
                )
                return result
            except Exception as exc:
                from opentelemetry.trace import Status, StatusCode

                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR))
                latency_ms = (time.monotonic_ns() - start_ns) / _NS_PER_MS
                safe_set_attributes(
                    span_name,
                    span,
                    {
                        "result": type(exc).__name__,
                        "result.size_chars": 0,
                        "latency_ms": round(latency_ms, 2),
                    },
                )
                raise

    # Replace the bound method on the specific ToolManager instance.
    # We use a closure that wraps the original, bypassing descriptor protocol.
    bound_patched = patched_call_tool.__get__(tm, type(tm))  # type: ignore[attr-defined]
    tm.call_tool = bound_patched  # type: ignore[assignment]


def register_tools(mcp: FastMCP, app_ctx: AppContext) -> None:
    """Register all MCP tools. Semantic memory is internal -- not exposed to LLM.

    Args:
        mcp: FastMCP server instance.
        app_ctx: Application context (pool, embedding_service, settings, logger).
    """
    topics.register(mcp, app_ctx)
    entries.register(mcp, app_ctx)
    search.register(mcp, app_ctx)
    conversations.register(mcp, app_ctx)
    context.register(mcp, app_ctx)

    # Patch the ToolManager to add OTel spans around every tool dispatch.
    # The isinstance guard protects against an upstream FastMCP refactor
    # that swaps `_tool_manager` to a different shape: rather than blow
    # up at the first tool call, log loud and skip instrumentation.
    tool_manager = getattr(mcp, "_tool_manager", None)
    if tool_manager is None:
        logger.warning(
            "Could not patch ToolManager: " "_tool_manager attribute not found on FastMCP instance"
        )
    elif not isinstance(tool_manager, ToolManager):
        logger.warning(
            "Could not patch ToolManager: _tool_manager is %s, expected ToolManager",
            type(tool_manager).__name__,
        )
    else:
        _patch_tool_manager(tool_manager)
        logger.debug("OTel tool-call span wrapper installed on ToolManager")
