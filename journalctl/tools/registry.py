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
from collections.abc import Collection
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.tools import ToolManager
from mcp.types import Tool as MCPTool

# Imported for type info when available; but we monkey-patch via ToolManager
from opentelemetry import trace

from journalctl.core.auth_context import current_token_scopes
from journalctl.core.context import AppContext
from journalctl.core.scope import SCOPE_GRANTS
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
    token_scopes: Collection[str],
) -> list[str]:
    """Filter a tool list by the token's scopes.

    Expands each token scope through ``SCOPE_GRANTS`` to determine which
    permissions are granted, then returns only the tools matching those
    permissions.  A token with both ``journal:read`` and ``journal:write``
    (or the legacy ``journal`` scope) sees every tool.

    This is the second line of defense behind ``BearerAuthMiddleware``,
    which already rejects no-scope tokens at the HTTP layer; if that check
    is ever bypassed or relaxed, this filter must not fail open.

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
    granted: set[str] = set()
    for ts in token_scopes:
        if ts in SCOPE_GRANTS:
            granted.update(SCOPE_GRANTS[ts])
    if not granted:
        return []

    # Both read and write -> all known tools
    if "journal:read" in granted and "journal:write" in granted:
        return [t for t in tool_names if t in ALL_TOOLS]

    visible: list[str] = []
    if "journal:read" in granted:
        visible.extend(t for t in tool_names if t in READ_TOOLS)
    if "journal:write" in granted:
        visible.extend(t for t in tool_names if t in WRITE_TOOLS)
    return visible


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


def _wire_scope_filter(mcp: FastMCP) -> None:
    """Wire scope filter into tools/list handler (defense in depth).

    Replaces the lowlevel handler set by FastMCP._setup_handlers() so
    that the ``tools/list`` response respects the token's scopes.
    ``mcp._mcp_server`` is a private attribute but stable across FastMCP
    versions; last call to ``list_tools()(handler)`` wins.
    """
    original_list_tools = mcp.list_tools  # bound async method

    async def _scope_filtered_list_tools() -> list[MCPTool]:
        all_tools = await original_list_tools()
        scopes = current_token_scopes.get() or frozenset()
        visible = set(filter_tools_by_scope([t.name for t in all_tools], scopes))
        return [t for t in all_tools if t.name in visible]

    mcp._mcp_server.list_tools()(_scope_filtered_list_tools)


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

    _wire_scope_filter(mcp)
