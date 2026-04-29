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
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from journalctl.core.context import AppContext
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
