"""Register all MCP tools on the FastMCP server."""

from mcp.server.fastmcp import FastMCP

from journalctl.core.context import AppContext
from journalctl.tools import (
    context,
    conversations,
    entries,
    search,
    topics,
)


def register_tools(mcp: FastMCP, app_ctx: AppContext) -> None:
    """Register all MCP tools. Semantic memory is internal — not exposed to LLM.

    Args:
        mcp: FastMCP server instance.
        app_ctx: Application context (pool, embedding_service, settings, logger).
    """
    topics.register(mcp, app_ctx)
    entries.register(mcp, app_ctx)
    search.register(mcp, app_ctx)
    conversations.register(mcp, app_ctx)
    context.register(mcp, app_ctx)
