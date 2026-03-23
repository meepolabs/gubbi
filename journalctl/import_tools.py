"""Register all MCP tools on the FastMCP server."""

from mcp.server.fastmcp import FastMCP

from journalctl.config import Settings
from journalctl.storage.index import SearchIndex
from journalctl.storage.markdown import MarkdownStorage
from journalctl.tools import (
    admin,
    context,
    conversations,
    entries,
    search,
    topics,
)


def import_tools(
    mcp: FastMCP,
    storage: MarkdownStorage,
    index: SearchIndex,
    settings: Settings,
) -> None:
    """Register all 12 MCP tools.

    Args:
        mcp: FastMCP server instance.
        storage: Markdown storage layer.
        index: FTS5 search index.
        settings: Application settings.
    """
    topics.register(mcp, storage)
    entries.register(mcp, storage, index)
    search.register(mcp, index)
    conversations.register(mcp, storage, index)
    context.register(mcp, storage, index, settings)
    admin.register(mcp, index)
