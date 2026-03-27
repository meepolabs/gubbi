"""Register all MCP tools on the FastMCP server."""

from mcp.server.fastmcp import FastMCP

from journalctl.config import Settings
from journalctl.memory.protocol import MemoryServiceProtocol
from journalctl.storage.database import DatabaseStorage
from journalctl.storage.index import SearchIndex
from journalctl.tools import (
    admin,
    context,
    conversations,
    entries,
    memory,
    search,
    topics,
)


def import_tools(
    mcp: FastMCP,
    storage: DatabaseStorage,
    index: SearchIndex,
    settings: Settings,
    memory_service: MemoryServiceProtocol | None = None,
) -> None:
    """Register MCP tools. Memory tools registered only if memory_service is provided.

    Args:
        mcp: FastMCP server instance.
        storage: Canonical database storage layer.
        index: FTS5 search index.
        settings: Application settings.
        memory_service: Optional MemoryService instance for semantic memory tools.
    """
    topics.register(mcp, storage)
    entries.register(mcp, storage, index)
    search.register(mcp, index)
    conversations.register(mcp, storage, index)
    context.register(mcp, storage, index, settings)
    admin.register(mcp, index, storage)

    if memory_service is not None:
        memory.register(mcp, memory_service)
