"""Register all MCP tools on the FastMCP server."""

from mcp.server.fastmcp import FastMCP

from journalctl.config import Settings
from journalctl.memory.client import MemoryServiceProtocol
from journalctl.storage.database import DatabaseStorage
from journalctl.storage.search_index import SearchIndex
from journalctl.tools import (
    admin,
    context,
    conversations,
    entries,
    search,
    topics,
)


def register_tools(
    mcp: FastMCP,
    storage: DatabaseStorage,
    index: SearchIndex,
    settings: Settings,
    memory_service: MemoryServiceProtocol,
) -> None:
    """Register MCP tools. Memory service is used internally — not exposed to LLM.

    Args:
        mcp: FastMCP server instance.
        storage: Canonical database storage layer.
        index: FTS5 search index.
        settings: Application settings.
        memory_service: MemoryService for semantic embedding/retrieval.
    """
    topics.register(mcp, storage)
    entries.register(mcp, storage, index, memory_service=memory_service)
    search.register(mcp, storage, index, memory_service=memory_service)
    conversations.register(mcp, storage, index)
    context.register(mcp, storage, index, settings, memory_service=memory_service)
    admin.register(mcp, index, storage, memory_service=memory_service)
