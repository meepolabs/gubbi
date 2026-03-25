"""MCP tools: semantic memory store, retrieve, search, list, delete, health."""

import re
from typing import Any, cast

from mcp.server.fastmcp import FastMCP

from journalctl.models.entry import sanitize_freetext, sanitize_label

# Valid SHA256 hex string (64 lowercase hex chars)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_MAX_N_RESULTS = 100
_MAX_TAGS = 50


def _sanitize_tags(tags: list[str] | None) -> list[str] | None:
    if tags is None:
        return None
    return [sanitize_label(t, max_len=50) for t in tags[:_MAX_TAGS]]


def register(mcp: FastMCP, memory_service: Any) -> None:
    """Register memory tools on the MCP server.

    Args:
        mcp: FastMCP server instance.
        memory_service: MemoryService instance from mcp-memory-service.
    """

    @mcp.tool()
    async def memory_store(
        content: str,
        tags: list[str] | None = None,
        memory_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Store a new semantic memory.

        Use this to save facts, preferences, decisions, and patterns for
        quick recall. Unlike the journal (which is a full historical record),
        memory stores distilled knowledge that answers "what is" questions.

        Args:
            content: The memory content to store.
            tags: Optional tags for categorization (e.g. ['project-x', 'decision']).
            memory_type: Optional type (e.g. 'fact', 'preference', 'experience', 'decision').
            metadata: Optional additional key-value metadata.

        Returns:
            Confirmation with content hash and stored memory details.
        """
        content = sanitize_freetext(content)
        tags = _sanitize_tags(tags)
        if memory_type is not None:
            memory_type = sanitize_label(memory_type)
        return cast(
            dict[str, Any],
            await memory_service.store_memory(
                content=content,
                tags=tags,
                memory_type=memory_type,
                metadata=metadata,
            ),
        )

    @mcp.tool()
    async def memory_retrieve(
        query: str,
        n_results: int = 10,
        tags: list[str] | None = None,
        memory_type: str | None = None,
    ) -> dict[str, Any]:
        """Retrieve memories by semantic similarity search.

        Finds memories semantically related to the query using vector embeddings.
        Unlike journal_search (keyword/FTS5), this finds conceptually similar
        content even without exact word matches.

        Args:
            query: Natural language search query.
            n_results: Maximum number of results to return (default 10, max 100).
            tags: Optional tag filter — only return memories with these tags.
            memory_type: Optional type filter.

        Returns:
            List of matching memories with similarity scores.
        """
        query = sanitize_freetext(query)
        tags = _sanitize_tags(tags)
        if memory_type is not None:
            memory_type = sanitize_label(memory_type)
        n_results = max(1, min(n_results, _MAX_N_RESULTS))
        return cast(
            dict[str, Any],
            await memory_service.retrieve_memories(
                query=query,
                n_results=n_results,
                tags=tags,
                memory_type=memory_type,
            ),
        )

    @mcp.tool()
    async def memory_search_by_tag(
        tags: list[str] | str,
        match_all: bool = False,
    ) -> dict[str, Any]:
        """Search memories by tags.

        Args:
            tags: Tag or list of tags to search for.
            match_all: If True, memory must have ALL tags. If False (default), ANY tag matches.

        Returns:
            List of matching memories.
        """
        if isinstance(tags, str):
            tags = sanitize_label(tags)
        else:
            tags = [sanitize_label(t) for t in tags[:_MAX_TAGS]]
        return cast(
            dict[str, Any],
            await memory_service.search_by_tag(tags=tags, match_all=match_all),
        )

    @mcp.tool()
    async def memory_list(
        page: int = 1,
        page_size: int = 10,
        tag: str | None = None,
        memory_type: str | None = None,
    ) -> dict[str, Any]:
        """List memories with pagination.

        Args:
            page: Page number, 1-based (default 1).
            page_size: Number of memories per page (default 10, max 100).
            tag: Filter by a specific tag.
            memory_type: Filter by memory type.

        Returns:
            Paginated list of memories with total count.
        """
        page = max(1, page)
        page_size = max(1, min(page_size, _MAX_N_RESULTS))
        if tag is not None:
            tag = sanitize_label(tag)
        if memory_type is not None:
            memory_type = sanitize_label(memory_type)
        return cast(
            dict[str, Any],
            await memory_service.list_memories(
                page=page,
                page_size=page_size,
                tag=tag,
                memory_type=memory_type,
            ),
        )

    @mcp.tool()
    async def memory_delete(content_hash: str) -> dict[str, Any]:
        """Delete a memory by its content hash.

        The content hash is returned by memory_store and memory_retrieve.

        Args:
            content_hash: SHA256 content hash of the memory to delete (64 hex chars).

        Returns:
            Confirmation of deletion or error message.
        """
        if not _SHA256_PATTERN.match(content_hash):
            return {
                "success": False,
                "error": "Invalid content_hash: must be a 64-char hex SHA256.",
            }
        return cast(
            dict[str, Any],
            await memory_service.delete_memory(content_hash=content_hash),
        )

    @mcp.tool()
    async def memory_health() -> dict[str, Any]:
        """Check memory service health and statistics.

        Returns:
            Health status, total memory count, storage backend info.
        """
        return cast(dict[str, Any], await memory_service.health_check())
