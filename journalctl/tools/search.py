"""MCP tool: journal_search."""

from mcp.server.fastmcp import FastMCP

from journalctl.storage.index import SearchIndex


def register(mcp: FastMCP, index: SearchIndex) -> None:
    """Register search tool on the MCP server."""

    @mcp.tool()
    async def journal_search(
        query: str,
        topic_prefix: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 10,
    ) -> dict:
        """Search across all journal entries and conversations.

        Uses FTS5 full-text search with keyword matching.

        Args:
            query: Search query. Supports FTS5 syntax:
                   AND, OR, NOT, "exact phrase".
            topic_prefix: Filter to topics under this prefix.
            date_from: Filter from this date (YYYY-MM-DD).
            date_to: Filter to this date (YYYY-MM-DD).
            limit: Maximum results (default 10).

        Returns:
            List of matching results with snippets and scores.
        """
        results = index.search(
            query=query,
            topic_prefix=topic_prefix,
            date_from=date_from,
            date_to=date_to,
            limit=limit,
        )
        return {
            "results": [r.model_dump() for r in results],
            "count": len(results),
            "query": query,
        }
