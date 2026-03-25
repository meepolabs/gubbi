"""MCP tool: journal_search."""

from mcp.server.fastmcp import FastMCP

from journalctl.models.entry import validate_topic
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
        """Search journal entries and conversations by keyword or date range.

        Use for specific, date-scoped, or keyword lookups — e.g. "what did I write
        about X last month." For vague or conceptual personal questions, prefer
        memory_retrieve instead (it matches by meaning, not keywords).

        Args:
            query: Search query. Supports: AND, OR, NOT, "exact phrase".
            topic_prefix: Filter to topics under this prefix (e.g. 'work').
            date_from: Filter entries on or after this date (YYYY-MM-DD).
            date_to: Filter entries on or before this date (YYYY-MM-DD).
            limit: Maximum results (default 10).

        Returns:
            List of matching results with snippets and relevance scores.
        """
        if topic_prefix:
            topic_prefix = topic_prefix.rstrip("/") or None
        if topic_prefix:
            validate_topic(topic_prefix)
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
