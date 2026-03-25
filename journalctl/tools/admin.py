"""MCP tool: journal_reindex."""

from mcp.server.fastmcp import FastMCP

from journalctl.storage.index import SearchIndex


def register(mcp: FastMCP, index: SearchIndex) -> None:
    """Register admin tools on the MCP server."""

    @mcp.tool()
    async def journal_reindex() -> dict:
        """Repair the search index — use when journal_search returns wrong or missing results.

        Only needed if search results seem stale or incomplete.
        Rarely needed during normal use.

        Returns:
            Number of documents indexed and duration.
        """
        result = index.rebuild()
        return {
            "status": "rebuilt",
            "documents_indexed": result["documents_indexed"],
            "duration_seconds": result["duration_seconds"],
        }
