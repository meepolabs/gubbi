"""MCP tool: journal_reindex."""

from mcp.server.fastmcp import FastMCP

from journalctl.storage.index import SearchIndex


def register(mcp: FastMCP, index: SearchIndex) -> None:
    """Register admin tools on the MCP server."""

    @mcp.tool()
    async def journal_reindex() -> dict:
        """Rebuild the FTS5 search index from markdown files.

        Use when search results seem wrong or after manual
        edits to markdown files outside the MCP server.

        Returns:
            Number of documents indexed and duration.
        """
        result = index.rebuild()
        return {
            "status": "rebuilt",
            "documents_indexed": result["documents_indexed"],
            "duration_seconds": result["duration_seconds"],
        }
