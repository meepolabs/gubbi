"""MCP tools: journal_list_topics, journal_create_topic."""

from mcp.server.fastmcp import FastMCP

from journalctl.models.entry import validate_topic
from journalctl.storage.markdown import MarkdownStorage


def register(mcp: FastMCP, storage: MarkdownStorage) -> None:
    """Register topic tools on the MCP server."""

    @mcp.tool()
    async def journal_list_topics(
        prefix: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict:
        """List all journal topics with metadata.

        Args:
            prefix: Filter topics under this prefix (e.g. 'work/').
            limit: Max topics to return (default 50).
            offset: Skip first N topics for pagination.

        Returns:
            List of topics with title, description, tags,
            entry_count, created, and updated dates.
        """
        topics = storage.list_topics(prefix=prefix)
        page = topics[offset : offset + limit]
        return {
            "topics": [t.model_dump() for t in page],
            "total": len(topics),
            "offset": offset,
            "limit": limit,
        }

    @mcp.tool()
    async def journal_create_topic(
        topic: str,
        title: str,
        description: str = "",
        tags: list[str] | None = None,
    ) -> dict:
        """Create a new journal topic with metadata.

        Args:
            topic: Topic path (e.g. 'hobbies/woodworking').
                   Max 2 levels, lowercase alphanumeric with hyphens.
            title: Human-readable title.
            description: One-line description of this topic.
            tags: Initial tags.

        Returns:
            Confirmation with the created file path.
        """
        validate_topic(topic)
        path = storage.create_topic(
            topic=topic,
            title=title,
            description=description,
            tags=tags,
        )
        return {
            "status": "created",
            "topic": topic,
            "path": str(path),
        }
