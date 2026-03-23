"""MCP tools: journal_list_topics, journal_create_topic."""

from mcp.server.fastmcp import FastMCP

from journalctl.models.entry import sanitize_freetext, sanitize_label, validate_topic
from journalctl.storage.markdown import MarkdownStorage


def register(mcp: FastMCP, storage: MarkdownStorage) -> None:
    """Register topic tools on the MCP server."""

    @mcp.tool()
    async def journal_list_topics(
        topic_prefix: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict:
        """List all journal topics with metadata.

        Args:
            topic_prefix: Filter topics under this prefix (e.g. 'work').
            limit: Max topics to return (default 50).
            offset: Skip first N topics for pagination.

        Returns:
            List of topics with title, description, tags,
            entry_count, created, and updated dates.
        """
        if topic_prefix:
            topic_prefix = topic_prefix.rstrip("/") or None
        if topic_prefix:
            validate_topic(topic_prefix)
        topics = storage.list_topics(topic_prefix=topic_prefix)
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
        title = sanitize_label(title, max_len=100)
        if description:
            description = sanitize_freetext(description, max_len=500)
        if tags:
            tags = [sanitize_label(t) for t in tags]
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
