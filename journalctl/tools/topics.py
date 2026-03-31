"""MCP tools: journal_list_topics, journal_create_topic."""

from typing import Any

from mcp.server.fastmcp import FastMCP

from journalctl.core.validation import (
    sanitize_freetext,
    sanitize_label,
    validate_date,
    validate_topic,
)
from journalctl.storage.database import DatabaseStorage
from journalctl.tools.errors import already_exists, invalid_date, invalid_topic


def register(mcp: FastMCP, storage: DatabaseStorage) -> None:
    """Register topic tools on the MCP server."""

    @mcp.tool()
    async def journal_list_topics(
        topic_prefix: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Browse all journal topics — "what topics do I have?" or "what do I track?"

        Use when the user asks about their journal structure, or when you need to
        discover valid topic paths for other tools.

        Args:
            topic_prefix: Filter topics under this prefix (e.g. 'work').
            limit: Max topics to return (default 50).
            offset: Skip first N topics for pagination.

        Returns:
            List of topics with title, description, tags,
            entry count, and created/updated dates.
        """
        if topic_prefix:
            topic_prefix = topic_prefix.rstrip("/") or None
        if topic_prefix:
            try:
                validate_topic(topic_prefix)
            except ValueError as e:
                return invalid_topic(topic_prefix, str(e))
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
        created_at: str | None = None,
    ) -> dict[str, Any]:
        """Create a new journal topic for an area of the user's life not yet tracked.
        e.g., "I want to start tracking my fitness" or "make a topic for the house renovation."

        Required before writing entries or conversations to a new topic.
        Check journal_list_topics or the briefing first to avoid duplicates.

        Args:
            topic: Topic path (e.g. 'hobbies/woodworking').
                   Max 2 levels, lowercase alphanumeric with hyphens.
            title: Human-readable title.
            description: One-line description of this topic.
            tags: Initial tags.
            created_at: Optional creation date (ISO 8601 format).

        Returns:
            Confirmation with the created topic path.
        """
        try:
            validate_topic(topic)
        except ValueError as e:
            return invalid_topic(topic, str(e))
        title = sanitize_label(title, max_len=100)
        if description:
            description = sanitize_freetext(description, max_len=500)
        if tags:
            tags = [sanitize_label(t) for t in tags]
        if created_at:
            try:
                validate_date(created_at)
            except ValueError:
                return invalid_date(created_at)
        try:
            topic_id = storage.create_topic(
                topic=topic,
                title=title,
                description=description,
                tags=tags,
                created_at=created_at,
            )
        except ValueError:
            return already_exists(topic)
        return {
            "status": "created",
            "topic": topic,
            "topic_id": topic_id,
        }
