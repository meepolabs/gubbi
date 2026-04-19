"""MCP tools: journal_list_topics, journal_create_topic."""

from typing import Any

from mcp.server.fastmcp import FastMCP

from journalctl.core.context import AppContext
from journalctl.core.db_context import user_scoped_connection
from journalctl.core.validation import (
    sanitize_freetext,
    sanitize_label,
    validate_topic,
)
from journalctl.storage.repositories import topics as topic_repo
from journalctl.tools.constants import DEFAULT_TOPICS_LIMIT, MAX_TOPICS_RESULTS
from journalctl.tools.errors import already_exists, invalid_topic, validation_error


def register(mcp: FastMCP, app_ctx: AppContext) -> None:
    """Register topic tools on the MCP server."""

    @mcp.tool()
    async def journal_list_topics(
        topic_prefix: str | None = None,
        limit: int = DEFAULT_TOPICS_LIMIT,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Browse all journal topics — "what topics do I have?" or "what do I track?"

        Use when the user asks about their journal structure, or when you need to
        discover valid topic paths for other tools.

        Args:
            topic_prefix: Filter topics under this prefix (e.g. 'work').
                          If omitted, lists all topics.
            limit: Max topics to return (default 50).
            offset: Skip first N topics for pagination (default 0).

        Returns:
            List of topics with title, description, entry count, and created/updated dates.
        """
        limit = max(1, min(limit, MAX_TOPICS_RESULTS))
        offset = max(0, offset)
        if topic_prefix:
            topic_prefix = topic_prefix.rstrip("/") or None
        if topic_prefix:
            try:
                topic_prefix = validate_topic(topic_prefix)
            except ValueError as e:
                return invalid_topic(topic_prefix, str(e))
        async with user_scoped_connection(app_ctx.pool) as conn:
            page, total = await topic_repo.list_all(
                conn, topic_prefix=topic_prefix, limit=limit, offset=offset
            )
        return {
            "topics": [t.model_dump() for t in page],
            "total": total,
            "offset": offset,
            "limit": limit,
        }

    @mcp.tool()
    async def journal_create_topic(
        topic: str,
        title: str,
        description: str = "",
    ) -> dict[str, Any]:
        """Create a new journal topic for an area of the user's life not yet tracked.
        e.g., "I want to start tracking my fitness" or "make a topic for the house renovation."

        Required before writing entries or conversations to a new topic.
        Check journal_list_topics or the briefing first to avoid duplicates.
        Topic paths are permanent and cannot be renamed after creation — choose carefully.

        Args:
            topic: Topic path (e.g. 'hobbies/woodworking').
                   Max 2 levels, lowercase alphanumeric with hyphens.
            title: Human-readable title (max 100 characters).
            description: One-line description of this topic (max 500 characters).

        Returns:
            Confirmation with topic (path), topic_id, and normalized_from if
            the path was auto-corrected from the input.
        """
        original_topic = topic
        try:
            topic = validate_topic(topic)
        except ValueError as e:
            return invalid_topic(topic, str(e))
        title = sanitize_label(title, max_len=100)
        if not title:
            return validation_error("title cannot be empty after sanitization")
        if description:
            description = sanitize_freetext(description, max_len=500)
        try:
            async with user_scoped_connection(app_ctx.pool) as conn:
                topic_id = await topic_repo.create(
                    conn,
                    topic=topic,
                    title=title,
                    description=description,
                )
        except ValueError:
            return already_exists(topic)
        result: dict[str, Any] = {
            "status": "created",
            "topic": topic,
            "topic_id": topic_id,
        }
        if original_topic != topic:
            result["normalized_from"] = original_topic
        return result
