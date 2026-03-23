"""MCP tools: journal_append, journal_read, journal_update."""

from typing import Final, Literal

from mcp.server.fastmcp import FastMCP

from journalctl.models.entry import sanitize_freetext, sanitize_label, validate_date, validate_topic
from journalctl.storage.index import SearchIndex
from journalctl.storage.markdown import MarkdownStorage

_DEFAULT_ENTRIES: Final = 10


def register(
    mcp: FastMCP,
    storage: MarkdownStorage,
    index: SearchIndex,
) -> None:
    """Register entry tools on the MCP server."""

    @mcp.tool()
    async def journal_append(
        topic: str,
        content: str,
        tags: list[str] | None = None,
        date: str | None = None,
    ) -> dict:
        """Add a dated entry to a journal topic.

        Creates the topic file if it doesn't exist.

        Args:
            topic: Topic path (e.g. 'work/acme').
            content: Entry content in markdown.
            tags: Inline tags (e.g. ['decision', 'important']).
            date: Date override as YYYY-MM-DD. Defaults to today.

        Returns:
            Confirmation with topic, date, and entry count.
        """
        validate_topic(topic)
        content = sanitize_freetext(content)
        if tags:
            tags = [sanitize_label(t) for t in tags]
        if date:
            validate_date(date)
        path, count = storage.append_entry(
            topic=topic,
            content=content,
            tags=tags,
            date=date,
        )
        index.upsert_file(path)
        return {
            "status": "appended",
            "topic": topic,
            "date": date or "today",
            "entry_count": count,
        }

    @mcp.tool()
    async def journal_read(
        topic: str,
        n: int | None = None,
    ) -> dict:
        """Read a journal topic.

        Args:
            topic: Topic path (e.g. 'work/acme').
            n: Number of most recent entries to return (default 10).
               Use a large number to retrieve more history.

        Returns:
            Topic metadata and most recent entries.
        """
        validate_topic(topic)
        meta, body = storage.read_topic(topic)
        entries = storage.parse_entries(body)
        count = n if n is not None else _DEFAULT_ENTRIES
        selected = entries[-count:] if count > 0 else []

        return {
            "metadata": meta.model_dump(),
            "entries": [e.model_dump() for e in selected],
            "total_entries": len(entries),
            "showing": len(selected),
        }

    @mcp.tool()
    async def journal_update(
        topic: str,
        entry_index: int,
        content: str,
        mode: Literal["replace", "append"] = "replace",
    ) -> dict:
        """Edit a past entry by its index (1-based).

        Args:
            topic: Topic path.
            entry_index: 1-based position of the entry in the topic.
                         Use journal_read to see entry indexes.
            content: New content for the entry.
            mode: 'replace' to overwrite, 'append' to add to entry.

        Returns:
            Confirmation with updated entry index.
        """
        validate_topic(topic)
        content = sanitize_freetext(content)
        path = storage.update_entry(
            topic=topic,
            entry_index=entry_index,
            content=content,
            mode=mode,
        )
        index.upsert_file(path)
        return {
            "status": "updated",
            "topic": topic,
            "entry_index": entry_index,
            "mode": mode,
        }
