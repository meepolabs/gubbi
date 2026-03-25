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
        """Record something in the user's journal — call proactively alongside memory_store.

        When the user shares updates, progress, reflections, events, or decisions,
        write a journal entry immediately. This is the dated, contextual record
        (what happened and when), while memory_store captures the distilled fact.
        Use both together: journal_append for the narrative, memory_store for the takeaway.

        Creates the topic automatically if it doesn't exist yet.

        Args:
            topic: Topic path (e.g. 'work/acme', 'health', 'hobbies/woodworking').
            content: Entry content in markdown.
            tags: Optional tags (e.g. ['decision', 'milestone', 'important']).
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
        """Read recent entries from a journal topic — "show me my notes on X."

        Use when the user wants to review what they've written about a specific topic.
        For searching across all topics by keyword, use journal_search instead.

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
        """Correct or expand a journal entry — use when an entry has errors or needs more detail.

        Use journal_read first to find the entry's index. Supports replacing the
        entire entry or appending additional content to it.

        Args:
            topic: Topic path (e.g. 'work/acme').
            entry_index: 1-based position of the entry (shown by journal_read).
            content: New content for the entry.
            mode: 'replace' to overwrite entirely, 'append' to add to the existing entry.

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
