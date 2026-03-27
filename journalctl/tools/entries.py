"""MCP tools: journal_append, journal_read, journal_update."""

from typing import Final, Literal

from mcp.server.fastmcp import FastMCP

from journalctl.models.entry import sanitize_freetext, sanitize_label, validate_date, validate_topic
from journalctl.storage.database import DatabaseStorage
from journalctl.storage.index import SearchIndex

_DEFAULT_ENTRIES: Final = 10


def register(
    mcp: FastMCP,
    storage: DatabaseStorage,
    index: SearchIndex,
) -> None:
    """Register entry tools on the MCP server."""

    @mcp.tool()
    async def journal_append(
        topic: str,
        content: str,
        context: str | None = None,
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
            content: Entry content in markdown. The headline — what happened.
            context: Optional reasoning, tradeoffs, or background. Loaded on demand,
                     not included in briefing. Use for decisions and key insights.
            tags: Optional tags (e.g. ['decision', 'milestone', 'important']).
            date: Date override as YYYY-MM-DD. Defaults to today.

        Returns:
            Confirmation with topic, date, entry_id, and entry count.
        """
        validate_topic(topic)
        content = sanitize_freetext(content)
        if context:
            context = sanitize_freetext(context)
        if tags:
            tags = [sanitize_label(t) for t in tags]
        if date:
            validate_date(date)

        entry_id, count = storage.append_entry(
            topic=topic,
            content=content,
            context=context,
            tags=tags,
            date=date,
        )

        # Update FTS5 index
        meta = storage.get_topic(topic)
        index.upsert_entry(
            entry_id=entry_id,
            topic=topic,
            title=meta.title if meta else topic,
            date=date or "today",
            content=content,
            context=context,
            tags=tags or [],
        )

        return {
            "status": "appended",
            "topic": topic,
            "date": date or "today",
            "entry_id": entry_id,
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
            Topic metadata and most recent entries (including context if set).
        """
        validate_topic(topic)
        count = n if n is not None else _DEFAULT_ENTRIES
        meta, entries = storage.read_entries(topic, n=count)

        return {
            "metadata": meta.model_dump(),
            "entries": [e.model_dump() for e in entries],
            "total_entries": meta.entry_count,
            "showing": len(entries),
        }

    @mcp.tool()
    async def journal_update(
        entry_id: int,
        content: str,
        context: str | None = None,
        mode: Literal["replace", "append"] = "replace",
    ) -> dict:
        """Correct or expand a journal entry — use when an entry has errors or needs more detail.

        Use journal_read first to get the entry's id field. Supports replacing the
        entire entry or appending additional content to it.

        Args:
            entry_id: Stable integer ID of the entry (shown in journal_read results).
            content: New content for the entry.
            context: Updated reasoning/context (optional).
            mode: 'replace' to overwrite entirely, 'append' to add to the existing entry.

        Returns:
            Confirmation with updated entry_id.
        """
        content = sanitize_freetext(content)
        if context:
            context = sanitize_freetext(context)

        storage.update_entry(
            entry_id=entry_id,
            content=content,
            context=context,
            mode=mode,
        )

        return {
            "status": "updated",
            "entry_id": entry_id,
            "mode": mode,
        }
