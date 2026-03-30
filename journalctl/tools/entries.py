"""MCP tools: journal_append, journal_read, journal_update, journal_delete."""

import hashlib
import json
import logging
import sqlite3
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP

from journalctl.core.validation import (
    sanitize_freetext,
    sanitize_label,
    validate_date,
    validate_topic,
)
from journalctl.memory.client import MemoryServiceProtocol
from journalctl.storage.database import DatabaseStorage
from journalctl.storage.search_index import SearchIndex
from journalctl.tools.constants import DEFAULT_ENTRIES_LIMIT, MAX_READ_ENTRIES
from journalctl.tools.errors import invalid_date, invalid_topic, not_found, validation_error

logger = logging.getLogger(__name__)


def register(
    mcp: FastMCP,
    storage: DatabaseStorage,
    index: SearchIndex,
    memory_service: MemoryServiceProtocol,
) -> None:
    """Register entry tools on the MCP server."""

    async def _embed_entry(entry_id: int, content: str, tags: list[str] | None = None) -> bool:
        """Store an embedding for a journal entry. Returns True on success.
        Internal — not exposed to LLM."""
        try:
            await memory_service.store_memory(
                content=content,
                tags=tags,
                metadata={"entry_id": entry_id, "source": "journal_entry"},
            )
            return True
        except Exception as e:
            logger.warning("Failed to embed entry %s: %s", entry_id, e, exc_info=True)
            return False

    async def _remove_embedding(content: str) -> None:
        """Remove an embedding by content hash. Internal — best-effort."""
        try:
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            await memory_service.delete_memory(content_hash=content_hash)
        except Exception:
            logger.warning("Could not remove embedding for content hash", exc_info=True)

    @mcp.tool()
    async def journal_append(
        topic: str,
        content: str,
        context: str | None = None,
        tags: list[str] | None = None,
        date: str | None = None,
    ) -> dict[str, Any]:
        """Record something in the user's journal.

        When the user shares updates, progress, reflections, events, or decisions,
        write a journal entry immediately. This is the dated event record (what happened
        and when).

        Example: User says "We decided to use PostgreSQL instead of MongoDB."
        → journal_append(topic="projects/alpha", content="Chose PostgreSQL for the database",
              context="Mongo had no ACID transactions, team already knows SQL")

        Auto-creates the topic if it doesn't exist. Reuse existing topic paths
        from the briefing to avoid duplicates (e.g. use 'work/meeting-notes'
        if it already exists, don't create 'work/meetings').

        Do NOT use for searching or reading — use journal_search or journal_read.

        Args:
            topic: Topic path (e.g. 'work/acme', 'health', 'hobbies/woodworking').
            content: What happened — the headline. Shown in briefing and timeline.
            context: Why it happened — reasoning or tradeoffs. Use for decisions
                     and key insights; leave empty for routine events.
            tags: Optional tags (e.g. ['decision', 'milestone', 'important']).
            date: Date override as YYYY-MM-DD. Defaults to today.

        Returns:
            Confirmation with topic, date, entry_id, and entry count.
        """

        try:
            validate_topic(topic)
        except ValueError as e:
            return invalid_topic(topic, str(e))
        content = sanitize_freetext(content)
        if not content.strip():
            return validation_error("Content cannot be empty")
        if context:
            context = sanitize_freetext(context)
        if tags:
            tags = [sanitize_label(t) for t in tags]
        if date:
            try:
                validate_date(date)
            except ValueError:
                return invalid_date(date)

        entry_id, count = storage.append_entry(
            topic=topic,
            content=content,
            context=context,
            tags=tags,
            date=date,
            commit=False,  # deferred — commit after FTS write for atomicity
        )

        # Write FTS5 index on the same connection, then commit both atomically
        meta = storage.get_topic(topic)
        try:
            index.upsert_entry_on_conn(
                storage.conn,
                entry_id=entry_id,
                topic=topic,
                title=meta.title if meta else topic,
                date=date or "today",
                content=content,
                context=context,
                tags=tags or [],
            )
            storage.conn.commit()
        except sqlite3.Error:
            storage.conn.rollback()
            raise

        # Auto-embed for semantic search; stamp indexed_at on success
        if await _embed_entry(entry_id, content, tags):
            storage.mark_entry_indexed(entry_id)

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
        date_from: str | None = None,
        date_to: str | None = None,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Read entries from a topic — "show me my notes on health" or
        "what did I write about work?"

        Use when the user wants to review a specific topic's entries.
        Returns entries in chronological order with content and context.

        Do NOT use for keyword search across topics — use journal_search instead.
        Do NOT use for time-based browsing — use journal_timeline instead.

        Args:
            topic: Topic path (e.g. 'work/acme').
            n: Max entries to return (default 10). Use a large number for more history.
            date_from: Only entries on or after this date (YYYY-MM-DD).
            date_to: Only entries on or before this date (YYYY-MM-DD).
            offset: Skip first N entries for pagination (default 0).

        Returns:
            Topic metadata and entries (including context if set).
        """

        try:
            validate_topic(topic)
        except ValueError as e:
            return invalid_topic(topic, str(e))
        if date_from:
            try:
                validate_date(date_from)
            except ValueError:
                return invalid_date(date_from)
        if date_to:
            try:
                validate_date(date_to)
            except ValueError:
                return invalid_date(date_to)
        if n is not None and n <= 0:
            return validation_error(f"n must be a positive integer, got {n}")
        if n is not None and n > MAX_READ_ENTRIES:
            return validation_error(f"n cannot exceed {MAX_READ_ENTRIES}, got {n}")
        count = n if n is not None else DEFAULT_ENTRIES_LIMIT
        try:
            meta, entries, total = storage.read_entries(
                topic,
                n=count,
                date_from=date_from,
                date_to=date_to,
                offset=offset,
            )
        except FileNotFoundError:
            return not_found("Topic", topic)

        return {
            "metadata": meta.model_dump(),
            "entries": [e.model_dump() for e in entries],
            "total": total,
            "limit": count,
            "offset": offset,
        }

    @mcp.tool()
    async def journal_update(
        entry_id: int,
        content: str | None = None,
        context: str | None = None,
        mode: Literal["replace", "append"] = "replace",
        date: str | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Correct or expand a journal entry — "fix that entry" or "add more detail."

        Use the entry's 'id' from journal_read, journal_search, or journal_timeline results.

        Do NOT use to remove an entry — use journal_delete instead.

        Args:
            entry_id: The entry's 'id' (from read, search, or timeline results).
            content: New content for the entry (optional — omit to only change date/tags).
            context: Updated reasoning/context (optional).
            mode: 'replace' to overwrite content, 'append' to add to existing entry.
            date: Correct the entry's date (YYYY-MM-DD). Omit to keep current date.
            tags: Replace the entry's tags. Omit to keep current tags.

        Returns:
            Confirmation with updated entry_id.
        """
        if content:
            content = sanitize_freetext(content)
        if context:
            context = sanitize_freetext(context)
        if date:
            try:
                validate_date(date)
            except ValueError:
                return invalid_date(date)
        if tags:
            tags = [sanitize_label(t) for t in tags]

        try:
            storage.update_entry(
                entry_id=entry_id,
                content=content,
                context=context,
                mode=mode,
                date=date,
                tags=tags,
            )
        except IndexError:
            return not_found("Entry", entry_id)

        # Re-index (read back the updated entry for current content)
        row = storage.get_entry_with_topic(entry_id)
        if row:
            index.upsert_entry(
                entry_id=entry_id,
                topic=row["path"],
                title=row["title"],
                date=row["date"],
                content=row["content"],
                context=row["context"],
                tags=json.loads(row["tags"] or "[]"),
            )
            # Update semantic embedding; stamp indexed_at on success
            await _remove_embedding(row["content"])
            if await _embed_entry(entry_id, row["content"], json.loads(row["tags"] or "[]")):
                storage.mark_entry_indexed(entry_id)

        return {
            "status": "updated",
            "entry_id": entry_id,
            "mode": mode,
        }

    @mcp.tool()
    async def journal_delete(
        entry_id: int,
    ) -> dict[str, Any]:
        """Remove a journal entry — wrong data, duplicate, or mistake.

        Use the entry's 'id' from journal_read, journal_search, or journal_timeline results.

        Do NOT use to correct an entry — use journal_update instead.

        Args:
            entry_id: The entry's 'id' (from read, search, or timeline results).

        Returns:
            Confirmation with deleted entry_id.
        """
        # Read content before delete for embedding cleanup
        content_before_delete = storage.get_entry_content(entry_id)

        try:
            storage.delete_entry(entry_id)
        except IndexError:
            return not_found("Entry", entry_id)

        # Remove from FTS5 index
        index.remove_entry(entry_id)

        # Remove semantic embedding
        if content_before_delete:
            await _remove_embedding(content_before_delete)

        return {
            "status": "deleted",
            "entry_id": entry_id,
        }
