"""MCP tool: journal_search (unified FTS5 + semantic)."""

import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

from journalctl.core.validation import sanitize_freetext, validate_date, validate_topic
from journalctl.memory.client import MemoryServiceProtocol
from journalctl.models.search import SearchResult
from journalctl.storage.constants import SUMMARY_TRUNCATE_LEN
from journalctl.storage.database import DatabaseStorage
from journalctl.storage.search_index import SearchIndex
from journalctl.tools.constants import (
    DEFAULT_SEARCH_LIMIT,
    MAX_QUERY_LEN,
    MAX_SEARCH_RESULTS,
    MEMORY_HASH_PREVIEW_LEN,
)
from journalctl.tools.errors import invalid_date, invalid_topic, validation_error

logger = logging.getLogger(__name__)


def register(
    mcp: FastMCP,
    storage: DatabaseStorage,
    index: SearchIndex,
    memory_service: MemoryServiceProtocol,
) -> None:
    """Register search tool on the MCP server."""

    @mcp.tool()
    async def journal_search(
        query: str,
        topic_prefix: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = DEFAULT_SEARCH_LIMIT,
    ) -> dict[str, Any]:
        """Search your journal by keyword or meaning — the primary search tool.

        Handles both exact keyword lookups ("deployment Phase 7") and conceptual
        questions ("what car do I drive?", "what's my salary?"). Searches across
        all topics, entries, and conversations.

        Do NOT use for browsing a single topic — use journal_read_topic instead.
        Do NOT use for time-based browsing — use journal_timeline instead.

        Args:
            query: Search query — keywords, phrases, or natural language questions.
            topic_prefix: Filter to topics under this prefix (e.g. 'work').
                          Must be a valid topic path if provided.
            date_from: Filter entries on or after this date (YYYY-MM-DD).
            date_to: Filter entries on or before this date (YYYY-MM-DD).
            limit: Maximum results (default 10).

        Returns:
            List of matching results with snippets, relevance scores,
            and entry_id/conversation_id for follow-up calls.
        """
        limit = max(1, min(limit, MAX_SEARCH_RESULTS))
        if len(query) > MAX_QUERY_LEN:
            return validation_error(
                f"Query too long: max {MAX_QUERY_LEN} characters, got {len(query)}"
            )

        if topic_prefix:
            topic_prefix = topic_prefix.rstrip("/") or None
        if topic_prefix:
            try:
                topic_prefix = validate_topic(topic_prefix)
            except ValueError as e:
                return invalid_topic(topic_prefix, str(e))
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

        # Run FTS5 keyword search
        try:
            fts_results = index.search(
                query=query,
                topic_prefix=topic_prefix,
                date_from=date_from,
                date_to=date_to,
                limit=limit,
            )
        except ValueError as e:
            return validation_error(str(e))

        # Run semantic search in parallel
        semantic_results: list[SearchResult] = []
        semantic_available = True
        try:
            sanitized_query = sanitize_freetext(query)
            mem_response = await memory_service.retrieve_memories(
                query=sanitized_query,
                n_results=limit,
            )
            # Convert memory results to SearchResult format
            memories = mem_response.get("memories", [])
            if isinstance(memories, list):
                for mem in memories:
                    if isinstance(mem, dict):
                        content = mem.get("content", "")
                        metadata = mem.get("metadata", {}) or {}
                        entry_id = metadata.get("entry_id")
                        similarity = float(mem.get("similarity_score", 0.0))
                        src_key = f"memory:{mem.get('content_hash', '')[:MEMORY_HASH_PREVIEW_LEN]}"
                        semantic_results.append(
                            SearchResult(
                                source_key=src_key,
                                doc_type="entry" if entry_id else "memory",
                                topic=metadata.get("topic", ""),
                                title=metadata.get("title", ""),
                                snippet=content[:SUMMARY_TRUNCATE_LEN],
                                rank=-similarity if similarity else 0.0,
                                date=metadata.get("date", ""),
                                entry_id=int(entry_id) if entry_id else None,
                                conversation_id=None,
                            )
                        )
        except Exception:
            logger.warning("Semantic search failed, using FTS5 only", exc_info=True)
            semantic_available = False

        # Filter orphaned semantic results (deleted entries still in vector index)
        semantic_entry_ids = {r.entry_id for r in semantic_results if r.entry_id}
        active_ids = storage.get_active_entry_ids(semantic_entry_ids)
        semantic_results = [
            r for r in semantic_results if not r.entry_id or r.entry_id in active_ids
        ]

        # Enrich semantic results with DB metadata (topic, date, title)
        enrichable_ids = {r.entry_id for r in semantic_results if r.entry_id}
        if enrichable_ids:
            brief = storage.get_entries_brief(enrichable_ids)
            for r in semantic_results:
                if r.entry_id and r.entry_id in brief:
                    meta = brief[r.entry_id]
                    if not r.topic:
                        r.topic = meta["topic"]
                    if not r.date:
                        r.date = meta["date"]
                    if not r.title:
                        r.title = meta["title"]

        # Post-filter semantic results by topic and date (memory service
        # doesn't support these filters natively, so we apply them here
        # after enrichment has filled in the DB metadata).
        if topic_prefix:
            semantic_results = [
                r
                for r in semantic_results
                if r.topic == topic_prefix or r.topic.startswith(topic_prefix + "/")
            ]
        if date_from:
            semantic_results = [r for r in semantic_results if r.date >= date_from]
        if date_to:
            semantic_results = [r for r in semantic_results if r.date <= date_to]

        # Merge and deduplicate
        seen_ids: set[str] = set()
        merged: list[SearchResult] = []

        for r in fts_results:
            key = r.source_key
            if key not in seen_ids:
                seen_ids.add(key)
                merged.append(r)

        for r in semantic_results:
            # Deduplicate by entry_id if it matches an FTS5 result
            if r.entry_id:
                entry_key = f"entry:{r.entry_id}"
                if entry_key in seen_ids:
                    continue
                seen_ids.add(entry_key)
            merged.append(r)

        # Sort by rank (lower is better in FTS5), limit
        merged.sort(key=lambda r: r.rank)
        merged = merged[:limit]

        return {
            "results": [r.model_dump() for r in merged],
            "total": len(merged),
            "query": query,
            "semantic_available": semantic_available,
        }
