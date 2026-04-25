"""MCP tool: journal_search (unified tsvector FTS + pgvector semantic)."""

import asyncio
import logging
from datetime import date as date_cls
from typing import Any

from mcp.server.fastmcp import FastMCP

from journalctl.core.context import AppContext
from journalctl.core.db_context import user_scoped_connection
from journalctl.core.validation import validate_date, validate_topic
from journalctl.models.search import SearchResult
from journalctl.storage.constants import SUMMARY_TRUNCATE_LEN
from journalctl.storage.repositories import search as search_repo
from journalctl.tools.constants import (
    DEFAULT_SEARCH_LIMIT,
    MAX_QUERY_LEN,
    MAX_SEARCH_RESULTS,
)
from journalctl.tools.errors import invalid_date, invalid_topic, validation_error

logger = logging.getLogger(__name__)


def register(mcp: FastMCP, app_ctx: AppContext) -> None:
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
                          If omitted, searches all topics.
                          Must be a valid topic path if provided.
            date_from: Filter entries on or after this date (YYYY-MM-DD).
            date_to: Filter entries on or before this date (YYYY-MM-DD).
            limit: Maximum results (default 10).

        Returns:
            List of matching results with snippets, ordered by relevance
            (best first). Each result includes entry_id/conversation_id
            for follow-up calls.
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

        # Encode query before acquiring a DB connection — ONNX inference is CPU-bound
        # and holds the thread for 10-200ms. Encoding first keeps the pool free.
        fts_results: list[SearchResult] = []
        semantic_results: list[SearchResult] = []

        try:
            query_embedding = await asyncio.to_thread(app_ctx.embedding_service.encode, query)
        except Exception:
            logger.warning("Query encoding failed, semantic search disabled", exc_info=True)
            query_embedding = None

        df = date_cls.fromisoformat(date_from) if date_from else None
        dt = date_cls.fromisoformat(date_to) if date_to else None

        async with user_scoped_connection(app_ctx.pool) as conn:
            # FTS search — topic prefix and dates filtered in SQL
            fts_results = await search_repo.fts_search(
                conn, query, topic_prefix, date_from, date_to, limit
            )

            # Semantic search — topic prefix and dates pushed into SQL, no pre-query needed
            if query_embedding is not None:
                try:
                    raw = await app_ctx.embedding_service.search_by_vector(
                        conn,
                        query_embedding,
                        limit=limit,
                        topic_prefix=topic_prefix,
                        date_from=df,
                        date_to=dt,
                    )
                    for r in raw:
                        entry_id = r.get("entry_id")
                        semantic_results.append(
                            SearchResult(
                                source_key=f"entry:{entry_id}",
                                doc_type="entry",
                                topic=r.get("topic", ""),
                                title=r.get("content", "").split("\n", 1)[0][:80],
                                snippet=(r.get("content") or "")[:SUMMARY_TRUNCATE_LEN],
                                rank=-(float(r.get("similarity", 0.0))),
                                date=r.get("date", ""),
                                entry_id=entry_id,
                                conversation_id=None,
                            )
                        )
                except Exception:
                    logger.warning("Semantic search failed, using FTS only", exc_info=True)

        # Merge and deduplicate by source_key
        seen_keys: set[str] = set()
        merged: list[SearchResult] = []

        for result in fts_results:
            if result.source_key not in seen_keys:
                seen_keys.add(result.source_key)
                merged.append(result)

        for result in semantic_results:
            if result.source_key not in seen_keys:
                seen_keys.add(result.source_key)
                merged.append(result)

        merged.sort(key=lambda x: x.rank)
        merged = merged[:limit]

        return {
            "results": [result.model_dump(exclude={"rank", "source_key"}) for result in merged],
            "total": len(merged),
            "query": query,
        }
