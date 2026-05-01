"""MCP tool: journal_search (tsvector FTS + pgvector semantic)."""

import asyncio
import logging
from datetime import date as date_cls
from typing import Any

import asyncpg
from gubbi_common.db.user_scoped import MissingUserIdError, user_scoped_connection
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from journalctl.core.auth_context import current_user_id
from journalctl.core.cipher_guard import require_cipher
from journalctl.core.context import AppContext
from journalctl.core.scope import require_scope
from journalctl.core.validation import validate_date, validate_topic
from journalctl.models.search import SearchResult
from journalctl.storage.repositories import conversations as conv_repo
from journalctl.storage.repositories import entries as entry_repo
from journalctl.storage.repositories import search as search_repo
from journalctl.tools._response_size import _assert_response_ok, _report_oversized
from journalctl.tools.constants import (
    DEFAULT_SEARCH_LIMIT,
    MAX_QUERY_LEN,
    MAX_SEARCH_CONTENT_CHARS,
    MAX_SEARCH_RESULTS,
)
from journalctl.tools.errors import invalid_date, invalid_topic, validation_error

logger = logging.getLogger(__name__)


def _truncate_text(value: str) -> str:
    return value[:MAX_SEARCH_CONTENT_CHARS]


def _truncate_title_summary(title: str, summary: str) -> tuple[str, str]:
    budget = MAX_SEARCH_CONTENT_CHARS
    if len(title) + len(summary) <= budget:
        return title, summary
    if len(title) >= budget:
        return title[:budget], ""
    remaining = budget - len(title)
    return title, summary[:remaining]


def register(mcp: FastMCP, app_ctx: AppContext) -> None:
    """Register search tool on the MCP server."""

    @mcp.tool(
        title="Search Journal",
        annotations=ToolAnnotations(
            readOnlyHint=True,
        ),
    )
    @require_scope("journal:read")
    async def journal_search(
        query: str,
        topic_prefix: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = DEFAULT_SEARCH_LIMIT,
    ) -> dict[str, Any]:
        """Search your journal by keyword or meaning - the primary search tool.

        Handles both exact keyword lookups ("deployment Phase 7") and conceptual
        questions ("what car do I drive?", "what's my salary?"). Searches across
        all topics, entries, and conversations.

        Do NOT use for browsing a single topic - use journal_read_topic instead.
        Do NOT use for time-based browsing - use journal_timeline instead.

        Args:
            query: Search query - keywords, phrases, or natural language questions.
            topic_prefix: Filter to topics under this prefix (e.g. 'work').
                          If omitted, searches all topics.
                          Must be a valid topic path if provided.
            date_from: Filter entries on or after this date (YYYY-MM-DD).
            date_to: Filter entries on or before this date (YYYY-MM-DD).
            limit: Maximum results (default 10).

        Returns:
            List of matching results with full decrypted content, ordered by
            relevance (best first). Each result includes entry_id/conversation_id
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

        fts_results: list[SearchResult] = []
        semantic_results: list[SearchResult] = []

        try:
            query_embedding = await asyncio.to_thread(app_ctx.embedding_service.encode, query)
        except Exception:
            logger.warning("Query encoding failed, semantic search disabled", exc_info=True)
            query_embedding = None

        df = date_cls.fromisoformat(date_from) if date_from else None
        dt = date_cls.fromisoformat(date_to) if date_to else None
        user_id = current_user_id.get()
        if user_id is None:
            raise MissingUserIdError("no authenticated user -- check BearerAuthMiddleware wiring")
        cipher = require_cipher(app_ctx)

        async with user_scoped_connection(app_ctx.pool, user_id=user_id) as conn:
            fts_results = await search_repo.fts_search(
                conn, query, topic_prefix, date_from, date_to, limit
            )

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
                    semantic_results = [
                        SearchResult(
                            source_key=f"entry:{r.get('entry_id')}",
                            doc_type="entry",
                            topic=str(r.get("topic", "")),
                            rank=-float(r.get("similarity", 0.0)),
                            date=str(r.get("date", "")),
                            entry_id=r.get("entry_id"),
                            conversation_id=None,
                        )
                        for r in raw
                        if r.get("entry_id") is not None
                    ]
                except asyncpg.PostgresError:
                    logger.warning("Semantic search failed, using FTS only", exc_info=True)
                except Exception:
                    logger.exception("Semantic search failed unexpectedly")
                    raise

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

            # Batch-collect unique IDs for a single round-trip per entity type.
            entry_ids: set[int] = set()
            conv_ids: set[int] = set()
            for result in merged:
                if result.doc_type == "entry" and result.entry_id is not None:
                    entry_ids.add(result.entry_id)
                elif result.doc_type == "conversation" and result.conversation_id is not None:
                    conv_ids.add(result.conversation_id)

            # Batched fetches — one query per entity type instead of N.
            entry_id_list = list(entry_ids)
            conv_id_list = list(conv_ids)

            decrypted_entries: dict[int, tuple[str, str | None]] = {}
            try:
                decrypted_entries = await entry_repo.get_texts(conn, cipher, entry_id_list)
            except asyncpg.PostgresError:
                logger.exception(
                    "Entry batch query failed, skipping %d entries: %s",
                    len(entry_id_list),
                    repr(entry_id_list),
                )

            decrypted_convs: dict[int, tuple[str, str]] = {}
            try:
                decrypted_convs = await conv_repo.get_titles_summaries(conn, cipher, conv_id_list)
            except asyncpg.PostgresError:
                logger.exception(
                    "Conversation batch query failed, skipping %d conversations: %s",
                    len(conv_id_list),
                    repr(conv_id_list),
                )

            hydrated: list[SearchResult] = []
            for result in merged:
                if (
                    result.doc_type == "entry"
                    and result.entry_id is not None
                    and result.entry_id in decrypted_entries
                ):
                    content, _reasoning = decrypted_entries[result.entry_id]
                    hydrated.append(result.model_copy(update={"content": _truncate_text(content)}))
                elif (
                    result.doc_type == "conversation"
                    and result.conversation_id is not None
                    and result.conversation_id in decrypted_convs
                ):
                    title, summary = decrypted_convs[result.conversation_id]
                    truncated_title, truncated_summary = _truncate_title_summary(title, summary)
                    hydrated.append(
                        result.model_copy(
                            update={
                                "title": truncated_title,
                                "summary": truncated_summary,
                            }
                        )
                    )

        hydrated.sort(key=lambda x: x.rank)
        hydrated = hydrated[:limit]

        payload: list[dict[str, Any]] = []
        for result in hydrated:
            if result.doc_type == "entry":
                payload.append(
                    {
                        "doc_type": "entry",
                        "topic": result.topic,
                        "date": result.date,
                        "entry_id": result.entry_id,
                        "conversation_id": None,
                        "content": result.content or "",
                    }
                )
            elif result.doc_type == "conversation":
                payload.append(
                    {
                        "doc_type": "conversation",
                        "topic": result.topic,
                        "date": result.date,
                        "entry_id": None,
                        "conversation_id": result.conversation_id,
                        "title": result.title or "",
                        "summary": result.summary or "",
                    }
                )

        search_result = {
            "results": payload,
            "total": len(payload),
            "query": query,
        }
        err = _assert_response_ok(search_result, tool_name="journal_search")
        if err:
            await _report_oversized("journal_search", err)
            return err
        return search_result
