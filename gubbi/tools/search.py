"""MCP tool: journal_search (tsvector FTS + pgvector semantic).

The hybrid search pipeline (dual search -> hydrate -> sort/slice/shape) lives in
``gubbi.services.search`` so the web REST surface can reuse it unchanged. This
module owns only the tool-layer contract: argument validation, the 1..20 limit
clamp, user/cipher resolution, the user-scoped connection, and the
response-size guard.
"""

from typing import Any

import structlog
from gubbi_common.db.user_scoped import MissingUserIdError, user_scoped_connection
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from gubbi.app_context import AppContext
from gubbi.auth.scope import require_scope
from gubbi.auth_context import current_user_id
from gubbi.crypto.guard import require_cipher
from gubbi.services.search import encode_query, run_journal_search
from gubbi.tools.constants import (
    DEFAULT_SEARCH_LIMIT,
    MAX_QUERY_LEN,
    MAX_SEARCH_RESULTS,
)
from gubbi.tools.errors import invalid_date, invalid_topic, validation_error
from gubbi.tools.response_size import _report_oversized, check_response_size
from gubbi.validation import validate_date, validate_topic

__all__: list[str] = ["register"]

logger = structlog.get_logger(__name__)


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

        user_id = current_user_id.get()
        if user_id is None:
            raise MissingUserIdError("no authenticated user -- check BearerAuthMiddleware wiring")
        cipher = require_cipher(app_ctx)

        # Encode the query before acquiring the connection so the CPU-bound
        # encode does not run while holding a user-scoped transaction.
        query_embedding = await encode_query(app_ctx, query)

        async with user_scoped_connection(app_ctx.pool, user_id=user_id) as conn:
            search_result = await run_journal_search(
                conn,
                cipher,
                app_ctx,
                query=query,
                topic_prefix=topic_prefix,
                date_from=date_from,
                date_to=date_to,
                limit=limit,
                query_embedding=query_embedding,
            )

        err = check_response_size(search_result, tool_name="journal_search")
        if err:
            await _report_oversized("journal_search", err)
            return err
        return search_result
