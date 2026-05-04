"""MCP tools: journal_append_entry, journal_read_topic, journal_update_entry,
journal_delete_entry."""

import asyncio
import logging
from typing import Any, Literal
from uuid import UUID

from gubbi_common.db.user_scoped import MissingUserIdError, user_scoped_connection
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from gubbi.core.audit_decorator import (
    ACTION_ENTRY_CREATED,
    ACTION_ENTRY_DELETED,
    ACTION_ENTRY_UPDATED,
    audited,
)
from gubbi.core.auth_context import current_user_id
from gubbi.core.cipher_guard import require_cipher
from gubbi.core.context import AppContext
from gubbi.core.scope import require_scope
from gubbi.core.validation import (
    is_future_date,
    local_today,
    reject_tool_call_syntax,
    sanitize_freetext,
    sanitize_label,
    validate_date,
    validate_topic,
)
from gubbi.storage.exceptions import EntryNotFoundError, TopicNotFoundError
from gubbi.storage.repositories import entries as entry_repo
from gubbi.tools._response_size import _assert_response_ok, _report_oversized
from gubbi.tools.constants import DEFAULT_ENTRIES_LIMIT, MAX_READ_ENTRIES
from gubbi.tools.errors import invalid_date, invalid_topic, not_found, validation_error

logger = logging.getLogger(__name__)


def register(mcp: FastMCP, app_ctx: AppContext) -> None:
    """Register entry tools on the MCP server."""

    async def _embed_entry(
        user_id: UUID,
        entry_id: int,
        content: str,
    ) -> list[float] | None:
        """Encode text and store embedding. Returns the embedding on success, None on failure.

        Encodes outside a DB connection so the pool is free during ONNX inference.
        """
        try:
            embedding = await asyncio.to_thread(app_ctx.embedding_service.encode, content)
            async with user_scoped_connection(app_ctx.pool, user_id=user_id) as conn:
                await app_ctx.embedding_service.store_by_vector(conn, entry_id, embedding)
            return embedding
        except Exception as e:
            logger.warning("Failed to embed entry %s: %s", entry_id, e, exc_info=True)
            return None

    @mcp.tool(
        title="Append Entry",
        annotations=ToolAnnotations(
            readOnlyHint=False,
            destructiveHint=False,
            openWorldHint=False,
            idempotentHint=False,
        ),
    )
    @require_scope("journal:write")
    @audited(ACTION_ENTRY_CREATED, target_type="entry", target_kind="entry", app_ctx=app_ctx)
    async def journal_append_entry(
        topic: str,
        content: str,
        reasoning: str | None = None,
        tags: list[str] | None = None,
        date: str | None = None,
    ) -> dict[str, Any]:
        """Record a life event, decision, or update — "remember this",
        "note that we decided X", or "I just did Y.". Call proactively when the
        user shares significant news, decisions, progress, or milestones.

        The topic must already exist — check the briefing for recently used topics,
        journal_list_topics to see all available topics, or create one with journal_create_topic.

        Example: User says "We decided to use PostgreSQL instead of MongoDB."
        → journal_append_entry(topic="projects/alpha", content="Chose PostgreSQL for the database",
            reasoning="Mongo had no ACID transactions, team already knows SQL")

        Do NOT use for searching or reading — use journal_search or journal_read_topic.

        Quality guidelines:
        - content: Write a clear, scannable headline — this appears in briefings and timelines.
          Good: "Chose PostgreSQL over MongoDB for Project Alpha"
          Bad:  "Database decision" or the user's full paragraph pasted verbatim.
        - reasoning: Capture the WHY — tradeoffs, context, constraints. Omit for routine events.
          This field is only loaded on full read, so it's the place for detail.
        - tags: Use any relevant tags for filtering and categorization
          (e.g. 'finance', 'idea', 'important').
        - date: Only override if the user is recording or planning something for a different day.

        Args:
            topic: Topic path (e.g. 'work/acme', 'health', 'hobbies/woodworking').
            content: What happened — the headline. Shown in briefing and timeline.
            reasoning: Why it happened — reasoning or tradeoffs. Only loaded when
                        the entry is read in full; leave empty for routine events.
            tags: Tags relevant to the entry (e.g. ['finance', 'idea', 'important']).
            date: Date of the entry as YYYY-MM-DD. Defaults to today.

        Returns:
            Confirmation with entry_id, topic, and date.
        """
        try:
            topic = validate_topic(topic)
        except ValueError as e:
            return invalid_topic(topic, str(e))
        content = sanitize_freetext(content)
        if not content.strip():
            return validation_error("Content cannot be empty")
        try:
            reject_tool_call_syntax(content)
        except ValueError as e:
            return validation_error(str(e))
        if reasoning:
            reasoning = sanitize_freetext(reasoning)
            try:
                reject_tool_call_syntax(reasoning)
            except ValueError as e:
                return validation_error(str(e))
        tags_dropped = 0
        if tags:
            original_tag_count = len(tags)
            tags = [s for t in tags if (s := sanitize_label(t))]
            tags_dropped = original_tag_count - len(tags)
        if date:
            try:
                validate_date(date)
            except ValueError:
                return invalid_date(date)

        resolved_date = date or local_today(app_ctx.settings.timezone)
        user_id = current_user_id.get()
        if user_id is None:
            raise MissingUserIdError("no authenticated user -- check BearerAuthMiddleware wiring")
        cipher = require_cipher(app_ctx)

        try:
            async with user_scoped_connection(app_ctx.pool, user_id=user_id) as conn:
                entry_id = await entry_repo.append(
                    conn,
                    cipher,
                    topic=topic,
                    content=content,
                    reasoning=reasoning,
                    tags=tags,
                    date=resolved_date,
                )
        except TopicNotFoundError:
            return not_found("Topic", topic)

        # Embed after the transaction commits (embedding is best-effort)
        if await _embed_entry(user_id, entry_id, content) is not None:
            async with user_scoped_connection(app_ctx.pool, user_id=user_id) as conn:
                await entry_repo.mark_indexed(conn, entry_id)

        result: dict[str, Any] = {
            "status": "appended",
            "topic": topic,
            "date": resolved_date,
            "entry_id": entry_id,
        }
        notes = []
        if date and is_future_date(date, app_ctx.settings.timezone):
            notes.append("Date is in the future")
        if tags_dropped:
            notes.append(f"{tags_dropped} tag(s) dropped (contained only unsupported characters)")
        if notes:
            result["note"] = "; ".join(notes)
        return result

    @mcp.tool(
        title="Read Topic",
        annotations=ToolAnnotations(
            readOnlyHint=True,
        ),
    )
    @require_scope("journal:read")
    async def journal_read_topic(
        topic: str,
        limit: int = DEFAULT_ENTRIES_LIMIT,
        date_from: str | None = None,
        date_to: str | None = None,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Read entries from a topic — "show me my notes on health" or
        "what did I write about work?"

        Use when the user wants to review a specific topic's entries.
        Returns entries in chronological order with content and reasoning.

        Do NOT use for keyword search across topics — use journal_search instead.
        Do NOT use for time-based browsing — use journal_timeline instead.

        Args:
            topic: Topic path — lowercase alphanumeric with hyphens, max 2 levels
                   (e.g. 'work/acme').
            limit: Max entries to return (default 10). Use a large number for more history.
            date_from: Only entries on or after this date (YYYY-MM-DD).
            date_to: Only entries on or before this date (YYYY-MM-DD).
            offset: Skip first N entries for pagination (default 0).

        Returns:
            metadata (topic info), entries (list with content and reasoning),
            total (total matching entries), limit, offset.
        """
        try:
            topic = validate_topic(topic)
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
        limit = max(1, min(limit, MAX_READ_ENTRIES))
        offset = max(0, offset)
        user_id = current_user_id.get()
        if user_id is None:
            raise MissingUserIdError("no authenticated user -- check BearerAuthMiddleware wiring")
        cipher = require_cipher(app_ctx)
        try:
            async with user_scoped_connection(app_ctx.pool, user_id=user_id) as conn:
                meta, entries, total = await entry_repo.read(
                    conn,
                    cipher,
                    topic,
                    limit=limit,
                    date_from=date_from,
                    date_to=date_to,
                    offset=offset,
                )
        except TopicNotFoundError:
            return not_found("Topic", topic)

        result = {
            "metadata": meta.model_dump(exclude={"id", "created", "updated"}),
            "entries": [e.model_dump() for e in entries],
            "total": total,
            "limit": limit,
            "offset": offset,
        }
        err = _assert_response_ok(result, tool_name="journal_read_topic")
        if err:
            await _report_oversized("journal_read_topic", err)
            return err
        return result

    @mcp.tool(
        title="Update Entry",
        annotations=ToolAnnotations(
            readOnlyHint=False,
            destructiveHint=False,
            openWorldHint=False,
            idempotentHint=True,
        ),
    )
    @require_scope("journal:write")
    @audited(ACTION_ENTRY_UPDATED, target_type="entry", target_kind="entry", app_ctx=app_ctx)
    async def journal_update_entry(
        entry_id: int,
        content: str | None = None,
        reasoning: str | None = None,
        mode: Literal["replace", "append"] = "replace",
        date: str | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Correct or expand a journal entry — "fix that entry" or "add more detail."

        Use the entry's 'id' from journal_read_topic, journal_search, or journal_timeline results.

        Do NOT use to remove an entry — use journal_delete_entry instead.

        Args:
            entry_id: The entry's 'id' (from read, search, or timeline results).
            content: New content for the entry (optional — omit to only change date/tags).
            reasoning: Updated reasoning (optional). Omit to keep current reasoning.
            mode: 'replace' overwrites the entire entry content (use for corrections or rewrites).
                  'append' adds new text to the end (use for follow-up notes or addenda).
                  Default: 'replace'.
            date: Correct the entry's date (YYYY-MM-DD). Omit to keep current date.
            tags: Replace the entry's tags. Omit to keep current tags.

        Returns:
            Confirmation with updated entry_id.
        """
        if content is not None:
            content = sanitize_freetext(content)
            if not content.strip():
                return validation_error("content cannot be empty")
            try:
                reject_tool_call_syntax(content)
            except ValueError as e:
                return validation_error(str(e))
        if reasoning is not None:
            reasoning = sanitize_freetext(reasoning)
            try:
                reject_tool_call_syntax(reasoning)
            except ValueError as e:
                return validation_error(str(e))
        if date:
            try:
                validate_date(date)
            except ValueError:
                return invalid_date(date)
        tags_dropped = 0
        if tags:
            original_tag_count = len(tags)
            tags = [s for t in tags if (s := sanitize_label(t))]
            tags_dropped = original_tag_count - len(tags)

        user_id = current_user_id.get()
        if user_id is None:
            raise MissingUserIdError("no authenticated user -- check BearerAuthMiddleware wiring")
        cipher = require_cipher(app_ctx)

        # Read committed text inside the same transaction — avoids a second round-trip.
        # Within a transaction, reads see writes from the same transaction (savepoint).
        row_data: tuple[str, str | None] | None = None
        try:
            async with user_scoped_connection(app_ctx.pool, user_id=user_id) as conn:
                await entry_repo.update(
                    conn,
                    cipher,
                    entry_id=entry_id,
                    content=content,
                    reasoning=reasoning,
                    mode=mode,
                    date=date,
                    tags=tags,
                )
                if content is not None or reasoning is not None:
                    row_data = await entry_repo.get_text(conn, cipher, entry_id)
        except EntryNotFoundError:
            return not_found("Entry", entry_id)

        # Re-embed if text changed: encode outside any connection, then store+mark in one.
        if row_data:
            embed_text = (row_data[0] or "") + " " + (row_data[1] or "")
            try:
                embedding = await asyncio.to_thread(
                    app_ctx.embedding_service.encode, embed_text.strip()
                )
                async with user_scoped_connection(app_ctx.pool, user_id=user_id) as conn:
                    await app_ctx.embedding_service.store_by_vector(conn, entry_id, embedding)
                    await entry_repo.mark_indexed(conn, entry_id)
            except Exception as e:
                logger.warning("Failed to embed updated entry %s: %s", entry_id, e, exc_info=True)

        result: dict[str, Any] = {
            "status": "updated",
            "entry_id": entry_id,
            "mode": mode,
        }
        notes = []
        if date and is_future_date(date, app_ctx.settings.timezone):
            notes.append("Date is in the future")
        if tags_dropped:
            notes.append(f"{tags_dropped} tag(s) dropped (contained only unsupported characters)")
        if notes:
            result["note"] = "; ".join(notes)
        return result

    @mcp.tool(
        title="Delete Entry",
        annotations=ToolAnnotations(
            readOnlyHint=False,
            destructiveHint=True,
            openWorldHint=False,
            idempotentHint=True,
        ),
    )
    @require_scope("journal:write")
    @audited(ACTION_ENTRY_DELETED, target_type="entry", target_kind="entry", app_ctx=app_ctx)
    async def journal_delete_entry(
        entry_id: int,
    ) -> dict[str, Any]:
        """Remove a journal entry permanently — wrong data, duplicate, or mistake.
        Trigger: 'delete that', 'forget that', 'undo that', 'scratch that', 'that was wrong.'

        Use the entry's 'id' from journal_read_topic, journal_search, or journal_timeline results.

        Do NOT use to correct an entry — use journal_update_entry instead.

        Args:
            entry_id: The entry's 'id' (from read, search, or timeline results).

        Returns:
            Confirmation with deleted entry_id.
        """
        user_id = current_user_id.get()
        if user_id is None:
            raise MissingUserIdError("no authenticated user -- check BearerAuthMiddleware wiring")
        try:
            async with user_scoped_connection(app_ctx.pool, user_id=user_id) as conn:
                # delete_entry soft-deletes the entry and removes its embedding
                await entry_repo.delete(conn, entry_id)
        except EntryNotFoundError:
            return not_found("Entry", entry_id)

        return {
            "status": "deleted",
            "entry_id": entry_id,
        }
