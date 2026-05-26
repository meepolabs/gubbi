"""MCP tools: journal_append_entry / read_topic / update_entry / delete_entry."""

import asyncio
from typing import Any, Literal
from uuid import UUID

import structlog
from gubbi_common.audit.targets import TargetKind
from gubbi_common.db.user_scoped import MissingUserIdError, user_scoped_connection
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from gubbi.app_context import AppContext
from gubbi.audit import (
    ACTION_ENTRY_CREATED,
    ACTION_ENTRY_DELETED,
    ACTION_ENTRY_UPDATED,
    audited,
)
from gubbi.auth.scope import require_scope
from gubbi.auth_context import current_user_id
from gubbi.crypto.guard import require_cipher
from gubbi.storage.exceptions import EntryNotFoundError, TopicNotFoundError
from gubbi.storage.repositories import entries as entry_repo
from gubbi.tools.constants import (
    DEFAULT_ENTRIES_LIMIT,
    MAX_ENTRY_CONTENT_CHARS,
    MAX_ENTRY_REASONING_CHARS,
    MAX_READ_ENTRIES,
)
from gubbi.tools.errors import invalid_date, invalid_topic, not_found, validation_error
from gubbi.tools.response_size import _report_oversized, check_response_size
from gubbi.validation import (
    is_future_date,
    local_today,
    reject_tool_call_syntax,
    sanitize_freetext,
    sanitize_label,
    validate_date,
    validate_topic,
)

__all__: list[str] = ["register"]

logger = structlog.get_logger(__name__)


async def _embed_entry(
    app_ctx: AppContext,
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
            await app_ctx.embedding_service.save_by_vector(conn, entry_id, embedding)
        return embedding
    except Exception as exc:
        await logger.warning(
            "Failed to embed entry",
            entry_id=entry_id,
            error=str(exc),
            exc_info=True,
        )
        return None


async def _journal_append_entry(
    app_ctx: AppContext,
    topic: str,
    content: str,
    reasoning: str | None = None,
    tags: list[str] | None = None,
    date: str | None = None,
) -> dict[str, Any]:
    try:
        topic = validate_topic(topic)
    except ValueError as e:
        return invalid_topic(topic, str(e))
    # Caps applied pre-sanitization so the error reports the real input size,
    # not the post-strip size.  An oversized blob must be rejected at the tool
    # boundary before encryption + insert -- check_response_size only fires on
    # output, by which point the data is already encrypted and stored.
    if len(content) > MAX_ENTRY_CONTENT_CHARS:
        return validation_error(f"content exceeds {MAX_ENTRY_CONTENT_CHARS} characters")
    if reasoning is not None and len(reasoning) > MAX_ENTRY_REASONING_CHARS:
        return validation_error(f"reasoning exceeds {MAX_ENTRY_REASONING_CHARS} characters")
    content = sanitize_freetext(content)
    if not content.strip():
        return validation_error("content cannot be empty")
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
    if await _embed_entry(app_ctx, user_id, entry_id, content) is not None:
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


async def _journal_read_topic(
    app_ctx: AppContext,
    topic: str,
    limit: int = DEFAULT_ENTRIES_LIMIT,
    date_from: str | None = None,
    date_to: str | None = None,
    offset: int = 0,
) -> dict[str, Any]:
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
    err = check_response_size(result, tool_name="journal_read_topic")
    if err:
        await _report_oversized("journal_read_topic", err)
        return err
    return result


async def _journal_update_entry(
    app_ctx: AppContext,
    entry_id: int,
    content: str | None = None,
    reasoning: str | None = None,
    mode: Literal["replace", "append"] = "replace",
    date: str | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    # Two-stage validation:
    #   (1) sanitize + normalize "no real intent" cases to None
    #   (2) re-check the no-op condition
    #
    # Stage (1) catches inputs that LOOK like a partial update but, after
    # sanitization, carry no actual change.  Without this, ``reasoning=""``
    # or ``reasoning="\x00"`` would pass the all-None guard (it's not None),
    # sanitize to an empty string, and flow into ``entry_repo.update`` which
    # treats ``reasoning is not None`` as "reasoning changed" -- re-encrypting
    # the empty value, nulling ``indexed_at`` (triggering reindex), and
    # writing an audit row.  Normalize empty-after-sanitize back to None so
    # stage (2) can short-circuit.
    if content is not None:
        # Cap pre-sanitization so the error reports real input size.
        if len(content) > MAX_ENTRY_CONTENT_CHARS:
            return validation_error(f"content exceeds {MAX_ENTRY_CONTENT_CHARS} characters")
        content = sanitize_freetext(content)
        if not content.strip():
            # NOTE: append-mode empty-content is locked as a distinct
            # user-facing error -- "content cannot be empty" rather
            # than the no-op message -- so we MUST keep the explicit
            # validation_error here for content, even though reasoning takes
            # the silent normalize path.  The asymmetry is deliberate:
            # content is required, reasoning is optional.
            return validation_error("content cannot be empty")
        try:
            reject_tool_call_syntax(content)
        except ValueError as e:
            return validation_error(str(e))
    if reasoning is not None:
        if len(reasoning) > MAX_ENTRY_REASONING_CHARS:
            return validation_error(f"reasoning exceeds {MAX_ENTRY_REASONING_CHARS} characters")
        reasoning = sanitize_freetext(reasoning)
        try:
            reject_tool_call_syntax(reasoning)
        except ValueError as e:
            return validation_error(str(e))
        # Stage (1) normalization: empty/whitespace-only reasoning is not a
        # "clear reasoning" intent -- treat it as "didn't really want to
        # update reasoning" and drop it so the no-op guard below can fire.
        # ``\x00``-only inputs sanitize to "" here; whitespace-only stays
        # whitespace (sanitize_freetext preserves whitespace) so we use
        # .strip() for the check.
        if not reasoning.strip():
            reasoning = None
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
    # tags=[] as an input remains intentional ("clear all tags") and is NOT
    # normalized to None.  The all-dropped case (every sanitize_label
    # returned empty) collapses to tags=[] here too; that ambiguity is
    # preserved deliberately -- callers that need stricter semantics should
    # validate tags upstream.

    # Stage (2) no-op guard: if no field has a real value after normalization,
    # reject at the tool boundary so @audited does NOT fire and no ghost
    # audit row is written.  Runs AFTER stage (1) so empty-reasoning paths
    # that normalize to None are caught here rather than slipping through
    # to the repo.
    if content is None and reasoning is None and date is None and tags is None:
        return validation_error("No fields to update")

    user_id = current_user_id.get()
    if user_id is None:
        raise MissingUserIdError("no authenticated user -- check BearerAuthMiddleware wiring")
    cipher = require_cipher(app_ctx)

    # Read committed text inside the same transaction -- avoids a second round-trip.
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
                current_row_data = await entry_repo.get_text(conn, cipher, entry_id)
                if current_row_data != row_data:
                    current_state = "deleted" if current_row_data is None else "updated_again"
                    await logger.warning(
                        "entry.embedding_update_skipped_stale",
                        entry_id=entry_id,
                        user_id=str(user_id),
                        current_state=current_state,
                    )
                else:
                    await app_ctx.embedding_service.save_by_vector(conn, entry_id, embedding)
                    await entry_repo.mark_indexed(conn, entry_id)
        except Exception as exc:
            # Best-effort: the atomic DELETE inside entry_repo.update has
            # already removed the stale entry_embeddings row, so failure
            # here leaves the entry semantic-blind (FTS-only) -- never
            # findable by old-content keywords. Log at error level so a
            # sustained rate is visible to the operator; a real reindex
            # path is the durable fix when one is wired.
            await logger.error(
                "entry.embedding_update_failed",
                entry_id=entry_id,
                user_id=str(user_id),
                error=str(exc),
                error_type=type(exc).__name__,
                exc_info=True,
            )

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


async def _journal_delete_entry(
    app_ctx: AppContext,
    entry_id: int,
) -> dict[str, Any]:
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


def register(mcp: FastMCP, app_ctx: AppContext) -> None:
    """Register entry tools on the MCP server."""

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
    @audited(
        ACTION_ENTRY_CREATED, target_type="entry", target_kind=TargetKind.ENTRY, app_ctx=app_ctx
    )
    async def journal_append_entry(
        topic: str,
        content: str,
        reasoning: str | None = None,
        tags: list[str] | None = None,
        date: str | None = None,
    ) -> dict[str, Any]:
        """Record a life event, decision, or update -- "remember this" / "I just did Y".

        Call proactively when the user shares significant news, decisions,
        progress, or milestones.

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
        return await _journal_append_entry(app_ctx, topic, content, reasoning, tags, date)

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
        """Read entries from a topic -- "show me my notes" / "what did I write".

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
        return await _journal_read_topic(app_ctx, topic, limit, date_from, date_to, offset)

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
    @audited(
        ACTION_ENTRY_UPDATED, target_type="entry", target_kind=TargetKind.ENTRY, app_ctx=app_ctx
    )
    async def journal_update_entry(
        entry_id: int,
        content: str | None = None,
        reasoning: str | None = None,
        mode: Literal["replace", "append"] = "replace",
        date: str | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Correct or expand a journal entry — "fix that entry" or "add more detail.".

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
        return await _journal_update_entry(app_ctx, entry_id, content, reasoning, mode, date, tags)

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
    @audited(
        ACTION_ENTRY_DELETED, target_type="entry", target_kind=TargetKind.ENTRY, app_ctx=app_ctx
    )
    async def journal_delete_entry(
        entry_id: int,
    ) -> dict[str, Any]:
        """Remove a journal entry permanently — wrong data, duplicate, or mistake.

        Trigger: 'delete that', 'forget that', 'undo that', 'scratch that', 'that was wrong.'.

        Use the entry's 'id' from journal_read_topic, journal_search, or journal_timeline results.

        Do NOT use to correct an entry — use journal_update_entry instead.

        Args:
            entry_id: The entry's 'id' (from read, search, or timeline results).

        Returns:
            Confirmation with deleted entry_id.
        """
        return await _journal_delete_entry(app_ctx, entry_id)
