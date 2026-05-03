"""MCP tools: journal_save_conversation, journal_list_conversations,
journal_read_conversation."""

import asyncio
import logging
from typing import Any, NotRequired, TypedDict

from gubbi_common.db.user_scoped import MissingUserIdError, user_scoped_connection
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from journalctl.core.audit_decorator import ACTION_CONVERSATION_SAVED, audited
from journalctl.core.auth_context import current_user_id
from journalctl.core.cipher_guard import require_cipher
from journalctl.core.context import AppContext
from journalctl.core.scope import require_scope
from journalctl.core.validation import (
    local_today,
    reject_tool_call_syntax,
    sanitize_freetext,
    sanitize_label,
    validate_date,
    validate_title,
    validate_topic,
)
from journalctl.models.conversation import Message
from journalctl.storage.exceptions import ConversationNotFoundError, TopicNotFoundError
from journalctl.storage.repositories import conversations as conv_repo
from journalctl.storage.repositories import entries as entry_repo
from journalctl.storage.repositories.conversations import (
    read_conversation_by_id,
    read_conversation_by_id_paginated,
)
from journalctl.tools._response_size import _assert_response_ok, _report_oversized
from journalctl.tools.constants import (
    DEFAULT_CONVERSATION_MESSAGES_LIMIT,
    DEFAULT_CONVERSATIONS_LIMIT,
    KEEP_ROLES,
    LIST_SUMMARY_PREVIEW_CHARS,
    MAX_CONVERSATION_MESSAGES,
    MAX_CONVERSATIONS_RESULTS,
    MAX_MESSAGES_PER_CONVERSATION,
    MAX_MSG_CHARS,
)
from journalctl.tools.errors import invalid_date, invalid_topic, not_found, validation_error

logger = logging.getLogger(__name__)


class MessageInput(TypedDict):
    """Schema for a single conversation message."""

    role: str
    content: str
    timestamp: NotRequired[str]


def _format_messages_as_markdown(title: str, messages: list[Message]) -> str:
    """Render a list of messages as a readable markdown string."""
    parts = [f"# {title}\n"]
    for msg in messages:
        role_label = "User" if msg.role == "user" else "Assistant"
        ts = f" ({msg.timestamp})" if msg.timestamp else ""
        parts.append(f"---\n\n### {role_label}{ts}\n\n{msg.content}")
    return "\n\n".join(parts)


def register(mcp: FastMCP, app_ctx: AppContext) -> None:
    """Register conversation tools on the MCP server."""

    @mcp.tool(
        title="Save Conversation",
        annotations=ToolAnnotations(
            readOnlyHint=False,
            destructiveHint=False,
            openWorldHint=False,
            idempotentHint=True,
        ),
    )
    @require_scope("journal:write")
    @audited(
        ACTION_CONVERSATION_SAVED,
        target_type="conversation",
        target_kind="conversation",
        app_ctx=app_ctx,
    )
    async def journal_save_conversation(
        topic: str,
        title: str,
        messages: list[MessageInput],
        summary: str,
        source: str = "claude",
        tags: list[str] | None = None,
        date: str | None = None,
    ) -> dict[str, Any]:
        """Save a conversation transcript to the journal — "save this chat" or
        "remember what we discussed."

        Call when the user asks to save, or offer during meaningful moments:
        decisions, plans, breakthroughs, or reflections. The topic must already
        exist — check the briefing, or create one with journal_create_topic.

        Re-saving the same topic + title updates the previous version.

        Quality guidelines:
        - title: Specific and scannable. "MCP caching strategy discussion", not "Chat about work."
        - summary: 1-3 sentences capturing the essence of the entire conversation.
        - messages: Include the entire conversation, not just few messages.

        Example: journal_save_conversation(topic='project/mcp',
            title='MCP caching strategy',
            summary='Decided to cache agent state in Redis with 50ms latency budget.',
            messages=[{"role": "user", "content": "Should we cache agent state?"},
                      {"role": "assistant", "content": "Yes, here is why..."}])

        Args:
            topic: Topic this conversation relates to (e.g. 'work/acme').
            title: Descriptive title for the conversation.
            messages: List of messages, each with role ('user' or 'assistant'),
                      content (str), and optional timestamp (str).
            summary: A concise summary of the conversation (1-3 sentences).
            source: Name of the app or LLM (e.g. 'claude', 'chatgpt'). Defaults to 'claude'.
            tags: Use any relevant tags for filtering and categorization
            (e.g. ['finance', 'car', 'decision']).
            date: When the conversation happened (YYYY-MM-DD). Defaults to today.

        Returns:
            Conversation ID, summary, and whether it was a new save or update.
        """
        try:
            topic = validate_topic(topic)
        except ValueError as e:
            return invalid_topic(topic, str(e))
        try:
            title = validate_title(title)
        except ValueError as e:
            return validation_error(str(e))
        source = sanitize_label(source)
        summary = sanitize_freetext(summary)
        try:
            reject_tool_call_syntax(summary)
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

        if len(messages) > MAX_MESSAGES_PER_CONVERSATION:
            return validation_error(
                f"Too many messages: max {MAX_MESSAGES_PER_CONVERSATION}, got {len(messages)}"
            )

        # Only keep human-readable turns. Tool calls, tool results, and system
        # messages are infrastructure noise — not part of the conversation record.
        keepable = [m for m in messages if m.get("role") in KEEP_ROLES]
        try:
            parsed_messages = [
                Message(
                    role=m.get("role", "user"),
                    content=sanitize_freetext(m.get("content", ""))[:MAX_MSG_CHARS],
                    timestamp=m.get("timestamp"),
                )
                for m in keepable
                if m.get("content", "").strip()
            ]
        except (TypeError, AttributeError) as e:
            return validation_error(
                "Invalid message format — each message"
                f" must be a dict with 'role' and 'content': {e}"
            )

        for msg in parsed_messages:
            try:
                reject_tool_call_syntax(msg.content)
            except ValueError as e:
                return validation_error(f"Message content: {e}")

        if not parsed_messages:
            return validation_error("No user/assistant messages found after filtering.")

        empty_dropped = len(keepable) - len(parsed_messages)

        user_id = current_user_id.get()
        if user_id is None:
            raise MissingUserIdError("no authenticated user -- check BearerAuthMiddleware wiring")

        cipher = require_cipher(app_ctx)

        try:
            async with user_scoped_connection(app_ctx.pool, user_id=user_id) as conn:
                save_result = await conv_repo.save_conversation(
                    conn,
                    cipher,
                    conversations_json_dir=app_ctx.settings.conversations_json_dir,
                    topic=topic,
                    title=title,
                    messages=parsed_messages,
                    summary=summary,
                    source=source,
                    tags=tags,
                    date=resolved_date,
                )
        except TopicNotFoundError:
            return not_found("Topic", topic)

        if save_result.superseded_json_path is not None:
            conv_repo.delete_superseded_json_archive(
                app_ctx.settings.conversations_json_dir, save_result.superseded_json_path
            )

        conv_id = save_result.conversation_id
        saved_summary = save_result.summary
        is_update = save_result.is_update
        linked_entry_id = save_result.linked_entry_id

        # Embed linked entry after transaction commits (best-effort)
        linked_content = f"Conversation saved: {title}\n\n{summary}"
        try:
            embedding = await asyncio.to_thread(app_ctx.embedding_service.encode, linked_content)
            async with user_scoped_connection(app_ctx.pool, user_id=user_id) as conn:
                await app_ctx.embedding_service.store_by_vector(conn, linked_entry_id, embedding)
                await entry_repo.mark_indexed(conn, linked_entry_id)
        except Exception as e:
            logger.warning("Failed to embed linked entry %s: %s", linked_entry_id, e, exc_info=True)

        result: dict[str, Any] = {
            "status": "updated" if is_update else "saved",
            "conversation_id": conv_id,
            "summary": saved_summary,
            "topic": topic,
            "title": title,
        }
        notes = []
        if tags_dropped:
            notes.append(f"{tags_dropped} tag(s) dropped (contained only unsupported characters)")
        if empty_dropped:
            notes.append(f"{empty_dropped} message(s) dropped (empty content)")
        if notes:
            result["note"] = "; ".join(notes)
        return result

    @mcp.tool(
        title="List Conversations",
        annotations=ToolAnnotations(
            readOnlyHint=True,
        ),
    )
    @require_scope("journal:read")
    async def journal_list_conversations(
        topic_prefix: str | None = None,
        limit: int = DEFAULT_CONVERSATIONS_LIMIT,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Browse saved conversations by topic — 'what conversations have we had about X?'

        Use this tool when the user wants to browse a list of chats, not find
        specific content within them. For keyword search across both entries AND
        conversations, use journal_search instead.

        Returns titles, dates, and summaries.  Summaries in list view are
        truncated to LIST_SUMMARY_PREVIEW_CHARS characters; call
        journal_read_conversation for the full text.

        Args:
            topic_prefix: Filter by topic prefix (e.g. 'work').
                          If omitted, lists all conversations.
            limit: Maximum conversations to return (default 50, max 200).
            offset: Number of conversations to skip for pagination (default 0).

        Returns:
            List of conversations with id, title, date, summary,
            and message count.
        """
        limit = max(1, min(limit, MAX_CONVERSATIONS_RESULTS))
        offset = max(0, offset)
        if topic_prefix:
            topic_prefix = topic_prefix.rstrip("/") or None
        if topic_prefix:
            try:
                topic_prefix = validate_topic(topic_prefix)
            except ValueError as e:
                return invalid_topic(topic_prefix, str(e))
        user_id = current_user_id.get()
        if user_id is None:
            raise MissingUserIdError("no authenticated user -- check BearerAuthMiddleware wiring")
        cipher = require_cipher(app_ctx)
        async with user_scoped_connection(app_ctx.pool, user_id=user_id) as conn:
            convs, total = await conv_repo.list_conversations(
                conn,
                cipher,
                topic_prefix=topic_prefix,
                limit=limit,
                offset=offset,
            )
        conversations_list = []
        for c in convs:
            rec = c.model_dump()
            summary = rec.get("summary", "") or ""
            if len(summary) > LIST_SUMMARY_PREVIEW_CHARS:
                rec["summary"] = summary[:LIST_SUMMARY_PREVIEW_CHARS]
                rec["summary_truncated"] = True
            else:
                rec["summary_truncated"] = False
            conversations_list.append(rec)

        result = {
            "conversations": conversations_list,
            "total": total,
            "limit": limit,
            "offset": offset,
        }
        err = _assert_response_ok(result, tool_name="journal_list_conversations")
        if err:
            await _report_oversized("journal_list_conversations", err)
            return err
        return result

    @mcp.tool(
        title="Read Conversation",
        annotations=ToolAnnotations(
            readOnlyHint=True,
        ),
    )
    @require_scope("journal:read")
    async def journal_read_conversation(
        conversation_id: int,
        preview: bool = False,
        messages_limit: int = DEFAULT_CONVERSATION_MESSAGES_LIMIT,
        messages_offset: int = 0,
    ) -> dict[str, Any]:
        """Read the full transcript of a saved conversation.

        Use the 'id' from journal_list_conversations or journal_search results.

        Args:
            conversation_id: The conversation's ID (from list or search results).
            preview: If True, return only first and last 3 messages instead of
                     the full transcript. Use for long conversations to avoid
                     consuming the entire context window. When True,
                     messages_limit/messages_offset are ignored.
            messages_limit: When preview=False, max messages to return
                            (default 20, max 100). Used as SQL LIMIT.
            messages_offset: When preview=False, messages to skip for
                             pagination (default 0). Used as SQL OFFSET.

        Returns:
            metadata (title, topic, summary, dates, participants),
            content (full transcript or subset as markdown),
            messages_shown, messages_total (total in conversation).
        """
        user_id = current_user_id.get()
        if user_id is None:
            raise MissingUserIdError("no authenticated user -- check BearerAuthMiddleware wiring")
        cipher = require_cipher(app_ctx)
        try:
            async with user_scoped_connection(app_ctx.pool, user_id=user_id) as conn:
                if preview:
                    meta, messages, total_messages = await read_conversation_by_id(
                        conn, cipher, conversation_id, preview=True
                    )
                else:
                    messages_limit = max(1, min(messages_limit, MAX_CONVERSATION_MESSAGES))
                    messages_offset = max(0, messages_offset)
                    meta, messages, total_messages = await read_conversation_by_id_paginated(
                        conn,
                        cipher,
                        conversation_id,
                        messages_limit=messages_limit,
                        messages_offset=messages_offset,
                    )
        except ConversationNotFoundError:
            return not_found("Conversation", conversation_id)
        content = _format_messages_as_markdown(meta.title, messages)
        result = {
            "metadata": meta.model_dump(),
            "content": content,
            "preview": preview,
            "messages_shown": len(messages),
            "messages_total": total_messages,
        }
        err = _assert_response_ok(result, tool_name="journal_read_conversation")
        if err:
            await _report_oversized("journal_read_conversation", err)
            return err
        return result
