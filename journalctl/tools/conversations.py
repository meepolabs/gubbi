"""MCP tools: journal_save_conversation, journal_list_conversations,
journal_read_conversation."""

from typing import Any, NotRequired, TypedDict

from mcp.server.fastmcp import FastMCP

from journalctl.core.validation import (
    sanitize_freetext,
    sanitize_label,
    validate_date,
    validate_topic,
)
from journalctl.models.conversation import Message
from journalctl.storage.database import DatabaseStorage
from journalctl.storage.exceptions import ConversationNotFoundError, TopicNotFoundError
from journalctl.storage.search_index import SearchIndex
from journalctl.tools.constants import (
    DEFAULT_CONVERSATIONS_LIMIT,
    KEEP_ROLES,
    MAX_CONVERSATIONS_RESULTS,
    MAX_MESSAGES_PER_CONVERSATION,
    MAX_MSG_CHARS,
)
from journalctl.tools.errors import invalid_date, invalid_topic, not_found, validation_error


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


def register(
    mcp: FastMCP,
    storage: DatabaseStorage,
    index: SearchIndex,
) -> None:
    """Register conversation tools on the MCP server."""

    @mcp.tool()
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
        title = sanitize_label(title, max_len=100)
        source = sanitize_label(source)
        summary = sanitize_freetext(summary)
        if tags:
            tags = [s for t in tags if (s := sanitize_label(t))]
        if date:
            try:
                validate_date(date)
            except ValueError:
                return invalid_date(date)

        if len(messages) > MAX_MESSAGES_PER_CONVERSATION:
            return validation_error(
                f"Too many messages: max {MAX_MESSAGES_PER_CONVERSATION}, got {len(messages)}"
            )

        # Only keep human-readable turns. Tool calls, tool results, and system
        # messages are infrastructure noise — not part of the conversation record.
        try:
            parsed_messages = [
                Message(
                    role=m.get("role", "user"),
                    content=sanitize_freetext(m.get("content", ""))[:MAX_MSG_CHARS],
                    timestamp=m.get("timestamp"),
                )
                for m in messages
                if m.get("role") in KEEP_ROLES and m.get("content", "").strip()
            ]
        except (TypeError, AttributeError) as e:
            return validation_error(
                "Invalid message format — each message"
                f" must be a dict with 'role' and 'content': {e}"
            )

        if not parsed_messages:
            return validation_error("No user/assistant messages found after filtering.")

        try:
            conv_id, saved_summary, is_update = storage.save_conversation(
                topic=topic,
                title=title,
                messages=parsed_messages,
                summary=summary,
                source=source,
                tags=tags,
                date=date,
            )
        except TopicNotFoundError:
            return not_found("Topic", topic)

        # Update FTS5 index
        message_content = "\n\n".join(m.content for m in parsed_messages)
        from datetime import date as date_cls  # noqa: PLC0415

        today = date_cls.today().isoformat()
        index.upsert_conversation(
            conversation_id=conv_id,
            topic=topic,
            title=title,
            summary=saved_summary,
            tags=tags or [],
            updated=date or today,
            message_content=message_content,
        )

        return {
            "status": "updated" if is_update else "saved",
            "conversation_id": conv_id,
            "summary": saved_summary,
            "topic": topic,
            "title": title,
        }

    @mcp.tool()
    async def journal_list_conversations(
        topic_prefix: str | None = None,
        limit: int = DEFAULT_CONVERSATIONS_LIMIT,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Browse saved conversations by topic — 'what conversations have we had about X?'

        Use this tool when the user wants to browse a list of chats, not find
        specific content within them. For keyword search across both entries AND
        conversations, use journal_search instead.

        Returns titles, dates, and summaries.

        To read a full transcript, pass the 'id' field to journal_read_conversation.

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
        total = storage.count_conversations(topic_prefix=topic_prefix)
        conversations = storage.list_conversations(
            topic_prefix=topic_prefix, limit=limit, offset=offset
        )
        return {
            "conversations": [c.model_dump() for c in conversations],
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    @mcp.tool()
    async def journal_read_conversation(
        conversation_id: int,
        preview: bool = False,
    ) -> dict[str, Any]:
        """Read the full transcript of a saved conversation.

        Use the 'id' from journal_list_conversations or journal_search results.

        Args:
            conversation_id: The conversation's ID (from list or search results).
            preview: If True, return only first and last 3 messages instead of
                     the full transcript. Use for long conversations to avoid
                     consuming the entire context window.

        Returns:
            metadata (title, topic, summary, dates, participants),
            content (full transcript as markdown), messages_shown, messages_total.
        """

        try:
            meta, messages = storage.read_conversation_by_id(conversation_id, preview=preview)
        except ConversationNotFoundError:
            return not_found("Conversation", conversation_id)
        content = _format_messages_as_markdown(meta.title, messages)
        return {
            "metadata": meta.model_dump(),
            "content": content,
            "preview": preview,
            "messages_shown": len(messages),
            "messages_total": meta.message_count,
        }
