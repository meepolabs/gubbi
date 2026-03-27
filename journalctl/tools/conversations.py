"""MCP tools: journal_save_conversation, journal_list_conversations,
journal_read_conversation."""

from typing import Final

from mcp.server.fastmcp import FastMCP

from journalctl.models.entry import Message, sanitize_freetext, sanitize_label, validate_topic
from journalctl.storage.database import DatabaseStorage
from journalctl.storage.index import SearchIndex

_KEEP_ROLES: Final = {"user", "assistant"}
_MAX_MSG_CHARS: Final = 10_000  # per message — prevent runaway tool output from bloating storage


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
        messages: list[dict],
        source: str = "claude",
        tags: list[str] | None = None,
        thread: str | None = None,
        thread_seq: int | None = None,
        summary: str | None = None,
    ) -> dict:
        """Save a full conversation transcript to the journal.

        Offer to save when a meaningful moment happens during a conversation — a
        decision made, a plan formed, a problem solved, or an insight worth keeping.
        Don't wait until the end. Good candidates: planning sessions, deployments,
        life updates, technical breakthroughs, personal reflections.

        Re-saving the same topic + title overwrites the previous version (history
        is preserved automatically). In future conversations on the same topic,
        proactively offer to update the saved record.

        Args:
            topic: Topic this conversation relates to.
            title: Descriptive title for the conversation.
            messages: List of message dicts with keys:
                      role ('user' or 'assistant'), content (str),
                      and optional timestamp (str).
                      Tool calls, tool results, and system messages
                      are filtered automatically — only pass the
                      human-readable turns.
            source: Name of the app or LLM (e.g. 'claude', 'chatgpt').
            tags: Tags for the conversation.
            thread: Thread ID linking related conversations.
            thread_seq: Sequence number within the thread.
            summary: Optional summary override. Auto-generated
                     if not provided.

        Returns:
            File path, summary, and whether it was a new save
            or an update.
        """
        validate_topic(topic)
        title = sanitize_label(title, max_len=100)
        source = sanitize_label(source)
        if tags:
            tags = [sanitize_label(t) for t in tags]
        if summary:
            summary = sanitize_freetext(summary)

        # Only keep human-readable turns. Tool calls, tool results, and system
        # messages are infrastructure noise — not part of the conversation record.
        try:
            parsed_messages = [
                Message(
                    role=m.get("role", "user"),
                    content=sanitize_freetext(m.get("content", ""))[:_MAX_MSG_CHARS],
                    timestamp=m.get("timestamp"),
                )
                for m in messages
                if m.get("role") in _KEEP_ROLES and m.get("content", "").strip()
            ]
        except (TypeError, AttributeError) as e:
            msg = (
                "Invalid message format — each message"
                + f" must be a dict with 'role' and 'content': {e}"
            )
            raise ValueError(msg) from e

        if not parsed_messages:
            raise ValueError("No user/assistant messages found after filtering.")

        # Check if update
        existing = storage.list_conversations(topic_prefix=topic)
        from journalctl.models.entry import slugify  # noqa: PLC0415

        slug = slugify(title)
        is_update = any(slugify(c.title) == slug for c in existing)

        conv_id, auto_summary = storage.save_conversation(
            topic=topic,
            title=title,
            messages=parsed_messages,
            source=source,
            tags=tags,
            thread=thread,
            thread_seq=thread_seq,
            summary=summary,
        )

        # Update FTS5 index
        message_content = "\n\n".join(m.content for m in parsed_messages)
        from datetime import date as date_cls  # noqa: PLC0415

        today = date_cls.today().isoformat()
        index.upsert_conversation(
            conversation_id=conv_id,
            topic=topic,
            title=title,
            summary=auto_summary,
            tags=tags or [],
            created=today,
            updated=today,
            message_content=message_content,
        )

        return {
            "status": "updated" if is_update else "saved",
            "conversation_id": conv_id,
            "summary": auto_summary,
            "topic": topic,
            "title": title,
        }

    @mcp.tool()
    async def journal_list_conversations(
        topic_prefix: str | None = None,
    ) -> dict:
        """Browse saved conversations — "what conversations have we had about X?"

        Use when the user asks to revisit a past conversation or see what exists.
        Returns titles, dates, and summaries. To read a full transcript, follow up
        with journal_read_conversation.

        Args:
            topic_prefix: Filter by topic prefix (e.g. 'work').
                          If omitted, lists all conversations.

        Returns:
            List of conversations with title, date, summary,
            and message count.
        """
        if topic_prefix:
            topic_prefix = topic_prefix.rstrip("/") or None
        if topic_prefix:
            validate_topic(topic_prefix)
        conversations = storage.list_conversations(topic_prefix=topic_prefix)
        return {
            "conversations": [c.model_dump() for c in conversations],
            "count": len(conversations),
        }

    @mcp.tool()
    async def journal_read_conversation(
        topic: str,
        title: str,
    ) -> dict:
        """Read the full transcript of a saved conversation.

        Use after journal_list_conversations to retrieve a specific conversation
        the user wants to revisit. Requires the exact topic and title from
        journal_list_conversations output.

        Args:
            topic: Topic the conversation is under (e.g. 'work/acme').
            title: Title of the conversation (as shown in journal_list_conversations).

        Returns:
            Full conversation metadata and transcript.
        """
        validate_topic(topic)
        meta, messages = storage.read_conversation(topic, title)
        content = _format_messages_as_markdown(title, messages)
        return {
            "metadata": meta.model_dump(),
            "content": content,
        }
