"""MCP tools: journal_save_conversation, journal_list_conversations,
journal_read_conversation."""

from mcp.server.fastmcp import FastMCP

from journalctl.models.entry import Message, sanitize_freetext, sanitize_label, validate_topic
from journalctl.storage.index import SearchIndex
from journalctl.storage.markdown import MarkdownStorage


def register(
    mcp: FastMCP,
    storage: MarkdownStorage,
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
        """Save a full conversation transcript.

        Idempotent: re-saving the same topic + title overwrites the
        file. Git (via daily cron) preserves all versions. Also
        auto-generates a summary entry in the relevant topic file.

        Args:
            topic: Topic this conversation relates to.
            title: Descriptive title for the conversation.
            messages: List of message dicts with keys:
                      role ('user'/'assistant'), content (str),
                      and optional timestamp (str).
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
        try:
            parsed_messages = [
                Message(
                    role=m.get("role", "user"),
                    content=sanitize_freetext(m.get("content", "")),
                    timestamp=m.get("timestamp"),
                )
                for m in messages
            ]
        except (TypeError, AttributeError) as e:
            msg = (
                "Invalid message format — each message"
                + f" must be a dict with 'role' and 'content': {e}"
            )
            raise ValueError(msg) from e

        # Check if this is an update
        conv_path = storage.conversation_path(topic, title)
        is_update = conv_path.exists()

        # Save the conversation file
        path, auto_summary = storage.save_conversation(
            topic=topic,
            title=title,
            messages=parsed_messages,
            source=source,
            tags=tags,
            thread=thread,
            thread_seq=thread_seq,
            summary=summary,
        )

        # Upsert summary entry in the topic file
        from datetime import date as date_cls

        conv_date = date_cls.today().isoformat()
        storage.upsert_conversation_summary(
            topic=topic,
            conv_title=title,
            summary=auto_summary,
            conv_date=conv_date,
        )

        # Update FTS5 index for both files
        index.upsert_file(path)
        topic_path = storage.topic_path(topic)
        if topic_path.exists():
            index.upsert_file(topic_path)

        return {
            "status": "updated" if is_update else "saved",
            "path": str(path),
            "summary": auto_summary,
            "topic": topic,
            "title": title,
        }

    @mcp.tool()
    async def journal_list_conversations(
        topic_prefix: str | None = None,
    ) -> dict:
        """List archived conversations.

        Args:
            topic_prefix: Filter by topic prefix (e.g. 'work').
                          If omitted, lists all conversations.

        Returns:
            List of conversations with title, date, summary,
            message count.
        """
        if topic_prefix:
            topic_prefix = topic_prefix.rstrip("/") or None
        if topic_prefix:
            validate_topic(topic_prefix)
        conversations = storage.list_conversations(
            topic_prefix=topic_prefix,
        )
        return {
            "conversations": [c.model_dump() for c in conversations],
            "count": len(conversations),
        }

    @mcp.tool()
    async def journal_read_conversation(
        topic: str,
        title: str,
    ) -> dict:
        """Read a specific archived conversation.

        Args:
            topic: Topic the conversation is under.
            title: Title of the conversation (used in the filename).

        Returns:
            Full conversation metadata and transcript.
        """
        validate_topic(topic)
        meta, content = storage.read_conversation(topic, title)
        return {
            "metadata": meta.model_dump(),
            "content": content,
        }
