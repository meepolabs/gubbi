"""Conversation storage mixin for DatabaseStorage.

All conversation CRUD lives here; DatabaseStorage inherits via:
    class DatabaseStorage(ConversationMixin): ...

The mixin relies on the host providing:
  - self.conn  (sqlite3.Connection, via DatabaseStorage.conn property)
  - self.conversations_json_dir  (Path)
  - self._get_topic_id(topic) -> int  (raises TopicNotFoundError if missing)
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import date as date_cls
from pathlib import Path

from journalctl.core.validation import slugify, validate_title, validate_topic
from journalctl.models.conversation import ConversationMeta, Message
from journalctl.storage.exceptions import ConversationNotFoundError

logger = logging.getLogger(__name__)


def _escape_like(s: str) -> str:
    """Escape SQL LIKE metacharacters (!, %, _) using ! as the escape char."""
    return s.replace("!", "!!").replace("%", "!%").replace("_", "!_")


class ConversationMixin:
    """Conversation CRUD mixed into DatabaseStorage.

    Type stubs below satisfy type checkers; the real implementations come
    from the host class (DatabaseStorage).
    """

    @property
    def conn(self) -> sqlite3.Connection:
        raise NotImplementedError

    conversations_json_dir: Path

    def _get_topic_id(self, topic: str) -> int:  # pragma: no cover
        raise NotImplementedError

    # ------------------------------------------------------------------
    # JSON archive
    # ------------------------------------------------------------------

    def _write_conversation_json(
        self,
        conv_id: int,
        meta: ConversationMeta,
        messages: list[Message],
    ) -> str:
        """Write conversation JSON archive. Returns relative path string."""
        self.conversations_json_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.conversations_json_dir / f"{conv_id}.json"

        payload = {
            "meta": meta.model_dump(exclude={"id"}),
            "messages": [m.model_dump() for m in messages],
        }
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return f"conversations_json/{conv_id}.json"

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_conversation(
        self,
        topic: str,
        title: str,
        messages: list[Message],
        summary: str,
        source: str = "claude",
        tags: list[str] | None = None,
        date: str | None = None,
    ) -> tuple[int, str, bool]:
        """Save a conversation. Idempotent — same topic+title overwrites.

        Returns (conversation_id, summary, is_update).
        """
        topic = validate_topic(topic)
        title = validate_title(title)
        slug = slugify(title)
        topic_id = self._get_topic_id(topic)
        conversation_date = date or date_cls.today().isoformat()
        today = date_cls.today().isoformat()
        participants = sorted({m.role for m in messages})

        meta = ConversationMeta(
            source=source,
            title=title,
            topic=topic,
            tags=tags or [],
            created=conversation_date,
            updated=today,
            summary=summary,
            participants=participants,
            message_count=len(messages),
        )
        # Upsert DB record first to get the stable conv_id
        conv_id, is_update = self._upsert_conversation_record(
            topic_id,
            title,
            slug,
            source,
            summary,
            tags or [],
            participants,
            messages,
            conversation_date,
        )

        # Write JSON archive keyed by ID (flat, no folder structure)
        json_path = self._write_conversation_json(conv_id, meta, messages)
        self.conn.execute(
            "UPDATE conversations SET json_path = ? WHERE id = ?", (json_path, conv_id)
        )

        self._insert_messages(conv_id, messages)
        self._upsert_linked_entry(topic_id, conv_id, title, summary, conversation_date)

        self.conn.execute(
            "UPDATE topics SET updated_at = ? WHERE id = ?", (conversation_date, topic_id)
        )
        self.conn.commit()
        return conv_id, summary, is_update

    def _upsert_conversation_record(
        self,
        topic_id: int,
        title: str,
        slug: str,
        source: str,
        summary: str,
        tags: list[str],
        participants: list[str],
        messages: list[Message],
        conversation_date: str,
    ) -> tuple[int, bool]:
        """Insert or update the conversations row. Returns (conversation_id, is_update).

        json_path is set separately after writing the archive file.
        """
        today = date_cls.today().isoformat()
        existing = self.conn.execute(
            "SELECT id, created_at FROM conversations WHERE topic_id = ? AND slug = ?",
            (topic_id, slug),
        ).fetchone()

        if existing:
            conv_id: int = existing["id"]
            self.conn.execute(
                """
                UPDATE conversations
                SET source=?, summary=?, tags=?, participants=?, message_count=?,
                    updated_at=?
                WHERE id=?
                """,
                (
                    source,
                    summary,
                    json.dumps(tags),
                    json.dumps(participants),
                    len(messages),
                    today,
                    conv_id,
                ),
            )
            self.conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
            return conv_id, True

        cur = self.conn.execute(
            """
            INSERT INTO conversations
                (topic_id, title, slug, source, summary, tags, participants,
                 message_count, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                topic_id,
                title,
                slug,
                source,
                summary,
                json.dumps(tags),
                json.dumps(participants),
                len(messages),
                conversation_date,
                today,
            ),
        )
        if cur.lastrowid is None:
            raise RuntimeError("INSERT conversations failed: no rowid")
        return cur.lastrowid, False

    def _insert_messages(self, conv_id: int, messages: list[Message]) -> None:
        """Insert all messages for a conversation."""
        self.conn.executemany(
            """
            INSERT INTO messages (conversation_id, role, content, timestamp, position)
            VALUES (?, ?, ?, ?, ?)
            """,
            [(conv_id, m.role, m.content, m.timestamp, i) for i, m in enumerate(messages)],
        )

    def _upsert_linked_entry(
        self,
        topic_id: int,
        conv_id: int,
        title: str,
        summary: str,
        today: str,
    ) -> None:
        """Upsert a linked entry so the conversation appears in journal_read_topic + timeline."""
        content = f"Conversation saved: {title}\n\n{summary}"
        existing = self.conn.execute(
            "SELECT id FROM entries WHERE conversation_id = ?", (conv_id,)
        ).fetchone()

        if existing:
            self.conn.execute(
                "UPDATE entries SET content = ?, updated_at = ? WHERE id = ?",
                (content, today, existing["id"]),
            )
        else:
            self.conn.execute(
                """
                INSERT INTO entries
                    (topic_id, date, content, conversation_id,
                     tags, position, created_at, updated_at)
                VALUES (?, ?, ?, ?,
                    ?,
                    (SELECT COALESCE(MAX(position), 0) + 1 FROM entries
                     WHERE topic_id = ?),
                    ?, ?)
                """,
                (
                    topic_id,
                    today,
                    content,
                    conv_id,
                    json.dumps(["conversation"]),
                    topic_id,
                    today,
                    today,
                ),
            )

    # ------------------------------------------------------------------
    # List / Read
    # ------------------------------------------------------------------

    def list_conversations(
        self,
        topic_prefix: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[ConversationMeta]:
        """List conversations, optionally filtered by topic prefix."""
        sql = """
            SELECT c.id, c.title, c.slug, c.source, c.summary, c.tags,
                   c.participants, c.message_count,
                   c.created_at, c.updated_at, t.path AS topic
            FROM conversations c
            JOIN topics t ON t.id = c.topic_id
        """
        params: list[str | int] = []
        if topic_prefix:
            topic_prefix = validate_topic(topic_prefix)
            sql += " WHERE t.path LIKE ? ESCAPE '!'"
            params += [f"{_escape_like(topic_prefix)}%"]
        sql += " ORDER BY c.created_at DESC"
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])

        rows = self.conn.execute(sql, params).fetchall()
        return [
            ConversationMeta(
                id=r["id"],
                source=r["source"],
                title=r["title"],
                topic=r["topic"],
                tags=json.loads(r["tags"] or "[]"),
                created=r["created_at"],
                updated=r["updated_at"],
                summary=r["summary"] or "",
                participants=json.loads(r["participants"] or "[]"),
                message_count=r["message_count"],
            )
            for r in rows
        ]

    def read_conversation(
        self,
        topic: str,
        title: str,
    ) -> tuple[ConversationMeta, list[Message]]:
        """Read a conversation by topic + title slug.

        Returns (ConversationMeta, messages). Raises ConversationNotFoundError if not found.
        """
        topic = validate_topic(topic)
        slug = slugify(title)

        row = self.conn.execute(
            """
            SELECT c.id, c.title, c.slug, c.source, c.summary, c.tags,
                   c.participants, c.message_count,
                   c.created_at, c.updated_at, t.path AS topic
            FROM conversations c
            JOIN topics t ON t.id = c.topic_id
            WHERE t.path = ? AND c.slug = ?
            """,
            (topic, slug),
        ).fetchone()

        if not row:
            msg = f"Conversation '{title}' not found under '{topic}'"
            raise ConversationNotFoundError(msg)

        meta = ConversationMeta(
            id=row["id"],
            source=row["source"],
            title=row["title"],
            topic=row["topic"],
            tags=json.loads(row["tags"] or "[]"),
            created=row["created_at"],
            updated=row["updated_at"],
            summary=row["summary"] or "",
            participants=json.loads(row["participants"] or "[]"),
            message_count=row["message_count"],
        )

        msg_rows = self.conn.execute(
            """
            SELECT role, content, timestamp FROM messages
            WHERE conversation_id = ?
            ORDER BY position ASC
            """,
            (row["id"],),
        ).fetchall()

        return meta, [
            Message(role=r["role"], content=r["content"], timestamp=r["timestamp"])
            for r in msg_rows
        ]

    def read_conversation_by_id(
        self,
        conversation_id: int,
        preview: bool = False,
    ) -> tuple[ConversationMeta, list[Message]]:
        """Read a conversation by its stable integer ID.

        Args:
            conversation_id: Database primary key.
            preview: If True, return only first 3 and last 3 messages.

        Returns (ConversationMeta, messages). Raises ConversationNotFoundError if not found.
        """
        row = self.conn.execute(
            """
            SELECT c.id, c.title, c.slug, c.source, c.summary, c.tags,
                   c.participants, c.message_count,
                   c.created_at, c.updated_at, t.path AS topic
            FROM conversations c
            JOIN topics t ON t.id = c.topic_id
            WHERE c.id = ?
            """,
            (conversation_id,),
        ).fetchone()

        if not row:
            msg = f"Conversation id {conversation_id} not found"
            raise ConversationNotFoundError(msg)

        meta = ConversationMeta(
            id=row["id"],
            source=row["source"],
            title=row["title"],
            topic=row["topic"],
            tags=json.loads(row["tags"] or "[]"),
            created=row["created_at"],
            updated=row["updated_at"],
            summary=row["summary"] or "",
            participants=json.loads(row["participants"] or "[]"),
            message_count=row["message_count"],
        )

        msg_rows = self.conn.execute(
            """
            SELECT role, content, timestamp FROM messages
            WHERE conversation_id = ?
            ORDER BY position ASC
            """,
            (conversation_id,),
        ).fetchall()

        messages = [
            Message(role=r["role"], content=r["content"], timestamp=r["timestamp"])
            for r in msg_rows
        ]

        if preview and len(messages) > 6:
            messages = messages[:3] + messages[-3:]

        return meta, messages
