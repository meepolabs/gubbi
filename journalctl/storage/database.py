"""SQLite canonical storage layer.

Replaces MarkdownStorage as the source of truth. All journal data
(topics, entries, conversations, messages) lives in journal.db.

Markdown files are generated views — produced by storage/export.py
for MkDocs and git. They are NOT the source of truth.

Conversation JSON archives are written alongside DB rows as the
archival record (designed for S3 offload).
"""

import json
import logging
import sqlite3
from datetime import date as date_cls
from pathlib import Path

from journalctl.models.entry import (
    ConversationMeta,
    Entry,
    Message,
    TopicMeta,
    slugify,
    validate_title,
    validate_topic,
)

logger = logging.getLogger(__name__)


SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS topics (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    path        TEXT UNIQUE NOT NULL,
    title       TEXT NOT NULL,
    description TEXT DEFAULT '',
    tags        TEXT DEFAULT '[]',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS entries (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_id        INTEGER NOT NULL REFERENCES topics(id),
    date            TEXT NOT NULL,
    content         TEXT NOT NULL,
    context         TEXT,
    conversation_id INTEGER REFERENCES conversations(id),
    tags            TEXT DEFAULT '[]',
    position        INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS conversations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_id      INTEGER REFERENCES topics(id),
    title         TEXT NOT NULL,
    slug          TEXT NOT NULL,
    source        TEXT NOT NULL DEFAULT 'claude',
    summary       TEXT DEFAULT '',
    tags          TEXT DEFAULT '[]',
    participants  TEXT DEFAULT '[]',
    message_count INTEGER DEFAULT 0,
    json_path     TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    UNIQUE(topic_id, slug)
);

CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    timestamp       TEXT,
    position        INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_entries_topic    ON entries(topic_id);
CREATE INDEX IF NOT EXISTS idx_entries_date     ON entries(date);
CREATE INDEX IF NOT EXISTS idx_entries_conv     ON entries(conversation_id);
CREATE INDEX IF NOT EXISTS idx_messages_conv    ON messages(conversation_id, position);
CREATE INDEX IF NOT EXISTS idx_conv_topic       ON conversations(topic_id);
CREATE INDEX IF NOT EXISTS idx_conv_slug        ON conversations(topic_id, slug);
CREATE INDEX IF NOT EXISTS idx_topics_updated   ON topics(updated_at DESC);
"""


class DatabaseStorage:
    """Canonical SQLite storage for all journal data."""

    def __init__(self, db_path: Path, journal_root: Path) -> None:
        self.db_path = db_path
        self.journal_root = journal_root
        self.conversations_json_dir = journal_root / "conversations_json"
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._init_schema()
        return self._conn

    def _init_schema(self) -> None:
        self._conn.executescript(SCHEMA)  # type: ignore[union-attr]

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Topics
    # ------------------------------------------------------------------

    def _get_or_create_topic(self, topic: str) -> int:
        """Return topic_id, creating the topic if it doesn't exist."""
        validate_topic(topic)
        row = self.conn.execute("SELECT id FROM topics WHERE path = ?", (topic,)).fetchone()
        if row:
            return int(row["id"])
        title = topic.split("/")[-1].replace("-", " ").title()
        return self.create_topic(topic, title)

    def create_topic(
        self,
        topic: str,
        title: str,
        description: str = "",
        tags: list[str] | None = None,
    ) -> int:
        """Create a new topic. Returns topic_id. Raises FileExistsError if duplicate."""
        validate_topic(topic)
        today = date_cls.today().isoformat()
        try:
            cur = self.conn.execute(
                """
                INSERT INTO topics (path, title, description, tags, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (topic, title, description, json.dumps(tags or []), today, today),
            )
            self.conn.commit()
            if cur.lastrowid is None:
                raise RuntimeError("INSERT topics failed: no rowid")
            return cur.lastrowid
        except sqlite3.IntegrityError as e:
            msg = f"Topic '{topic}' already exists"
            raise ValueError(msg) from e

    def list_topics(
        self,
        topic_prefix: str | None = None,
    ) -> list[TopicMeta]:
        """List topics, sorted by most recently updated."""
        sql = """
            SELECT t.id, t.path, t.title, t.description, t.tags,
                   t.created_at, t.updated_at,
                   COUNT(e.id) AS entry_count
            FROM topics t
            LEFT JOIN entries e ON e.topic_id = t.id
        """
        params: list[str] = []
        if topic_prefix:
            validate_topic(topic_prefix)
            # Safe: validate_topic() enforces [a-z0-9/-] — no SQL wildcards possible
            sql += " WHERE t.path = ? OR t.path LIKE ?"
            params += [topic_prefix, f"{topic_prefix}/%"]
        sql += " GROUP BY t.id ORDER BY t.updated_at DESC"

        rows = self.conn.execute(sql, params).fetchall()
        return [
            TopicMeta(
                id=r["id"],
                topic=r["path"],
                title=r["title"],
                description=r["description"] or "",
                tags=json.loads(r["tags"] or "[]"),
                created=r["created_at"],
                updated=r["updated_at"],
                entry_count=r["entry_count"],
            )
            for r in rows
        ]

    def get_topic(self, topic: str) -> TopicMeta | None:
        """Get a single topic by path."""
        validate_topic(topic)
        row = self.conn.execute(
            """
            SELECT t.id, t.path, t.title, t.description, t.tags,
                   t.created_at, t.updated_at,
                   COUNT(e.id) AS entry_count
            FROM topics t
            LEFT JOIN entries e ON e.topic_id = t.id
            WHERE t.path = ?
            GROUP BY t.id
            """,
            (topic,),
        ).fetchone()
        if not row:
            return None
        return TopicMeta(
            id=row["id"],
            topic=row["path"],
            title=row["title"],
            description=row["description"] or "",
            tags=json.loads(row["tags"] or "[]"),
            created=row["created_at"],
            updated=row["updated_at"],
            entry_count=row["entry_count"],
        )

    # ------------------------------------------------------------------
    # Entries
    # ------------------------------------------------------------------

    def append_entry(
        self,
        topic: str,
        content: str,
        context: str | None = None,
        tags: list[str] | None = None,
        date: str | None = None,
    ) -> tuple[int, int]:
        """Append a dated entry to a topic. Auto-creates topic if needed.

        Returns (entry_id, total_entry_count).
        """
        validate_topic(topic)
        topic_id = self._get_or_create_topic(topic)
        d = date or date_cls.today().isoformat()

        # Determine position (max position in this topic + 1)
        row = self.conn.execute(
            "SELECT COALESCE(MAX(position), 0) FROM entries WHERE topic_id = ?",
            (topic_id,),
        ).fetchone()
        position = (row[0] or 0) + 1

        now = date_cls.today().isoformat()
        try:
            cur = self.conn.execute(
                """
                INSERT INTO entries (topic_id, date, content, context, tags, position, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (topic_id, d, content, context, json.dumps(tags or []), position, now, now),
            )
            if cur.lastrowid is None:
                raise RuntimeError("INSERT entries failed: no rowid")
            entry_id: int = cur.lastrowid

            self.conn.execute(
                "UPDATE topics SET updated_at = ? WHERE id = ?",
                (d, topic_id),
            )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

        count: int = self.conn.execute(
            "SELECT COUNT(*) FROM entries WHERE topic_id = ?", (topic_id,)
        ).fetchone()[0]

        return entry_id, count

    def read_entries(
        self,
        topic: str,
        n: int | None = None,
    ) -> tuple[TopicMeta, list[Entry]]:
        """Read entries for a topic, most-recent-last.

        Returns (TopicMeta, entries). Raises FileNotFoundError if topic missing.
        """
        meta = self.get_topic(topic)
        if meta is None:
            msg = f"Topic '{topic}' not found"
            raise FileNotFoundError(msg)

        sql = """
            SELECT id, date, content, context, conversation_id, tags, position
            FROM entries WHERE topic_id = ?
            ORDER BY date ASC, position ASC
        """
        rows = self.conn.execute(sql, (meta.id,)).fetchall()

        entries = [
            Entry(
                id=r["id"],
                date=r["date"],
                content=r["content"],
                context=r["context"],
                conversation_id=r["conversation_id"],
                tags=json.loads(r["tags"] or "[]"),
            )
            for r in rows
        ]

        if n is not None and n > 0:
            entries = entries[-n:]

        return meta, entries

    def update_entry(
        self,
        entry_id: int,
        content: str,
        context: str | None = None,
        mode: str = "replace",
    ) -> None:
        """Update an entry by its stable ID.

        Args:
            entry_id: Stable integer ID from entries table.
            content: New content string.
            context: New context string (None = leave unchanged).
            mode: 'replace' overwrites content; 'append' adds to it.
        """
        row = self.conn.execute(
            "SELECT id, content, context, topic_id FROM entries WHERE id = ?",
            (entry_id,),
        ).fetchone()
        if not row:
            msg = f"Entry id {entry_id} not found"
            raise IndexError(msg)

        if mode == "replace":
            new_content = content
            new_context = context if context is not None else row["context"]
        elif mode == "append":
            new_content = f"{row['content']}\n\n{content}".strip()
            new_context = (
                f"{row['context']}\n\n{context}".strip()
                if row["context"] and context
                else context or row["context"]
            )
        else:
            msg = f"Invalid mode '{mode}'. Use 'replace' or 'append'."
            raise ValueError(msg)

        now = date_cls.today().isoformat()
        self.conn.execute(
            "UPDATE entries SET content = ?, context = ?, updated_at = ? WHERE id = ?",
            (new_content, new_context, now, entry_id),
        )
        # Update topic updated_at
        self.conn.execute(
            "UPDATE topics SET updated_at = ? WHERE id = ?",
            (now, row["topic_id"]),
        )
        self.conn.commit()

    def get_entries_by_date_range(
        self,
        date_from: str,
        date_to: str,
    ) -> list[dict]:
        """Get entries and conversations updated within a date range.

        Used by journal_briefing and journal_timeline.
        Returns lightweight dicts (no context field for brevity).
        """
        entry_rows = self.conn.execute(
            """
            SELECT e.id, e.date, e.content, e.tags, t.path AS topic, t.title,
                   'entry' AS doc_type
            FROM entries e
            JOIN topics t ON t.id = e.topic_id
            WHERE e.date >= ? AND e.date <= ?
            ORDER BY e.date ASC, e.position ASC
            """,
            (date_from, date_to),
        ).fetchall()

        conv_rows = self.conn.execute(
            """
            SELECT c.id, c.created_at AS date, c.summary AS content, c.tags,
                   t.path AS topic, c.title, 'conversation' AS doc_type
            FROM conversations c
            JOIN topics t ON t.id = c.topic_id
            WHERE c.created_at >= ? AND c.created_at <= ?
            ORDER BY c.created_at ASC
            """,
            (date_from, date_to),
        ).fetchall()

        results = []
        for r in entry_rows:
            results.append(
                {
                    "file_path": f"entry:{r['id']}",
                    "doc_type": r["doc_type"],
                    "topic": r["topic"],
                    "title": r["title"],
                    "description": "",
                    "tags": r["tags"],
                    "updated": r["date"],
                }
            )
        for r in conv_rows:
            results.append(
                {
                    "file_path": f"conversation:{r['id']}",
                    "doc_type": r["doc_type"],
                    "topic": r["topic"],
                    "title": r["title"],
                    "description": r["content"][:120] if r["content"] else "",
                    "tags": r["tags"],
                    "updated": r["date"],
                }
            )

        results.sort(key=lambda x: x["updated"])
        return results

    def get_stats(self) -> dict[str, int]:
        """Return counts for briefing."""
        total_entries = self.conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
        topics = self.conn.execute("SELECT COUNT(*) FROM topics").fetchone()[0]
        conversations = self.conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
        return {
            "total_documents": total_entries + conversations,
            "topics": topics,
            "conversations": conversations,
        }

    # ------------------------------------------------------------------
    # Conversations
    # ------------------------------------------------------------------

    def _write_conversation_json(
        self,
        topic: str,
        slug: str,
        meta: ConversationMeta,
        messages: list[Message],
    ) -> str:
        """Write conversation JSON archive. Returns relative path string."""
        out_dir = self.conversations_json_dir / topic
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{slug}.json"

        payload = {
            "meta": meta.model_dump(exclude={"id"}),
            "messages": [m.model_dump() for m in messages],
        }
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return f"conversations_json/{topic}/{slug}.json"

    def save_conversation(
        self,
        topic: str,
        title: str,
        messages: list[Message],
        source: str = "claude",
        tags: list[str] | None = None,
        summary: str | None = None,
    ) -> tuple[int, str]:
        """Save a conversation. Idempotent — same topic+title overwrites.

        Returns (conversation_id, summary).
        """
        validate_topic(topic)
        validate_title(title)
        slug = slugify(title)
        topic_id = self._get_or_create_topic(topic)
        today = date_cls.today().isoformat()
        auto_summary = summary or self._generate_summary(title, messages)
        participants = sorted({m.role for m in messages})

        meta = ConversationMeta(
            source=source,
            title=title,
            topic=topic,
            tags=tags or [],
            created=today,
            updated=today,
            summary=auto_summary,
            participants=participants,
            message_count=len(messages),
        )
        json_path = self._write_conversation_json(topic, slug, meta, messages)

        conv_id = self._upsert_conversation_record(
            topic_id,
            title,
            slug,
            source,
            auto_summary,
            tags or [],
            participants,
            messages,
            json_path,
            today,
        )
        self._insert_messages(conv_id, messages)
        self._upsert_linked_entry(topic_id, conv_id, title, auto_summary, today)

        self.conn.execute("UPDATE topics SET updated_at = ? WHERE id = ?", (today, topic_id))
        self.conn.commit()
        return conv_id, auto_summary  # type: ignore[return-value]

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
        json_path: str,
        today: str,
    ) -> int:
        """Insert or update the conversations row. Returns conversation_id."""
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
                    json_path=?, updated_at=?
                WHERE id=?
                """,
                (
                    source,
                    summary,
                    json.dumps(tags),
                    json.dumps(participants),
                    len(messages),
                    json_path,
                    today,
                    conv_id,
                ),
            )
            self.conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
            return conv_id

        cur = self.conn.execute(
            """
            INSERT INTO conversations
                (topic_id, title, slug, source, summary, tags, participants,
                 message_count, json_path, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                json_path,
                today,
                today,
            ),
        )
        if cur.lastrowid is None:
            raise RuntimeError("INSERT conversations failed: no rowid")
        return cur.lastrowid

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
        """Upsert a linked entry so the conversation appears in journal_read + timeline."""
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
                    (topic_id, date, content, conversation_id, tags, position, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, (SELECT COALESCE(MAX(position), 0) + 1 FROM entries WHERE topic_id = ?), ?, ?)
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

    def list_conversations(
        self,
        topic_prefix: str | None = None,
    ) -> list[ConversationMeta]:
        """List conversations, optionally filtered by topic prefix."""
        sql = """
            SELECT c.id, c.title, c.slug, c.source, c.summary, c.tags,
                   c.participants, c.message_count,
                   c.created_at, c.updated_at, t.path AS topic
            FROM conversations c
            JOIN topics t ON t.id = c.topic_id
        """
        params: list[str] = []
        if topic_prefix:
            validate_topic(topic_prefix)
            sql += " WHERE t.path = ? OR t.path LIKE ?"
            params += [topic_prefix, f"{topic_prefix}/%"]
        sql += " ORDER BY c.created_at DESC"

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
        """Read a conversation and its messages.

        Returns (ConversationMeta, messages list).
        Raises FileNotFoundError if not found.
        """
        validate_topic(topic)
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
            raise FileNotFoundError(msg)

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

        messages = [
            Message(role=r["role"], content=r["content"], timestamp=r["timestamp"])
            for r in msg_rows
        ]

        return meta, messages

    # ------------------------------------------------------------------
    # Knowledge (still file-based)
    # ------------------------------------------------------------------

    def read_knowledge(self, name: str) -> str:
        """Read a knowledge file (e.g. user-profile). Still file-based."""
        path = self.journal_root / "knowledge" / f"{name}.md"
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _generate_summary(self, title: str, messages: list[Message]) -> str:
        """Generate a simple summary from first user message."""
        first_user = next(
            (m.content for m in messages if m.role == "user"),
            "",
        )
        if not first_user:
            return title

        # Truncate at sentence boundary within 200 chars
        truncated = first_user[:200]
        for punct in (".", "!", "?", "\n"):
            idx = truncated.find(punct)
            if 0 < idx < 180:
                truncated = truncated[: idx + 1]
                break

        return f"Q: {truncated.strip()}"
