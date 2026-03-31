"""SQLite canonical storage layer.

Replaces MarkdownStorage as the source of truth. All journal data
(topics, entries, conversations, messages) lives in journal.db.

Markdown files are generated views — produced by storage/export.py
for MkDocs and git. They are NOT the source of truth.

Conversation JSON archives are written alongside DB rows as the
archival record (designed for S3 offload).

Conversation CRUD lives in storage/conversations.py (ConversationMixin);
DatabaseStorage inherits from it so all call sites are unchanged.
"""

import json
import logging
import re
import sqlite3
import threading
from datetime import date as date_cls
from pathlib import Path

from journalctl.storage.constants import (
    DB_BUSY_TIMEOUT_MS,
    MAX_KNOWLEDGE_FILE_SIZE,
    SNIPPET_PREVIEW_LEN,
)
from journalctl.storage.conversations import ConversationMixin
from journalctl.storage.exceptions import TopicNotFoundError

_KNOWLEDGE_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")


def _escape_like(s: str) -> str:
    """Escape SQL LIKE metacharacters (!, %, _) using ! as the escape char.

    Use with ``ESCAPE '!'`` in the SQL clause. validate_topic() already
    prevents wildcards in topic paths; this is a defense-in-depth measure.
    """
    return s.replace("!", "!!").replace("%", "!%").replace("_", "!_")


from journalctl.core.validation import validate_topic  # noqa: E402
from journalctl.models.journal import Entry, TopicMeta  # noqa: E402

logger = logging.getLogger(__name__)


def _build_entries_where_clause(
    topic_id: int,
    date_from: str | None,
    date_to: str | None,
) -> tuple[str, list[str | int]]:
    """Build the shared WHERE predicate for entries queries.

    All clause fragments are hardcoded constants — user values are bound
    via parameterized placeholders, so string concatenation here is safe.
    """
    clauses = ["topic_id = ?", "deleted_at IS NULL"]
    params: list[str | int] = [topic_id]
    if date_from:
        clauses.append("date >= ?")
        params.append(date_from)
    if date_to:
        clauses.append("date <= ?")
        params.append(date_to)
    return " AND ".join(clauses), params


def _calculate_pagination(
    total: int,
    limit: int | None,
    offset: int,
    date_from: str | None,
) -> tuple[int | None, int]:
    """Return (sql_limit, sql_offset) for a read_entries query.

    "Last N" semantics: when no explicit offset and no date_from filter,
    skip to the tail of the result set so the N most recent entries are returned.
    """
    sql_limit: int | None = None
    sql_offset: int = 0
    if limit is not None and limit > 0:
        if offset == 0 and not date_from:
            sql_offset = max(0, total - limit)
            sql_limit = limit
        else:
            sql_offset = offset
            sql_limit = limit
    elif offset > 0:
        sql_offset = offset
    return sql_limit, sql_offset


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
    reasoning       TEXT,
    conversation_id INTEGER REFERENCES conversations(id),
    tags            TEXT DEFAULT '[]',
    position        INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    deleted_at      TEXT,
    indexed_at      TEXT
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

CREATE INDEX IF NOT EXISTS idx_entries_topic      ON entries(topic_id);
CREATE INDEX IF NOT EXISTS idx_entries_date       ON entries(date);
CREATE INDEX IF NOT EXISTS idx_entries_conv       ON entries(conversation_id);
CREATE INDEX IF NOT EXISTS idx_entries_indexed_at ON entries(indexed_at);
CREATE INDEX IF NOT EXISTS idx_messages_conv    ON messages(conversation_id, position);
CREATE INDEX IF NOT EXISTS idx_conv_topic       ON conversations(topic_id);
CREATE INDEX IF NOT EXISTS idx_conv_slug        ON conversations(topic_id, slug);
CREATE INDEX IF NOT EXISTS idx_topics_updated   ON topics(updated_at DESC);
"""


class DatabaseStorage(ConversationMixin):
    """Canonical SQLite storage for all journal data."""

    def __init__(self, db_path: Path, journal_root: Path) -> None:
        self.db_path = db_path
        self.journal_root = journal_root
        self.conversations_json_dir = journal_root / "conversations_json"
        self._conn: sqlite3.Connection | None = None
        self._conn_lock = threading.Lock()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            with self._conn_lock:
                if self._conn is None:
                    self._conn = sqlite3.connect(
                        str(self.db_path),
                        check_same_thread=False,
                    )
                    self._conn.row_factory = sqlite3.Row
                    self._conn.execute(f"PRAGMA busy_timeout={DB_BUSY_TIMEOUT_MS}")
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

    def _get_topic_id(self, topic: str) -> int:
        """Return topic_id. Raises TopicNotFoundError if the topic does not exist."""
        topic = validate_topic(topic)
        row = self.conn.execute("SELECT id FROM topics WHERE path = ?", (topic,)).fetchone()
        if row:
            return int(row["id"])
        msg = f"Topic '{topic}' not found — create it first with journal_create_topic"
        raise TopicNotFoundError(msg)

    def create_topic(
        self,
        topic: str,
        title: str,
        description: str = "",
        tags: list[str] | None = None,
        created_at: str | None = None,
    ) -> int:
        """Create a new topic. Returns topic_id. Raises ValueError if duplicate."""
        topic = validate_topic(topic)
        today = date_cls.today().isoformat()
        created = created_at or today
        try:
            with self.conn:
                cur = self.conn.execute(
                    """
                    INSERT INTO topics (path, title, description, tags, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (topic, title, description, json.dumps(tags or []), created, today),
                )
            if cur.lastrowid is None:
                raise RuntimeError("INSERT topics failed: no rowid")
            return cur.lastrowid
        except sqlite3.IntegrityError as e:
            msg = f"Topic '{topic}' already exists"
            raise ValueError(msg) from e

    def list_topics(
        self,
        topic_prefix: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[TopicMeta]:
        """List topics, sorted by most recently updated."""
        sql = """
            SELECT t.id, t.path, t.title, t.description, t.tags,
                   t.created_at, t.updated_at,
                   COUNT(e.id) AS entry_count
            FROM topics t
            LEFT JOIN entries e ON e.topic_id = t.id AND e.deleted_at IS NULL
        """
        params: list[str | int] = []
        if topic_prefix:
            topic_prefix = validate_topic(topic_prefix)
            sql += " WHERE t.path LIKE ? ESCAPE '!'"
            params += [f"{_escape_like(topic_prefix)}%"]
        sql += " GROUP BY t.id ORDER BY t.updated_at DESC"
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])

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

    def count_topics(self, topic_prefix: str | None = None) -> int:
        """Return total topic count, optionally filtered by prefix."""
        sql = "SELECT COUNT(*) FROM topics t"
        params: list[str | int] = []
        if topic_prefix:
            topic_prefix = validate_topic(topic_prefix)
            sql += " WHERE t.path LIKE ? ESCAPE '!'"
            params += [f"{_escape_like(topic_prefix)}%"]
        return int(self.conn.execute(sql, params).fetchone()[0])

    def get_topic(self, topic: str) -> TopicMeta | None:
        """Get a single topic by path."""
        topic = validate_topic(topic)
        row = self.conn.execute(
            """
            SELECT t.id, t.path, t.title, t.description, t.tags,
                   t.created_at, t.updated_at,
                   COUNT(e.id) AS entry_count
            FROM topics t
            LEFT JOIN entries e ON e.topic_id = t.id AND e.deleted_at IS NULL
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
        reasoning: str | None = None,
        tags: list[str] | None = None,
        date: str | None = None,
        commit: bool = True,
    ) -> tuple[int, int]:
        """Append a dated entry to a topic. Raises TopicNotFoundError if topic missing.

        Returns (entry_id, total_entry_count).

        Pass commit=False to defer the commit (e.g. when the caller wants to
        batch the FTS index write into the same transaction for atomicity).
        """
        topic_id = self._get_topic_id(topic)
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
                INSERT INTO entries (topic_id, date, content, reasoning, tags, position, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (topic_id, d, content, reasoning, json.dumps(tags or []), position, now, now),
            )
            if cur.lastrowid is None:
                raise RuntimeError("INSERT entries failed: no rowid")
            entry_id: int = cur.lastrowid

            self.conn.execute(
                "UPDATE topics SET updated_at = ? WHERE id = ?",
                (d, topic_id),
            )
            if commit:
                self.conn.commit()
        except (sqlite3.Error, RuntimeError):
            self.conn.rollback()
            raise

        count: int = self.conn.execute(
            "SELECT COUNT(*) FROM entries WHERE topic_id = ?", (topic_id,)
        ).fetchone()[0]

        return entry_id, count

    def read_entries(
        self,
        topic: str,
        limit: int | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        offset: int = 0,
    ) -> tuple[TopicMeta, list[Entry], int]:
        """Read entries for a topic, most-recent-last.

        Returns (TopicMeta, entries, total_matching). Raises TopicNotFoundError if topic missing.
        """
        meta = self.get_topic(topic)
        if meta is None:
            msg = f"Topic '{topic}' not found"
            raise TopicNotFoundError(msg)

        if meta.id is None:
            raise RuntimeError(f"Topic '{topic}' has no database ID — storage may be corrupt")

        where, where_params = _build_entries_where_clause(meta.id, date_from, date_to)

        total: int = self.conn.execute(
            "SELECT COUNT(*) FROM entries WHERE " + where,  # noqa: S608
            where_params,
        ).fetchone()[0]

        sql_limit, sql_offset = _calculate_pagination(total, limit, offset, date_from)

        data_sql = (
            "SELECT id, date, content, reasoning, conversation_id, tags, position"  # noqa: S608
            " FROM entries WHERE " + where + " ORDER BY date ASC, position ASC"
        )
        data_params: list[str | int] = list(where_params)
        if sql_limit is not None:
            data_sql += " LIMIT ? OFFSET ?"
            data_params.extend([sql_limit, sql_offset])
        elif sql_offset > 0:
            data_sql += " LIMIT -1 OFFSET ?"
            data_params.append(sql_offset)

        rows = self.conn.execute(data_sql, data_params).fetchall()
        entries = [
            Entry(
                id=r["id"],
                date=r["date"],
                content=r["content"],
                reasoning=r["reasoning"],
                conversation_id=r["conversation_id"],
                tags=json.loads(r["tags"] or "[]"),
            )
            for r in rows
        ]

        return meta, entries, total

    def update_entry(
        self,
        entry_id: int,
        content: str | None = None,
        reasoning: str | None = None,
        mode: str = "replace",
        date: str | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """Update an entry by its stable ID.

        Args:
            entry_id: Stable integer ID from entries table.
            content: New content string (None = leave unchanged).
            reasoning: New reasoning string (None = leave unchanged).
            mode: 'replace' overwrites content; 'append' adds to it.
            date: New date string YYYY-MM-DD (None = leave unchanged).
            tags: New tags list (None = leave unchanged).
        """
        row = self.conn.execute(
            "SELECT id, content, reasoning, topic_id, date, tags FROM entries WHERE id = ? AND deleted_at IS NULL",
            (entry_id,),
        ).fetchone()
        if not row:
            msg = f"Entry id {entry_id} not found"
            raise IndexError(msg)

        # Resolve content
        if content is not None:
            if mode == "replace":
                new_content = content
            elif mode == "append":
                new_content = f"{row['content']}\n\n{content}".strip()
            else:
                msg = f"Invalid mode '{mode}'. Use 'replace' or 'append'."
                raise ValueError(msg)
        else:
            new_content = row["content"]

        # Resolve reasoning
        if reasoning is not None:
            if mode == "append" and row["reasoning"]:
                new_reasoning = f"{row['reasoning']}\n\n{reasoning}".strip()
            else:
                new_reasoning = reasoning
        else:
            new_reasoning = row["reasoning"]

        new_date = date or row["date"]
        new_tags = json.dumps(tags) if tags is not None else row["tags"]
        now = date_cls.today().isoformat()

        with self.conn:
            self.conn.execute(
                "UPDATE entries SET content = ?, reasoning = ?, date = ?, tags = ?, updated_at = ? WHERE id = ?",
                (new_content, new_reasoning, new_date, new_tags, now, entry_id),
            )
            self.conn.execute(
                "UPDATE topics SET updated_at = ? WHERE id = ?",
                (now, row["topic_id"]),
            )

    def delete_entry(self, entry_id: int) -> int:
        """Soft-delete an entry. Returns the topic_id for index cleanup.

        Raises IndexError if entry not found or already deleted.
        """
        row = self.conn.execute(
            "SELECT id, topic_id FROM entries WHERE id = ? AND deleted_at IS NULL",
            (entry_id,),
        ).fetchone()
        if not row:
            msg = f"Entry id {entry_id} not found"
            raise IndexError(msg)

        now = date_cls.today().isoformat()
        topic_id = row["topic_id"]
        with self.conn:
            self.conn.execute(
                "UPDATE entries SET deleted_at = ?, updated_at = ? WHERE id = ?",
                (now, now, entry_id),
            )
            self.conn.execute(
                "UPDATE topics SET updated_at = ? WHERE id = ?",
                (now, topic_id),
            )
        return int(topic_id)

    def mark_entry_indexed(self, entry_id: int) -> None:
        """Stamp indexed_at = today on an entry after a successful embedding.

        Used by journal_append and journal_reindex to track which entries
        have up-to-date semantic embeddings.  The watermark lets
        journal_reindex skip already-indexed entries and resume after
        interruption.
        """
        now = date_cls.today().isoformat()
        with self.conn:
            self.conn.execute(
                "UPDATE entries SET indexed_at = ? WHERE id = ?",
                (now, entry_id),
            )

    def get_entries_by_date_range(
        self,
        date_from: str,
        date_to: str,
    ) -> list[dict]:
        """Get entries and conversations updated within a date range.

        Used by journal_briefing and journal_timeline.
        Returns lightweight dicts (no reasoning field for brevity).
        """
        entry_rows = self.conn.execute(
            """
            SELECT e.id, e.date, e.content, e.tags, t.path AS topic, t.title,
                   'entry' AS doc_type
            FROM entries e
            JOIN topics t ON t.id = e.topic_id
            WHERE e.date >= ? AND e.date <= ? AND e.deleted_at IS NULL
                  AND e.conversation_id IS NULL
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
            content = r["content"] or ""
            first_line = content.split("\n", 1)[0][:80]
            results.append(
                {
                    "entry_id": r["id"],
                    "conversation_id": None,
                    "doc_type": r["doc_type"],
                    "topic": r["topic"],
                    "title": first_line if first_line else r["title"],
                    "description": content[:SNIPPET_PREVIEW_LEN],
                    "tags": json.loads(r["tags"] or "[]"),
                    "updated": r["date"],
                }
            )
        for r in conv_rows:
            results.append(
                {
                    "entry_id": None,
                    "conversation_id": r["id"],
                    "doc_type": r["doc_type"],
                    "topic": r["topic"],
                    "title": r["title"],
                    "description": r["content"][:SNIPPET_PREVIEW_LEN] if r["content"] else "",
                    "tags": json.loads(r["tags"] or "[]"),
                    "updated": r["date"],
                }
            )

        results.sort(
            key=lambda x: (
                x["updated"],
                x["doc_type"],
                x.get("entry_id") or 0,
                x.get("conversation_id") or 0,
            ),
        )
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
    # Knowledge (still file-based)
    # ------------------------------------------------------------------

    def read_knowledge(self, name: str) -> str:
        """Read a knowledge file (e.g. user-profile). Still file-based."""
        if not _KNOWLEDGE_NAME_PATTERN.match(name):
            raise ValueError(f"Invalid knowledge file name: {name!r}")
        path = self.journal_root / "knowledge" / f"{name}.md"
        if not path.exists():
            return ""
        if path.stat().st_size > MAX_KNOWLEDGE_FILE_SIZE:
            raise ValueError(
                f"Knowledge file '{name}' exceeds size limit ({MAX_KNOWLEDGE_FILE_SIZE // (1024 * 1024)} MB)"
            )
        return path.read_text(encoding="utf-8")

    # ------------------------------------------------------------------
    # Query helpers used by tool layer (keeps SQL out of tools/)
    # ------------------------------------------------------------------

    def get_active_entry_ids(self, entry_ids: set[int]) -> set[int]:
        """Return the subset of entry_ids that exist and are not soft-deleted."""
        if not entry_ids:
            return set()
        placeholders = ",".join("?" * len(entry_ids))
        rows = self.conn.execute(
            "SELECT id FROM entries"  # noqa: S608
            f" WHERE id IN ({placeholders}) AND deleted_at IS NULL",
            list(entry_ids),
        ).fetchall()
        return {r["id"] for r in rows}

    def get_entries_brief(self, entry_ids: set[int]) -> dict[int, dict[str, str]]:
        """Return topic, date, and first line of content for a batch of entries.

        Used to enrich semantic search results with metadata from the DB.
        """
        if not entry_ids:
            return {}
        placeholders = ",".join("?" * len(entry_ids))
        rows = self.conn.execute(
            "SELECT e.id, t.path AS topic, e.date, e.content"  # noqa: S608
            f" FROM entries e JOIN topics t ON t.id = e.topic_id"
            f" WHERE e.id IN ({placeholders}) AND e.deleted_at IS NULL",
            list(entry_ids),
        ).fetchall()
        result: dict[int, dict[str, str]] = {}
        for r in rows:
            content = r["content"] or ""
            first_line = content.split("\n", 1)[0][:80]
            result[r["id"]] = {
                "topic": r["topic"],
                "date": r["date"],
                "title": first_line,
            }
        return result

    def get_entry_content(self, entry_id: int) -> str | None:
        """Return content of a non-deleted entry, or None if not found."""
        row = self.conn.execute(
            "SELECT content FROM entries WHERE id = ? AND deleted_at IS NULL",
            (entry_id,),
        ).fetchone()
        return row["content"] if row else None

    def get_entry_with_topic(self, entry_id: int) -> sqlite3.Row | None:
        """Return entry + topic columns needed for FTS/embedding re-sync."""
        return self.conn.execute(  # type: ignore[no-any-return]
            "SELECT e.content, e.reasoning, e.date, e.tags, t.path, t.title "
            "FROM entries e JOIN topics t ON t.id = e.topic_id WHERE e.id = ?",
            (entry_id,),
        ).fetchone()

    def get_unindexed_entries(self, last_id: int, batch_size: int) -> list[sqlite3.Row]:
        """Return a cursor-paginated batch of entries needing semantic indexing."""
        return self.conn.execute(
            """
            SELECT e.id, e.content, e.tags, e.date, t.path AS topic, t.title
            FROM entries e
            JOIN topics t ON t.id = e.topic_id
            WHERE e.deleted_at IS NULL
              AND (e.indexed_at IS NULL OR e.indexed_at < e.updated_at)
              AND e.id > ?
            ORDER BY e.id
            LIMIT ?
            """,
            (last_id, batch_size),
        ).fetchall()
