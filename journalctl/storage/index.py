"""FTS5 SQLite search index.

The index is a disposable acceleration layer over the canonical
DatabaseStorage tables. It can be deleted and rebuilt from the
database at any time via rebuild_from_db().

Timeline and knowledge files are excluded from indexing.
"""

import json
import logging
import sqlite3
import time
from pathlib import Path

from journalctl.models.entry import SearchResult

logger = logging.getLogger(__name__)


SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source_key  TEXT NOT NULL UNIQUE,
    doc_type    TEXT NOT NULL,
    topic       TEXT NOT NULL,
    title       TEXT NOT NULL,
    description TEXT DEFAULT '',
    tags        TEXT DEFAULT '[]',
    updated     TEXT,
    content     TEXT NOT NULL,
    indexed_at  INTEGER NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
    title, description, tags, content,
    content='documents', content_rowid='id',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
    INSERT INTO documents_fts(rowid, title, description, tags, content)
    VALUES (new.id, new.title, new.description, new.tags, new.content);
END;

CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
    INSERT INTO documents_fts(documents_fts, rowid, title, description, tags, content)
    VALUES ('delete', old.id, old.title, old.description, old.tags, old.content);
END;

CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN
    INSERT INTO documents_fts(documents_fts, rowid, title, description, tags, content)
    VALUES ('delete', old.id, old.title, old.description, old.tags, old.content);
    INSERT INTO documents_fts(rowid, title, description, tags, content)
    VALUES (new.id, new.title, new.description, new.tags, new.content);
END;

CREATE INDEX IF NOT EXISTS idx_documents_topic   ON documents(topic);
CREATE INDEX IF NOT EXISTS idx_documents_type    ON documents(doc_type);
CREATE INDEX IF NOT EXISTS idx_documents_updated ON documents(updated);
"""


class SearchIndex:
    """FTS5 search index. Disposable — rebuilt from DatabaseStorage."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
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
            self._init_schema()
        return self._conn

    def _init_schema(self) -> None:
        self._conn.executescript(SCHEMA)  # type: ignore[union-attr]

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Upsert (called after writes to DatabaseStorage)
    # ------------------------------------------------------------------

    def upsert_entry(
        self,
        entry_id: int,
        topic: str,
        title: str,
        date: str,
        content: str,
        context: str | None,
        tags: list[str],
    ) -> None:
        """Index a single journal entry."""
        source_key = f"entry:{entry_id}"
        full_content = content
        if context:
            full_content = f"{content}\n\n{context}"

        self.conn.execute(
            """
            INSERT INTO documents
                (source_key, doc_type, topic, title, description,
                 tags, updated, content, indexed_at)
            VALUES (?, 'entry', ?, ?, '', ?, ?, ?, ?)
            ON CONFLICT(source_key) DO UPDATE SET
                topic=excluded.topic,
                title=excluded.title,
                tags=excluded.tags,
                updated=excluded.updated,
                content=excluded.content,
                indexed_at=excluded.indexed_at
            """,
            (
                source_key,
                topic,
                title,
                json.dumps(tags),
                date,
                full_content,
                int(time.time()),
            ),
        )
        self.conn.commit()

    def upsert_conversation(
        self,
        conversation_id: int,
        topic: str,
        title: str,
        summary: str,
        tags: list[str],
        updated: str,
        message_content: str,
    ) -> None:
        """Index a conversation (summary + concatenated messages)."""
        source_key = f"conversation:{conversation_id}"
        full_content = f"{summary}\n\n{message_content}" if summary else message_content

        self.conn.execute(
            """
            INSERT INTO documents
                (source_key, doc_type, topic, title, description,
                 tags, updated, content, indexed_at)
            VALUES (?, 'conversation', ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_key) DO UPDATE SET
                topic=excluded.topic,
                title=excluded.title,
                description=excluded.description,
                tags=excluded.tags,
                updated=excluded.updated,
                content=excluded.content,
                indexed_at=excluded.indexed_at
            """,
            (
                source_key,
                topic,
                title,
                summary,
                json.dumps(tags),
                updated,
                full_content,
                int(time.time()),
            ),
        )
        self.conn.commit()

    def remove_entry(self, entry_id: int) -> None:
        """Remove an entry from the index."""
        self.conn.execute(
            "DELETE FROM documents WHERE source_key = ?",
            (f"entry:{entry_id}",),
        )
        self.conn.commit()

    def remove_conversation(self, conversation_id: int) -> None:
        """Remove a conversation from the index."""
        self.conn.execute(
            "DELETE FROM documents WHERE source_key = ?",
            (f"conversation:{conversation_id}",),
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        topic_prefix: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Full-text search using FTS5."""
        if not query or not query.strip():
            return []

        query = query.strip()[:500]  # cap to prevent oversized FTS5 expressions

        sql = """
            SELECT
                d.source_key,
                d.doc_type,
                d.topic,
                d.title,
                snippet(documents_fts, 3, '<mark>', '</mark>', '...', 40) AS snippet,
                rank,
                d.updated AS date
            FROM documents_fts
            JOIN documents d ON d.id = documents_fts.rowid
            WHERE documents_fts MATCH ?
        """
        params: list[str | int] = [query]

        if topic_prefix:
            sql += " AND d.topic LIKE ?"
            params.append(f"{topic_prefix}%")

        if date_from:
            sql += " AND d.updated >= ?"
            params.append(date_from)

        if date_to:
            sql += " AND d.updated <= ?"
            params.append(date_to)

        sql += " ORDER BY rank LIMIT ?"
        params.append(limit)

        try:
            rows = self.conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError as e:
            logger.warning("FTS5 query failed: query=%s error=%s", query, e)
            msg = (
                "Invalid search query syntax. Use simple keywords or "
                'FTS5 operators: AND, OR, NOT, "exact phrase", prefix*'
            )
            raise ValueError(msg) from None

        results = []
        for row in rows:
            source_key = row["source_key"]
            entry_id = None
            conversation_id = None
            if source_key.startswith("entry:"):
                entry_id = int(source_key[6:])
            elif source_key.startswith("conversation:"):
                conversation_id = int(source_key[13:])

            results.append(
                SearchResult(
                    source_key=source_key,
                    doc_type=row["doc_type"],
                    topic=row["topic"],
                    title=row["title"],
                    snippet=row["snippet"],
                    rank=row["rank"],
                    date=row["date"],
                    entry_id=entry_id,
                    conversation_id=conversation_id,
                )
            )
        return results

    # ------------------------------------------------------------------
    # Rebuild
    # ------------------------------------------------------------------

    def rebuild_from_db(self, db_storage: "DatabaseStorage") -> dict[str, int | float]:  # type: ignore[name-defined]  # noqa: F821
        """Full rebuild from canonical DatabaseStorage.

        Clears the FTS5 index and re-indexes all entries and conversations.
        """

        start = time.time()

        self.conn.execute("DELETE FROM documents")
        self.conn.commit()

        count = 0

        # Index all entries
        entry_rows = db_storage.conn.execute(
            """
            SELECT e.id, e.date, e.content, e.context, e.tags,
                   t.path AS topic, t.title
            FROM entries e
            JOIN topics t ON t.id = e.topic_id
            """
        ).fetchall()

        for r in entry_rows:
            try:
                self.upsert_entry(
                    entry_id=r["id"],
                    topic=r["topic"],
                    title=r["title"],
                    date=r["date"],
                    content=r["content"],
                    context=r["context"],
                    tags=json.loads(r["tags"] or "[]"),
                )
                count += 1
            except (sqlite3.Error, KeyError, json.JSONDecodeError, AssertionError) as e:
                logger.warning("Failed to index entry %s: %s", r["id"], e)

        # Index all conversations (concatenate messages)
        conv_rows = db_storage.conn.execute(
            """
            SELECT c.id, c.title, c.summary, c.tags, c.updated_at,
                   t.path AS topic
            FROM conversations c
            JOIN topics t ON t.id = c.topic_id
            """
        ).fetchall()

        for r in conv_rows:
            msg_rows = db_storage.conn.execute(
                "SELECT content FROM messages WHERE conversation_id = ? ORDER BY position",
                (r["id"],),
            ).fetchall()
            message_content = "\n\n".join(m["content"] for m in msg_rows)

            try:
                self.upsert_conversation(
                    conversation_id=r["id"],
                    topic=r["topic"],
                    title=r["title"],
                    summary=r["summary"] or "",
                    tags=json.loads(r["tags"] or "[]"),
                    updated=r["updated_at"],
                    message_content=message_content,
                )
                count += 1
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to index conversation %s: %s", r["id"], e)

        duration = time.time() - start
        return {
            "documents_indexed": count,
            "duration_seconds": round(duration, 2),
        }
