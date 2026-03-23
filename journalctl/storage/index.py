"""FTS5 SQLite search index.

The index is a disposable acceleration layer. It can be deleted
and rebuilt from markdown files at any time. Timeline and knowledge
files are excluded from indexing.
"""

import json
import logging
import sqlite3
import time
from pathlib import Path

import frontmatter

from journalctl.models.entry import SearchResult

logger = logging.getLogger(__name__)


SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path   TEXT NOT NULL UNIQUE,
    doc_type    TEXT NOT NULL,
    topic       TEXT NOT NULL,
    title       TEXT NOT NULL,
    description TEXT DEFAULT '',
    tags        TEXT DEFAULT '[]',
    created     TEXT,
    updated     TEXT,
    entry_count INTEGER DEFAULT 0,
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

CREATE INDEX IF NOT EXISTS idx_documents_topic ON documents(topic);
CREATE INDEX IF NOT EXISTS idx_documents_doc_type ON documents(doc_type);
CREATE INDEX IF NOT EXISTS idx_documents_updated ON documents(updated);
"""

# Directories to skip when indexing
SKIP_DIRS = {"timeline", "knowledge", ".git"}


class SearchIndex:
    """FTS5 SQLite index over journal markdown files."""

    def __init__(self, db_path: Path, journal_root: Path) -> None:
        self.db_path = db_path
        self.journal_root = journal_root
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
            )
            self._conn.row_factory = sqlite3.Row
            # Allow concurrent writers to retry for up to 5s instead of
            # failing immediately with SQLITE_BUSY. Needed because
            # gunicorn runs multiple worker processes sharing this DB.
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._init_schema()
        return self._conn

    def _init_schema(self) -> None:
        """Create tables and indexes if they don't exist."""
        self._conn.executescript(SCHEMA)  # type: ignore[union-attr]

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------

    def upsert_file(self, file_path: Path) -> None:
        """Parse a markdown file and upsert it into the index."""
        rel_path = file_path.relative_to(self.journal_root).as_posix()
        post = frontmatter.load(str(file_path))
        meta = post.metadata

        # Determine doc_type from path
        if rel_path.startswith("topics/"):
            doc_type = "topic"
        elif rel_path.startswith("conversations/"):
            doc_type = "conversation"
        else:
            return  # Skip other files

        self.conn.execute(
            """
            INSERT INTO documents
                (file_path, doc_type, topic, title, description,
                 tags, created, updated, entry_count, content, indexed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(file_path) DO UPDATE SET
                doc_type=excluded.doc_type,
                topic=excluded.topic,
                title=excluded.title,
                description=excluded.description,
                tags=excluded.tags,
                created=excluded.created,
                updated=excluded.updated,
                entry_count=excluded.entry_count,
                content=excluded.content,
                indexed_at=excluded.indexed_at
            """,
            (
                rel_path,
                doc_type,
                meta.get("topic", ""),
                meta.get("title", ""),
                meta.get("description", ""),
                json.dumps(meta.get("tags", [])),
                meta.get("created", ""),
                meta.get("updated", ""),
                meta.get("entry_count", 0),
                post.content,
                int(time.time()),
            ),
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
        """Full-text search using FTS5.

        The query is passed to FTS5 MATCH via parameterized binding
        (safe from SQL injection). Malformed FTS5 syntax (unmatched
        quotes, invalid operators) raises ValueError.
        """
        if not query or not query.strip():
            return []

        sql = """
            SELECT
                d.file_path,
                d.doc_type,
                d.topic,
                d.title,
                snippet(documents_fts, 3, '<mark>', '</mark>', '...', 40)
                    AS snippet,
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

        return [
            SearchResult(
                file_path=row["file_path"],
                doc_type=row["doc_type"],
                topic=row["topic"],
                title=row["title"],
                snippet=row["snippet"],
                rank=row["rank"],
                date=row["date"],
            )
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Timeline queries
    # ------------------------------------------------------------------

    def get_entries_by_date_range(
        self,
        date_from: str,
        date_to: str,
    ) -> list[dict[str, str]]:
        """Get all documents updated within a date range.

        Used by journal_timeline to build dynamic temporal views.
        """
        rows = self.conn.execute(
            """
            SELECT file_path, doc_type, topic, title, description,
                   tags, updated
            FROM documents
            WHERE updated >= ? AND updated <= ?
            ORDER BY updated ASC
            """,
            (date_from, date_to),
        ).fetchall()

        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, int]:
        """Get index statistics."""
        total = self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        topics = self.conn.execute(
            "SELECT COUNT(*) FROM documents WHERE doc_type='topic'"
        ).fetchone()[0]
        conversations = self.conn.execute(
            "SELECT COUNT(*) FROM documents WHERE doc_type='conversation'"
        ).fetchone()[0]
        return {
            "total_documents": total,
            "topics": topics,
            "conversations": conversations,
        }

    # ------------------------------------------------------------------
    # Rebuild
    # ------------------------------------------------------------------

    def rebuild(self) -> dict[str, int | float]:
        """Full rebuild: drop all data and re-index from markdown.

        Returns stats about what was indexed.
        """
        start = time.time()

        # Clear existing data (triggers handle FTS cleanup)
        self.conn.execute("DELETE FROM documents")
        self.conn.commit()

        count = 0
        for md_file in self.journal_root.rglob("*.md"):
            # Skip files in excluded directories
            rel = md_file.relative_to(self.journal_root)
            if any(part in SKIP_DIRS for part in rel.parts):
                continue

            # Only index topics/ and conversations/
            rel_posix = rel.as_posix()
            if not (rel_posix.startswith("topics/") or rel_posix.startswith("conversations/")):
                continue

            try:
                self.upsert_file(md_file)
                count += 1
            except (OSError, ValueError, KeyError) as e:
                logger.warning("Failed to index %s: %s", md_file, e)
                continue

        duration = time.time() - start
        return {
            "documents_indexed": count,
            "duration_seconds": round(duration, 2),
        }

    def incremental_rebuild(self) -> int:
        """Incremental rebuild: only re-index files newer than indexed_at.

        Returns count of files re-indexed.
        """
        count = 0
        for md_file in self.journal_root.rglob("*.md"):
            rel = md_file.relative_to(self.journal_root)
            if any(part in SKIP_DIRS for part in rel.parts):
                continue
            rel_posix = rel.as_posix()
            if not (rel_posix.startswith("topics/") or rel_posix.startswith("conversations/")):
                continue

            rel_str = rel_posix
            file_mtime = int(md_file.stat().st_mtime)

            # Check if already indexed and up to date
            row = self.conn.execute(
                "SELECT indexed_at FROM documents WHERE file_path = ?",
                (rel_str,),
            ).fetchone()

            if row and row["indexed_at"] >= file_mtime:
                continue

            try:
                self.upsert_file(md_file)
                count += 1
            except (OSError, ValueError, KeyError) as e:
                logger.warning("Failed to index %s: %s", md_file, e)
                continue

        # Remove orphaned entries (files that no longer exist)
        all_paths = self.conn.execute("SELECT file_path FROM documents").fetchall()
        orphaned = [
            row["file_path"]
            for row in all_paths
            if not (self.journal_root / row["file_path"]).exists()
        ]
        if orphaned:
            placeholders = ",".join("?" * len(orphaned))
            self.conn.execute(
                f"DELETE FROM documents WHERE file_path IN ({placeholders})",  # noqa: S608
                orphaned,
            )
            count += len(orphaned)

        self.conn.commit()
        return count
