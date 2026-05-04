"""Topic repository — all SQL for topics table."""

from __future__ import annotations

from datetime import UTC
from datetime import datetime as datetime_cls
from typing import Any

import asyncpg

from gubbi.core.validation import validate_topic
from gubbi.models.journal import TopicMeta
from gubbi.storage.exceptions import TopicNotFoundError
from gubbi.storage.repositories.base import _add_param, _escape_like


def _row_to_topic_meta(row: asyncpg.Record) -> TopicMeta:
    return TopicMeta(
        id=row["id"],
        topic=row["path"],
        title=row["title"],
        description=row["description"] or "",
        created=row["created_at"].date().isoformat(),
        updated=row["updated_at"].date().isoformat(),
        entry_count=row["entry_count"],
    )


async def get_id(conn: asyncpg.Connection, topic: str) -> int:
    """Return topic_id. Raises TopicNotFoundError if missing."""
    topic = validate_topic(topic)
    row = await conn.fetchrow("SELECT id FROM topics WHERE path = $1", topic)
    if row:
        return int(row["id"])
    msg = f"Topic '{topic}' not found — create it first with journal_create_topic"
    raise TopicNotFoundError(msg)


async def get(conn: asyncpg.Connection, topic: str) -> TopicMeta | None:
    """Get a single topic by path."""
    topic = validate_topic(topic)
    row = await conn.fetchrow(
        """
        SELECT t.id, t.path, t.title, t.description,
               t.created_at, t.updated_at,
               COUNT(e.id) AS entry_count
        FROM topics t
        LEFT JOIN entries e ON e.topic_id = t.id AND e.deleted_at IS NULL
        WHERE t.path = $1
        GROUP BY t.id
        """,
        topic,
    )
    return _row_to_topic_meta(row) if row else None


async def create(
    conn: asyncpg.Connection,
    topic: str,
    title: str,
    description: str = "",
    created_at: datetime_cls | None = None,
) -> int:
    """Create a new topic. Returns topic_id. Raises ValueError if duplicate."""
    topic = validate_topic(topic)
    now = datetime_cls.now(UTC)
    created = created_at or now
    try:
        row = await conn.fetchrow(
            """
            INSERT INTO topics (path, title, description, user_id, created_at, updated_at)
            VALUES (
                $1, $2, $3,
                (SELECT NULLIF(current_setting('app.current_user_id', true), '')::uuid),
                $4, $5
            )
            RETURNING id
            """,
            topic,
            title,
            description,
            created,
            now,
        )
        if row is None:
            raise RuntimeError("INSERT topics failed: no row returned")
        return int(row["id"])
    except asyncpg.UniqueViolationError as e:
        msg = f"Topic '{topic}' already exists"
        raise ValueError(msg) from e


async def list_all(
    conn: asyncpg.Connection,
    topic_prefix: str | None = None,
    limit: int | None = None,
    offset: int = 0,
) -> tuple[list[TopicMeta], int]:
    """List topics sorted by most recently updated. Returns (topics, total_count).

    total_count reflects the full filtered set before LIMIT — use for pagination.
    """
    params: list[Any] = []
    where = ""
    if topic_prefix:
        topic_prefix = validate_topic(topic_prefix)
        where = (
            f"WHERE t.path LIKE {_add_param(params, _escape_like(topic_prefix) + '%')} ESCAPE '!'"
        )

    pagination = ""
    if limit is not None:
        pagination = f"LIMIT {_add_param(params, limit)} OFFSET {_add_param(params, offset)}"

    sql = f"""
        SELECT t.id, t.path, t.title, t.description,
               t.created_at, t.updated_at,
               COUNT(e.id) AS entry_count,
               COUNT(*) OVER() AS total_count
        FROM topics t
        LEFT JOIN entries e ON e.topic_id = t.id AND e.deleted_at IS NULL
        {where}
        GROUP BY t.id
        ORDER BY t.updated_at DESC
        {pagination}
    """
    rows = await conn.fetch(sql, *params)
    total = int(rows[0]["total_count"]) if rows else 0
    return [_row_to_topic_meta(r) for r in rows], total


async def count(
    conn: asyncpg.Connection,
    topic_prefix: str | None = None,
) -> int:
    """Return total topic count, optionally filtered by prefix."""
    params: list[Any] = []
    where = ""
    if topic_prefix:
        topic_prefix = validate_topic(topic_prefix)
        where = (
            f"WHERE t.path LIKE {_add_param(params, _escape_like(topic_prefix) + '%')} ESCAPE '!'"
        )
    sql = f"SELECT COUNT(*) FROM topics t {where}"  # noqa: S608 — safe: topic_prefix is validated by validate_topic(); all user values go through _add_param()
    return int(await conn.fetchval(sql, *params) or 0)
