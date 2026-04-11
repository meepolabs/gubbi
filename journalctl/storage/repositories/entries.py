"""Entry repository — all SQL for entries table."""

from __future__ import annotations

import logging
from datetime import UTC
from datetime import date as date_cls
from datetime import datetime as datetime_cls
from typing import Any

import asyncpg

from journalctl.core.validation import validate_date as _validate_date
from journalctl.models.journal import Entry, TopicMeta
from journalctl.storage.constants import SNIPPET_PREVIEW_LEN
from journalctl.storage.exceptions import EntryNotFoundError, TopicNotFoundError
from journalctl.storage.repositories.base import _add_param
from journalctl.storage.repositories.topics import get as get_topic
from journalctl.storage.repositories.topics import get_id as get_topic_id

logger = logging.getLogger(__name__)


async def append(
    conn: asyncpg.Connection,
    topic: str,
    content: str,
    reasoning: str | None = None,
    tags: list[str] | None = None,
    date: str | None = None,
) -> int:
    """Append a dated entry to a topic. Returns the new entry_id.

    Single query: INSERT the entry and UPDATE topics.updated_at atomically
    via a CTE. No COUNT, no separate round-trip.

    Raises TopicNotFoundError if the topic does not exist.
    """
    topic_id = await get_topic_id(conn, topic)
    d: date_cls = date_cls.fromisoformat(date) if date else date_cls.today()
    now = datetime_cls.now(UTC)

    row = await conn.fetchrow(
        """
        WITH new_entry AS (
            INSERT INTO entries
                (topic_id, date, content, reasoning, tags, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $6)
            RETURNING id
        ),
        _upd AS (
            UPDATE topics SET updated_at = $6
            WHERE id = $1
        )
        SELECT id FROM new_entry
        """,
        topic_id,
        d,
        content,
        reasoning,
        tags or [],
        now,
    )
    if row is None:
        raise RuntimeError("INSERT entries failed: no row returned")
    return int(row["id"])


async def read(
    conn: asyncpg.Connection,
    topic: str,
    limit: int | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    offset: int = 0,
) -> tuple[TopicMeta, list[Entry], int]:
    """Read entries for a topic, oldest-first.

    Returns (TopicMeta, entries, total_matching).
    Raises TopicNotFoundError if topic missing.
    """
    # Defense-in-depth: validate date formats here even though the tool layer
    # validates first — protects migration scripts and direct test calls.
    if date_from:
        _validate_date(date_from)
    if date_to:
        _validate_date(date_to)

    meta = await get_topic(conn, topic)
    if meta is None:
        msg = f"Topic '{topic}' not found"
        raise TopicNotFoundError(msg)
    if meta.id is None:
        raise RuntimeError(f"Topic '{topic}' has no database ID")

    where_parts = ["topic_id = $1", "deleted_at IS NULL"]
    params: list[Any] = [meta.id]
    if date_from:
        where_parts.append(f"date >= {_add_param(params, date_cls.fromisoformat(date_from))}")
    if date_to:
        where_parts.append(f"date <= {_add_param(params, date_cls.fromisoformat(date_to))}")
    where = " AND ".join(where_parts)

    def _build_entry(r: Any) -> Entry:
        return Entry(
            id=r["id"],
            date=str(r["date"]),
            content=r["content"],
            reasoning=r["reasoning"],
            conversation_id=r["conversation_id"],
            tags=list(r["tags"] or []),
        )

    if limit is not None and limit > 0 and offset == 0 and not date_from:
        # "Last N" case: ORDER BY DESC + LIMIT avoids COUNT + OFFSET scan.
        # Window function gives total in the same pass.
        data_params = list(params)
        limit_ph = _add_param(data_params, limit)
        rows = await conn.fetch(
            f"SELECT id, date, content, reasoning, conversation_id, tags,"  # noqa: S608 — safe: see above
            f" COUNT(*) OVER() AS total_count"
            f" FROM entries WHERE {where}"
            f" ORDER BY date DESC, created_at DESC LIMIT {limit_ph}",
            *data_params,
        )
        total = int(rows[0]["total_count"]) if rows else 0
        return meta, [_build_entry(r) for r in reversed(rows)], total

    # Explicit offset or date filter — window function gives total without extra query.
    sql_limit: int | None = limit if (limit is not None and limit > 0) else None
    sql_offset: int = offset if offset > 0 else 0
    data_params = list(params)
    data_sql = (
        f"SELECT id, date, content, reasoning, conversation_id, tags,"  # noqa: S608 — safe: see above
        f" COUNT(*) OVER() AS total_count"
        f" FROM entries WHERE {where} ORDER BY date ASC, created_at ASC"
    )
    if sql_limit is not None:
        limit_ph = _add_param(data_params, sql_limit)
        offset_ph = _add_param(data_params, sql_offset)
        data_sql += f" LIMIT {limit_ph} OFFSET {offset_ph}"
    elif sql_offset > 0:
        data_sql += f" OFFSET {_add_param(data_params, sql_offset)}"

    rows = await conn.fetch(data_sql, *data_params)
    if rows:
        total = int(rows[0]["total_count"])
    else:
        # Offset past end — fallback count (uncommon path)
        total = int(
            await conn.fetchval(
                f"SELECT COUNT(*) FROM entries WHERE {where}",  # noqa: S608 — safe: see above
                *params,
            )
            or 0
        )
    return meta, [_build_entry(r) for r in rows], total


async def update(
    conn: asyncpg.Connection,
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
    async with conn.transaction():
        row = await conn.fetchrow(
            "SELECT id, content, reasoning, topic_id, date, tags"
            " FROM entries WHERE id = $1 AND deleted_at IS NULL FOR UPDATE",
            entry_id,
        )
        if not row:
            msg = f"Entry id {entry_id} not found"
            raise EntryNotFoundError(msg)

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

        if reasoning is not None:
            if mode == "append" and row["reasoning"]:
                new_reasoning = f"{row['reasoning']}\n\n{reasoning}".strip()
            else:
                new_reasoning = reasoning
        else:
            new_reasoning = row["reasoning"]

        new_date: date_cls = date_cls.fromisoformat(date) if date else row["date"]
        new_tags: list[str] = tags if tags is not None else list(row["tags"] or [])
        now = datetime_cls.now(UTC)

        # CTE: update entry + update topic timestamp in one round-trip.
        # indexed_at = NULL signals the embedding needs regenerating.
        await conn.execute(
            """
            WITH updated AS (
                UPDATE entries
                SET content=$1, reasoning=$2, date=$3, tags=$4,
                    updated_at=$5, indexed_at=NULL
                WHERE id=$6
                RETURNING topic_id
            )
            UPDATE topics SET updated_at=$5
            FROM updated WHERE topics.id = updated.topic_id
            """,
            new_content,
            new_reasoning,
            new_date,
            new_tags,
            now,
            entry_id,
        )


async def delete(conn: asyncpg.Connection, entry_id: int) -> int:
    """Soft-delete an entry. Returns the topic_id.

    Single CTE: soft-deletes the entry, removes its embedding, and updates
    the topic timestamp atomically in one round-trip.
    Raises EntryNotFoundError if the entry is not found or already deleted.
    """
    now = datetime_cls.now(UTC)
    row = await conn.fetchrow(
        """
        WITH deleted AS (
            UPDATE entries
            SET deleted_at = $1, updated_at = $1
            WHERE id = $2 AND deleted_at IS NULL
            RETURNING id, topic_id
        ),
        _emb AS (
            DELETE FROM entry_embeddings
            WHERE entry_id = (SELECT id FROM deleted)
        ),
        _topic AS (
            UPDATE topics SET updated_at = $1
            FROM deleted WHERE topics.id = deleted.topic_id
        )
        SELECT topic_id FROM deleted
        """,
        now,
        entry_id,
    )
    if not row:
        msg = f"Entry id {entry_id} not found"
        raise EntryNotFoundError(msg)
    return int(row["topic_id"])


async def mark_indexed(conn: asyncpg.Connection, entry_id: int) -> None:
    """Stamp indexed_at = now() for a single entry after embedding store."""
    await conn.execute(
        "UPDATE entries SET indexed_at = now() WHERE id = $1",
        entry_id,
    )


async def mark_indexed_batch(conn: asyncpg.Connection, entry_ids: list[int]) -> None:
    """Stamp indexed_at = now() for a batch of entries in one query."""
    if not entry_ids:
        return
    await conn.execute(
        "UPDATE entries SET indexed_at = now() WHERE id = ANY($1)",
        entry_ids,
    )


async def reset_indexed_at(conn: asyncpg.Connection) -> None:
    """Clear indexed_at on all non-deleted entries so reindex re-embeds everything."""
    await conn.execute("UPDATE entries SET indexed_at = NULL WHERE deleted_at IS NULL")


async def get_by_date_range(
    conn: asyncpg.Connection,
    date_from: str,
    date_to: str,
    limit: int | None = None,
) -> list[dict]:
    """Get entries and conversations updated within a date range.

    Used by journal_briefing and journal_timeline.
    Returns lightweight dicts (no reasoning for brevity).
    Single UNION ALL query — one round-trip to the database.

    When limit is provided, returns the most recent N items (ORDER BY DESC)
    so callers avoid fetching the full week when only a tail is needed.
    Results are still returned in ascending order regardless.
    """
    # Use DESC + LIMIT when a cap is requested, ASC otherwise (timeline needs full set)
    order = "DESC" if limit is not None else "ASC"
    limit_clause = f"LIMIT {limit}" if limit is not None else ""
    rows = await conn.fetch(
        f"""
        SELECT
            e.id           AS doc_id,
            'entry'        AS doc_type,
            e.date::text   AS date,
            e.content,
            e.tags,
            t.path         AS topic,
            t.title        AS topic_title,
            NULL::int      AS conv_id
        FROM entries e
        JOIN topics t ON t.id = e.topic_id
        WHERE e.date >= $1 AND e.date <= $2
          AND e.deleted_at IS NULL
          AND e.conversation_id IS NULL

        UNION ALL

        SELECT
            c.id                AS doc_id,
            'conversation'      AS doc_type,
            c.created_at::date::text  AS date,
            c.summary           AS content,
            c.tags,
            t.path              AS topic,
            c.title             AS topic_title,
            c.id                AS conv_id
        FROM conversations c
        JOIN topics t ON t.id = c.topic_id
        WHERE c.created_at::date >= $1 AND c.created_at::date <= $2

        ORDER BY date {order}, doc_id {order}
        {limit_clause}
        """,
        date_cls.fromisoformat(date_from),
        date_cls.fromisoformat(date_to),
    )

    results: list[dict] = []
    for r in rows:
        content = r["content"] or ""
        if r["doc_type"] == "entry":
            first_line = content.split("\n", 1)[0][:80]
            results.append(
                {
                    "entry_id": r["doc_id"],
                    "conversation_id": None,
                    "doc_type": "entry",
                    "topic": r["topic"],
                    "title": first_line if first_line else r["topic_title"],
                    "description": content[:SNIPPET_PREVIEW_LEN],
                    "tags": list(r["tags"] or []),
                    "updated": r["date"],
                }
            )
        else:
            results.append(
                {
                    "entry_id": None,
                    "conversation_id": r["conv_id"],
                    "doc_type": "conversation",
                    "topic": r["topic"],
                    "title": r["topic_title"],
                    "description": content[:SNIPPET_PREVIEW_LEN],
                    "tags": list(r["tags"] or []),
                    "updated": r["date"],
                }
            )
    return results


async def get_stats(conn: asyncpg.Connection) -> dict[str, int]:
    """Return document counts for journal_briefing. Single round-trip."""
    row = await conn.fetchrow(
        """
        SELECT
            (SELECT COUNT(*) FROM entries WHERE deleted_at IS NULL) AS entry_count,
            (SELECT COUNT(*) FROM conversations)                    AS conv_count
        """
    )
    if row is None:
        return {"total_documents": 0, "conversations": 0}
    entry_count = int(row["entry_count"] or 0)
    conv_count = int(row["conv_count"] or 0)
    return {
        "total_documents": entry_count + conv_count,
        "conversations": conv_count,
    }


async def get_unindexed(
    conn: asyncpg.Connection,
    last_id: int,
    batch_size: int,
) -> list[Any]:
    """Return a cursor-paginated batch of entries needing semantic indexing."""
    return await conn.fetch(  # type: ignore[no-any-return]
        """
        SELECT e.id, e.content, e.tags, e.date::text AS date, t.path AS topic, t.title
        FROM entries e
        JOIN topics t ON t.id = e.topic_id
        WHERE e.deleted_at IS NULL
          AND e.indexed_at IS NULL
          AND e.id > $1
        ORDER BY e.id
        LIMIT $2
        """,
        last_id,
        batch_size,
    )


async def get_text(conn: asyncpg.Connection, entry_id: int) -> tuple[str, str | None] | None:
    """Return (content, reasoning) for an active entry, or None if not found."""
    row = await conn.fetchrow(
        "SELECT content, reasoning FROM entries WHERE id = $1 AND deleted_at IS NULL",
        entry_id,
    )
    if not row:
        return None
    return row["content"], row["reasoning"]


async def get_max_indexed_at(conn: asyncpg.Connection) -> datetime_cls | None:
    """Return the most recent indexed_at timestamp across all active entries, or None."""
    return await conn.fetchval(  # type: ignore[no-any-return]
        "SELECT MAX(indexed_at) FROM entries WHERE deleted_at IS NULL AND indexed_at IS NOT NULL"
    )
