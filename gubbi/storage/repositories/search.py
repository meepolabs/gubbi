"""Search repository - FTS and topic prefix lookup SQL."""

from __future__ import annotations

from datetime import date as date_cls
from typing import Any

import asyncpg

from gubbi.models.search import SearchResult
from gubbi.storage.repositories.base import _add_param, _escape_like


async def get_topic_ids_by_prefix(
    conn: asyncpg.Connection,
    topic_prefix: str,
) -> list[int]:
    rows = await conn.fetch(
        "SELECT id FROM topics WHERE path LIKE $1 ESCAPE '!'",
        _escape_like(topic_prefix) + "%",
    )
    return [r["id"] for r in rows]


async def fts_search(
    conn: asyncpg.Connection,
    query: str,
    topic_prefix: str | None,
    date_from: str | None,
    date_to: str | None,
    limit: int,
) -> list[SearchResult]:
    params: list[Any] = [query]

    entry_where = [
        "e.deleted_at IS NULL",
        "e.search_vector @@ websearch_to_tsquery('english', $1)",
    ]
    conv_where = [
        "c.search_vector @@ websearch_to_tsquery('english', $1)",
    ]

    if topic_prefix:
        topic_ph = _add_param(params, _escape_like(topic_prefix) + "%")
        entry_where.append(f"t.path LIKE {topic_ph} ESCAPE '!'")
        conv_where.append(f"t.path LIKE {topic_ph} ESCAPE '!'")
    if date_from:
        df_ph = _add_param(params, date_cls.fromisoformat(date_from))
        entry_where.append(f"e.date >= {df_ph}")
        conv_where.append(f"c.created_at::date >= {df_ph}")
    if date_to:
        dt_ph = _add_param(params, date_cls.fromisoformat(date_to))
        entry_where.append(f"e.date <= {dt_ph}")
        conv_where.append(f"c.created_at::date <= {dt_ph}")

    limit_ph = _add_param(params, limit)
    tsq = "websearch_to_tsquery('english', $1)"
    ew = " AND ".join(entry_where)
    cw = " AND ".join(conv_where)

    sql = f"""
        SELECT entry_id, conversation_id, doc_type, topic, rank, date
        FROM (
            SELECT
                e.id            AS entry_id,
                NULL::integer   AS conversation_id,
                'entry'::text   AS doc_type,
                t.path          AS topic,
                ts_rank(e.search_vector, {tsq}) AS rank,
                e.date::text    AS date
            FROM entries e
            JOIN topics t ON t.id = e.topic_id
            WHERE {ew}

            UNION ALL

            SELECT
                NULL::integer        AS entry_id,
                c.id                 AS conversation_id,
                'conversation'::text AS doc_type,
                t.path               AS topic,
                ts_rank(c.search_vector, {tsq}) AS rank,
                c.created_at::date::text AS date
            FROM conversations c
            JOIN topics t ON t.id = c.topic_id
            WHERE {cw}
        ) combined
        ORDER BY rank DESC
        LIMIT {limit_ph}
    """  # noqa: S608
    rows = await conn.fetch(sql, *params)
    return [
        SearchResult(
            source_key=f"{r['doc_type']}:{r['entry_id'] or r['conversation_id']}",
            doc_type=r["doc_type"],
            topic=r["topic"],
            rank=-float(r["rank"]),
            date=r["date"] or "",
            entry_id=r["entry_id"],
            conversation_id=r["conversation_id"],
        )
        for r in rows
    ]
