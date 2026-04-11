"""Search repository — FTS and topic prefix lookup SQL."""

from __future__ import annotations

from datetime import date as date_cls
from typing import Any

import asyncpg

from journalctl.models.search import SearchResult
from journalctl.storage.repositories.base import _add_param, _escape_like

# STX/ETX (0x02/0x03) are stripped from all content by sanitize_freetext(),
# so they can never appear in stored text — guaranteed collision-free markers.
# Passed to ts_headline as a query parameter (not interpolated into SQL).
# Python strips them from the output — snippet selection is the value, not the markers.
_MATCH_START = "\x02"
_MATCH_END = "\x03"
_SNIPPET_OPTS = f"MaxWords=20,MinWords=10,StartSel={_MATCH_START},StopSel={_MATCH_END}"


def _format_snippet(raw: str | None) -> str:
    return (raw or "").replace(_MATCH_START, "").replace(_MATCH_END, "")


async def get_topic_ids_by_prefix(
    conn: asyncpg.Connection,
    topic_prefix: str,
) -> list[int]:
    """Return all topic IDs whose path starts with topic_prefix."""
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
    """Full-text search using PostgreSQL tsvector + websearch_to_tsquery.

    Single UNION ALL query across entries and conversations — one round-trip,
    globally ranked, returns at most `limit` rows (never 2×limit).
    """
    params: list[Any] = [query]  # $1 = tsquery input

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

    opts_ph = _add_param(params, _SNIPPET_OPTS)
    limit_ph = _add_param(params, limit)
    tsq = "websearch_to_tsquery('english', $1)"
    ew = " AND ".join(entry_where)
    cw = " AND ".join(conv_where)

    sql = f"""
        SELECT entry_id, conversation_id, doc_type, topic, title, snippet, rank, date
        FROM (
            SELECT
                e.id            AS entry_id,
                NULL::integer   AS conversation_id,
                'entry'::text   AS doc_type,
                t.path          AS topic,
                t.title         AS title,
                ts_headline('english', e.content, {tsq}, {opts_ph}) AS snippet,
                ts_rank(e.search_vector, {tsq})                      AS rank,
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
                c.title              AS title,
                ts_headline('english', c.title || ' ' || c.summary, {tsq}, {opts_ph}) AS snippet,
                ts_rank(c.search_vector, {tsq})                                        AS rank,
                c.created_at::date::text AS date
            FROM conversations c
            JOIN topics t ON t.id = c.topic_id
            WHERE {cw}
        ) combined
        ORDER BY rank DESC
        LIMIT {limit_ph}
    """  # noqa: S608
    rows = await conn.fetch(sql, *params)
    results = []
    for r in rows:
        results.append(
            SearchResult(
                source_key=f"{r['doc_type']}:{r['entry_id'] or r['conversation_id']}",
                doc_type=r["doc_type"],
                topic=r["topic"],
                title=r["title"] or "",
                snippet=_format_snippet(r["snippet"]),
                rank=-float(r["rank"]),
                date=r["date"] or "",
                entry_id=r["entry_id"],
                conversation_id=r["conversation_id"],
            )
        )
    return results
