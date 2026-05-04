"""Entry repository — all SQL for entries table."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from datetime import UTC
from datetime import date as date_cls
from datetime import datetime as datetime_cls
from typing import Any, cast

import asyncpg

from journalctl.core.crypto import ContentCipher, DecryptionError, decrypt_or_raise
from journalctl.core.validation import validate_date as _validate_date
from journalctl.models.journal import Entry, TopicMeta
from journalctl.storage.constants import SNIPPET_PREVIEW_LEN
from journalctl.storage.exceptions import EntryNotFoundError, TopicNotFoundError
from journalctl.storage.repositories.base import _add_param
from journalctl.storage.repositories.topics import get as get_topic
from journalctl.storage.repositories.topics import get_id as get_topic_id

logger = logging.getLogger(__name__)


# ── module-private helpers ────────────────────────────────────────────────────


def _decrypt_content_field(
    cipher: ContentCipher,
    row: asyncpg.Record | Mapping[str, Any],
    encrypted_key: str,
    nonce_key: str,
) -> str | None:
    """Return decrypted content for a row, or ``None`` if no value is stored.

    ``row`` may be an asyncpg.Record or a dict. Three cases:

    * Both ciphertext and nonce present -- decrypt via ``decrypt_or_raise``
      which flattens any cipher failure into an opaque ``DecryptionError``.
    * Both NULL -- legitimately "no value" (reasoning is nullable, so an
      entry with no reasoning has both encrypted/nonce as NULL). Return
      ``None``; callers that require a value (e.g. ``content``) must
      verify the result is not None themselves.
    * Exactly one NULL -- corruption signal, raises ``DecryptionError``.
    """
    ct = row[encrypted_key]
    nonce = row[nonce_key]
    if ct is not None and nonce is not None:
        return decrypt_or_raise(cipher, bytes(ct), bytes(nonce))
    if ct is None and nonce is None:
        return None
    raise DecryptionError("encrypted column and nonce must both be present")


async def append(
    conn: asyncpg.Connection,
    cipher: ContentCipher,
    topic: str,
    content: str,
    reasoning: str | None = None,
    tags: Sequence[str] | None = None,
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

    content_ct, content_nonce = cipher.encrypt(content)
    if reasoning is not None:
        reasoning_ct, reasoning_nonce = cipher.encrypt(reasoning)
    else:
        reasoning_ct = None
        reasoning_nonce = None

    row = await conn.fetchrow(
        """
        WITH new_entry AS (
            INSERT INTO entries
                (topic_id, date, content_encrypted, content_nonce,
                 reasoning_encrypted, reasoning_nonce,
                 tags, user_id, created_at, updated_at, search_vector)
            VALUES (
                $1, $2, $3, $4, $5, $6, $7,
                (SELECT NULLIF(current_setting('app.current_user_id', true), '')::uuid),
                $8, $8, to_tsvector('english', $9)
            )
            RETURNING id
        ),
        _upd AS (
            UPDATE topics SET updated_at = $8
            WHERE id = $1
        )
        SELECT id FROM new_entry
        """,
        topic_id,
        d,
        content_ct,
        content_nonce,
        reasoning_ct,
        reasoning_nonce,
        tags or [],
        now,
        content,
    )
    if row is None:
        raise RuntimeError("INSERT entries failed: no row returned")
    return int(row["id"])


async def read(
    conn: asyncpg.Connection,
    cipher: ContentCipher,
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
    # validates first -- protects migration scripts and direct test calls.
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
        content = cast(str, _decrypt_content_field(cipher, r, "content_encrypted", "content_nonce"))
        reasoning = _decrypt_content_field(cipher, r, "reasoning_encrypted", "reasoning_nonce")
        return Entry(
            id=r["id"],
            date=str(r["date"]),
            content=content,
            reasoning=reasoning,
            conversation_id=r["conversation_id"],
            tags=list(r["tags"] or []),
        )

    if limit is not None and limit > 0 and offset == 0 and not date_from:
        # "Last N" case: ORDER BY DESC + LIMIT avoids COUNT + OFFSET scan.
        # Window function gives total in the same pass.
        data_params = list(params)
        limit_ph = _add_param(data_params, limit)
        rows = await conn.fetch(
            f"SELECT id, date, content_encrypted, content_nonce,"  # noqa: S608 - safe: see above
            f" reasoning_encrypted, reasoning_nonce, conversation_id, tags,"
            f" COUNT(*) OVER() AS total_count"
            f" FROM entries WHERE {where}"
            f" ORDER BY date DESC, created_at DESC LIMIT {limit_ph}",
            *data_params,
        )
        total = int(rows[0]["total_count"]) if rows else 0
        return meta, [_build_entry(r) for r in reversed(rows)], total

    # Explicit offset or date filter - window function gives total without extra query.
    sql_limit: int | None = limit if (limit is not None and limit > 0) else None
    sql_offset: int = offset if offset > 0 else 0
    data_params = list(params)
    data_sql = (
        f"SELECT id, date, content_encrypted, content_nonce,"  # noqa: S608 - safe: see above
        f" reasoning_encrypted, reasoning_nonce, conversation_id, tags,"
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
        # Offset past end - fallback count (uncommon path)
        total = int(
            await conn.fetchval(
                f"SELECT COUNT(*) FROM entries WHERE {where}",  # noqa: S608 - safe: see above
                *params,
            )
            or 0
        )
    return meta, [_build_entry(r) for r in rows], total


async def update(
    conn: asyncpg.Connection,
    cipher: ContentCipher,
    entry_id: int,
    content: str | None = None,
    reasoning: str | None = None,
    mode: str = "replace",
    date: str | None = None,
    tags: Sequence[str] | None = None,
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
            "SELECT id, content_encrypted, content_nonce,"
            " reasoning_encrypted, reasoning_nonce, topic_id, date, tags"
            " FROM entries WHERE id = $1 AND deleted_at IS NULL FOR UPDATE",
            entry_id,
        )
        if not row:
            msg = f"Entry id {entry_id} not found"
            raise EntryNotFoundError(msg)

        old_content = _decrypt_content_field(cipher, row, "content_encrypted", "content_nonce")
        old_reasoning = _decrypt_content_field(
            cipher, row, "reasoning_encrypted", "reasoning_nonce"
        )
        if old_content is None:
            raise RuntimeError(
                f"Entry {entry_id}: content decrypted to None; schema invariant violated"
            )

        new_content: str
        new_reasoning: str | None
        if content is not None:
            if mode == "replace":
                new_content = content
            elif mode == "append":
                new_content = f"{old_content}\n\n{content}".strip()
            else:
                msg = f"Invalid mode '{mode}'. Use 'replace' or 'append'."
                raise ValueError(msg)
        else:
            new_content = old_content

        if reasoning is not None:
            if mode == "append" and old_reasoning:
                new_reasoning = f"{old_reasoning}\n\n{reasoning}".strip()
            else:
                new_reasoning = reasoning
        else:
            new_reasoning = old_reasoning

        new_date: date_cls = date_cls.fromisoformat(date) if date else row["date"]
        new_tags: Sequence[str] = tags if tags is not None else list(row["tags"] or [])
        now = datetime_cls.now(UTC)

        # Encrypt new content and reasoning (if not None); else None pair for reasoning.
        new_content_ct, new_content_nonce = cipher.encrypt(new_content)
        if new_reasoning is not None:
            new_reasoning_ct, new_reasoning_nonce = cipher.encrypt(new_reasoning)
        else:
            new_reasoning_ct = None
            new_reasoning_nonce = None

        # CTE: update entry + update topic timestamp in one round-trip.
        # indexed_at = NULL signals the embedding needs regenerating.
        await conn.execute(
            """
            WITH updated AS (
                UPDATE entries
                SET date=$1, tags=$2,
                    updated_at=$3, indexed_at=NULL,
                    content_encrypted=$4, content_nonce=$5,
                    reasoning_encrypted=$6, reasoning_nonce=$7,
                    search_vector=to_tsvector('english', $8)
                WHERE id=$9
                RETURNING topic_id
            )
            UPDATE topics SET updated_at=$3
            FROM updated WHERE topics.id = updated.topic_id
            """,
            new_date,
            new_tags,
            now,
            new_content_ct,
            new_content_nonce,
            new_reasoning_ct,
            new_reasoning_nonce,
            new_content,
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
    cipher: ContentCipher | None,
    date_from: str,
    date_to: str,
    limit: int | None = None,
    ascending: bool = True,
    offset: int = 0,
    title_only: bool = False,
) -> list[dict]:
    """Get entries and conversations updated within a date range.

    Used by journal_briefing and journal_timeline.
    Returns lightweight dicts (no reasoning for brevity).
    Single UNION ALL query - one round-trip to the database.

    ascending=True  (default): oldest-first - use for timeline/date-range views.
    ascending=False: newest-first - use with limit for briefing (most-recent N).

    Args:
        conn: asyncpg connection.
        cipher: ContentCipher for decryption. May be None when title_only=True.
        date_from: Start date string (YYYY-MM-DD).
        date_to: End date string (YYYY-MM-DD).
        limit: Max rows to return (passed as SQL LIMIT, None = no limit).
        ascending: Sort direction.
        offset: Number of rows to skip for pagination.
        title_only: When True, return only IDs/titles/fields -- skips decryption
                    entirely. Use for timeline index views.  Briefing uses
                    title_only=False so it retains full content previews.
    """
    order = "ASC" if ascending else "DESC"
    params: list[Any] = [
        date_cls.fromisoformat(date_from),
        date_cls.fromisoformat(date_to),
    ]
    limit_clause = ""
    offset_clause = ""
    if limit is not None:
        limit_clause = f" LIMIT {_add_param(params, limit)}"
    if offset > 0:
        offset_clause = f" OFFSET {_add_param(params, offset)}"

    rows = await conn.fetch(
        f"""
        SELECT
            e.id              AS doc_id,
            'entry'           AS doc_type,
            e.date::text      AS date,
            e.content_encrypted,
            e.content_nonce,
            NULL::bytea       AS title_encrypted,
            NULL::bytea       AS title_nonce,
            NULL::bytea       AS summary_encrypted,
            NULL::bytea       AS summary_nonce,
            e.tags,
            t.path            AS topic,
            t.title           AS topic_title,
            NULL::int         AS conv_id
        FROM entries e
        JOIN topics t ON t.id = e.topic_id
        WHERE e.date >= $1 AND e.date <= $2
          AND e.deleted_at IS NULL
          AND e.conversation_id IS NULL

        UNION ALL

        SELECT
            c.id                      AS doc_id,
            'conversation'            AS doc_type,
            c.created_at::date::text  AS date,
            NULL::bytea               AS content_encrypted,
            NULL::bytea               AS content_nonce,
            c.title_encrypted,
            c.title_nonce,
            c.summary_encrypted,
            c.summary_nonce,
            c.tags,
            t.path                    AS topic,
            t.title                   AS topic_title,
            c.id                      AS conv_id
        FROM conversations c
        JOIN topics t ON t.id = c.topic_id
        WHERE c.created_at::date >= $1 AND c.created_at::date <= $2

        ORDER BY date {order}, doc_type ASC, doc_id {order}
        {limit_clause}{offset_clause}
        """,
        *params,
    )

    if cipher is None and not title_only:
        raise RuntimeError("get_by_date_range requires cipher when title_only=False")

    results: list[dict] = []
    for r in rows:
        if title_only:
            # Minimal metadata -- no decryption at all (or best-effort conv title decode).
            if r["doc_type"] == "entry":
                results.append(
                    {
                        "entry_id": r["doc_id"],
                        "conversation_id": None,
                        "doc_type": "entry",
                        "topic": r["topic"],
                        "topic_title": r["topic_title"],
                        "title": _build_entry_title(r),
                        "updated": r["date"],
                        "tags": list(r["tags"] or []),
                    }
                )
            else:
                results.append(
                    {
                        "entry_id": None,
                        "conversation_id": r["conv_id"],
                        "doc_type": "conversation",
                        "topic": r["topic"],
                        "topic_title": r["topic_title"],
                        "title": _decrypt_conv_title_or_none(cipher, r),
                        "updated": r["date"],
                        "tags": list(r["tags"] or []),
                    }
                )
        else:
            if cipher is None:
                raise RuntimeError("get_by_date_range: cipher required when title_only=False")
            # Full content path (used by journal_briefing).
            if r["doc_type"] == "entry":
                decrypted = _decrypt_content_field(cipher, r, "content_encrypted", "content_nonce")
                if decrypted is None:
                    raise RuntimeError(
                        f"Entry {r['doc_id']}: content decrypted to None; schema invariant violated"
                    )
                content = decrypted
                title = content.split("\n", 1)[0][:80]
            else:
                conv_title = _decrypt_content_field(cipher, r, "title_encrypted", "title_nonce")
                summary = _decrypt_content_field(cipher, r, "summary_encrypted", "summary_nonce")
                if conv_title is None or summary is None:
                    raise RuntimeError(
                        "Conversation title/summary decrypted to None; schema invariant violated"
                    )
                title = conv_title
                content = summary
            if r["doc_type"] == "entry":
                results.append(
                    {
                        "entry_id": r["doc_id"],
                        "conversation_id": None,
                        "doc_type": "entry",
                        "topic": r["topic"],
                        "title": title if title else r["topic_title"],
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
                        "title": title,
                        "description": content[:SNIPPET_PREVIEW_LEN],
                        "tags": list(r["tags"] or []),
                        "updated": r["date"],
                    }
                )
    return results


def _build_entry_title(row: asyncpg.Record) -> str:
    """Short label for a timeline entry row (no decryption available).

    Combines topic title + date so same-topic entries on different days
    are distinguishable in the navigation index.
    """
    topic_title = str(row.get("topic_title") or "")
    date_val = row.get("date")
    date_str = str(date_val) if date_val is not None else ""
    if topic_title and date_str:
        return f"{topic_title} ({date_str})"
    return topic_title or date_str


def _decrypt_conv_title_or_none(cipher: ContentCipher | None, row: asyncpg.Record) -> str:
    """Return decrypted conversation title or an empty string if unavailable."""
    if cipher is None:
        return str(row.get("topic_title") or "")
    try:
        title = _decrypt_content_field(cipher, row, "title_encrypted", "title_nonce")
        return title or ""
    except DecryptionError:
        return str(row.get("topic_title") or "")


async def get_stats(conn: asyncpg.Connection) -> dict[str, int]:
    """Return document counts for journal_briefing. Single round-trip."""
    row = await conn.fetchrow(
        """
        SELECT
            (SELECT COUNT(*) FROM entries WHERE deleted_at IS NULL) AS entry_count,
            (SELECT COUNT(*) FROM conversations)                    AS conv_count,
            (SELECT COUNT(DISTINCT t.id) FROM topics t WHERE EXISTS (
                SELECT 1 FROM entries e WHERE e.topic_id = t.id AND e.deleted_at IS NULL
            ))                                                      AS topic_count
        """
    )
    if row is None:
        return {"total_documents": 0, "conversations": 0, "topics": 0}
    entry_count = int(row["entry_count"] or 0)
    conv_count = int(row["conv_count"] or 0)
    topic_count = int(row["topic_count"] or 0)
    return {
        "total_documents": entry_count + conv_count,
        "conversations": conv_count,
        "topics": topic_count,
    }


async def get_unindexed(
    conn: asyncpg.Connection,
    cipher: ContentCipher,
    last_id: int,
    batch_size: int,
) -> list[dict]:
    """Return a cursor-paginated batch of entries needing semantic indexing."""
    rows = await conn.fetch(
        """
        SELECT e.id, e.content_encrypted, e.content_nonce,
               e.tags, e.date::text AS date, t.path AS topic, t.title
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
    result: list[dict] = []
    for r in rows:
        decrypted = _decrypt_content_field(cipher, r, "content_encrypted", "content_nonce")
        if decrypted is None:
            raise RuntimeError(
                f"Entry {r['id']}: content decrypted to None; schema invariant violated"
            )
        result.append(
            {
                "id": r["id"],
                "content": decrypted,
                "tags": list(r["tags"] or []),
                "date": r["date"],
                "topic": r["topic"],
                "title": r["title"],
            }
        )
    return result


async def get_text(
    conn: asyncpg.Connection,
    cipher: ContentCipher,
    entry_id: int,
) -> tuple[str, str | None] | None:
    """Return (content, reasoning) for an active entry, or None if not found."""
    row = await conn.fetchrow(
        "SELECT content_encrypted, content_nonce,"
        " reasoning_encrypted, reasoning_nonce"
        " FROM entries WHERE id = $1 AND deleted_at IS NULL",
        entry_id,
    )
    if not row:
        return None
    return (
        cast(
            str,
            _decrypt_content_field(cipher, row, "content_encrypted", "content_nonce"),
        ),
        _decrypt_content_field(cipher, row, "reasoning_encrypted", "reasoning_nonce"),
    )


async def get_texts(
    conn: asyncpg.Connection,
    cipher: ContentCipher,
    entry_ids: list[int],
) -> dict[int, tuple[str, str | None]]:
    """Return {entry_id: (content, reasoning)} for all active entries in entry_ids.

    Missing ids (deleted or not found) are silently omitted from the dict.
    Uses WHERE id = ANY($1) for a single round-trip.
    """
    if not entry_ids:
        return {}
    rows = await conn.fetch(
        "SELECT id,"
        " content_encrypted, content_nonce,"
        " reasoning_encrypted, reasoning_nonce"
        " FROM entries WHERE id = ANY($1) AND deleted_at IS NULL",
        entry_ids,
    )
    result: dict[int, tuple[str, str | None]] = {}
    for r in rows:
        eid = int(r["id"])
        try:
            content = cast(
                str,
                _decrypt_content_field(cipher, r, "content_encrypted", "content_nonce"),
            )
            reasoning = _decrypt_content_field(cipher, r, "reasoning_encrypted", "reasoning_nonce")
            result[eid] = (content, reasoning)
        except DecryptionError as exc:
            logger.warning(
                "Entry %d could not be decrypted (%s); included with failure marker",
                eid,
                type(exc).__name__,
            )
            result[eid] = ("[decryption-failed]", None)  # sentinel for search.py to surface (M-9.8)
    return result


async def get_max_indexed_at(conn: asyncpg.Connection) -> datetime_cls | None:
    """Return the most recent indexed_at timestamp across all active entries, or None."""
    return await conn.fetchval(  # type: ignore[no-any-return]
        "SELECT MAX(indexed_at) FROM entries WHERE deleted_at IS NULL AND indexed_at IS NOT NULL"
    )
