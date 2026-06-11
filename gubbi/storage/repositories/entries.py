"""Entry repository -- all SQL for entries table."""

from __future__ import annotations

import logging
from datetime import UTC
from datetime import date as date_cls
from datetime import datetime as datetime_cls
from typing import TYPE_CHECKING, Any, cast

import structlog

from gubbi.crypto.cipher import (
    ContentCipher,
    DecryptionError,
)
from gubbi.crypto.cipher import (
    decrypt_content_field as _decrypt_content_field,
)
from gubbi.models.journal import Entry, TopicMeta
from gubbi.storage.constants import SNIPPET_PREVIEW_LEN
from gubbi.storage.exceptions import EntryNotFoundError, TopicNotFoundError
from gubbi.storage.repositories.base import _add_param
from gubbi.storage.repositories.topics import get as get_topic
from gubbi.storage.repositories.topics import get_id as get_topic_id
from gubbi.validation import validate_date as _validate_date

if TYPE_CHECKING:
    from collections.abc import Sequence

    import asyncpg

__all__: list[str] = [
    "append",
    "delete",
    "get_by_date_range",
    "get_max_indexed_at",
    "get_stats",
    "get_text",
    "get_texts",
    "get_unindexed",
    "mark_indexed",
    "mark_indexed_batch",
    "read",
    "reset_indexed_at",
    "update",
]

logger = structlog.get_logger(__name__)
# Sync stdlib logger -- used inside the sync ``_build_entry`` closure where we
# cannot ``await`` an AsyncBoundLogger. Mirrors the embedding_service.py pattern.
_sync_logger = logging.getLogger(__name__)


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
    """Read entries for a topic, newest-first (reverse-chronological).

    Returns (TopicMeta, entries, total_matching).
    Page 0 (offset 0) is the most recent ``limit`` entries; ``offset`` skips
    the N most-recent entries before the page.
    Raises TopicNotFoundError if topic missing.
    """
    assert conn.is_in_transaction(), "entries.read: caller must wrap in conn.transaction()"  # noqa: S101
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
        try:
            content = cast(
                str, _decrypt_content_field(cipher, r, "content_encrypted", "content_nonce")
            )
            reasoning = _decrypt_content_field(cipher, r, "reasoning_encrypted", "reasoning_nonce")
        except DecryptionError:
            _sync_logger.warning(
                "entry_decryption_failed entry_id=%s topic=%s",
                r["id"],
                topic,
            )
            return Entry(
                id=r["id"],
                date=str(r["date"]),
                content="[decryption-failed]",
                reasoning=None,
                conversation_id=r["conversation_id"],
                tags=list(r["tags"] or []),
            )
        return Entry(
            id=r["id"],
            date=str(r["date"]),
            content=content,
            reasoning=reasoning,
            conversation_id=r["conversation_id"],
            tags=list(r["tags"] or []),
        )

    # All reads share one newest-first ordering so page N is the Nth slice
    # of the same reverse-chronological sequence. Page 0 is the most-recent
    # ``limit`` rows; ``offset`` skips the N most-recent before the page.
    # A prior "offset==0 fast path" also ordered DESC + LIMIT, but then
    # ``reversed()``-flipped that page to ascending; the offset>0 branch
    # ordered ASC. The two paths were anchored at OPPOSITE ends of the
    # sequence, so limit/offset paging double-returned the newest rows
    # and never reached the oldest. Unifying on a single DESC ordering
    # makes page 0/N/2N a real partition. The window-function COUNT
    # below gives ``total`` in the same round-trip, so removing the fast
    # path adds no extra query for the offset==0 case.
    #
    # ORDER BY tie-break: ``date`` and ``created_at`` can both tie -- the
    # caller may seed multiple entries on one date, and ``created_at`` is
    # caller-supplied ($8 in the INSERT) so it is not guaranteed to be
    # unique either. Without ``id DESC`` as the tail tie-break, OFFSET
    # pagination is non-deterministic for tied rows: Postgres is free to
    # reorder ties between calls, so the same row can land in two pages
    # or be skipped entirely. ``id`` is the INSERT-returned serial,
    # monotonic with insertion, and ``id DESC`` is the natural newest-
    # first tie-break (highest id = most recent insertion) AND a stable
    # total order on the unique column.
    sql_limit: int | None = limit if (limit is not None and limit > 0) else None
    sql_offset: int = offset if offset > 0 else 0
    data_params = list(params)
    data_sql = (
        f"SELECT id, date, content_encrypted, content_nonce,"  # noqa: S608 - safe: see above
        f" reasoning_encrypted, reasoning_nonce, conversation_id, tags,"
        f" COUNT(*) OVER() AS total_count"
        f" FROM entries WHERE {where} ORDER BY date DESC, created_at DESC, id DESC"
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

    SET clause is built dynamically:
      * content changed       -> writes ciphertext + nonce + search_vector
                                 + indexed_at=NULL (re-embed needed).
      * reasoning changed     -> writes reasoning ciphertext + nonce
                                 + indexed_at=NULL (re-embed needed).
      * date / tags only      -> writes the changed column only.
                                 Does NOT null indexed_at, does NOT re-encrypt
                                 -- embedding is a function of content/reasoning
                                 text, so date/tag-only edits leave the
                                 semantic index intact and skip wasted
                                 encrypt cycles + WAL amplification.
      * updated_at + topic    -> always written.
    """
    assert conn.is_in_transaction(), "entries.update: caller must wrap in conn.transaction()"  # noqa: S101
    # FOR UPDATE locks the row; mode='append' needs the existing content to
    # concatenate against, so we still SELECT the row even when not all
    # ciphertext columns are about to change.
    row = await conn.fetchrow(
        "SELECT id, content_encrypted, content_nonce,"
        " reasoning_encrypted, reasoning_nonce, topic_id, date, tags"
        " FROM entries WHERE id = $1 AND deleted_at IS NULL FOR UPDATE",
        entry_id,
    )
    if not row:
        msg = f"Entry id {entry_id} not found"
        raise EntryNotFoundError(msg)

    # Track which fields changed so we can build the SET clause + decide
    # whether to re-encrypt and whether to null indexed_at.
    content_changed = content is not None
    reasoning_changed = reasoning is not None
    date_changed = date is not None
    tags_changed = tags is not None

    # mode='append' needs the existing plaintext to concatenate; decrypt only
    # when needed.
    new_content: str | None = None
    new_reasoning: str | None = None
    if content_changed:
        if mode == "append":
            old_content = _decrypt_content_field(cipher, row, "content_encrypted", "content_nonce")
            if old_content is None:
                raise RuntimeError(
                    f"Entry {entry_id}: content decrypted to None; schema invariant violated"
                )
            new_content = f"{old_content}\n\n{content}".strip()
        elif mode == "replace":
            new_content = content
        else:
            msg = f"Invalid mode '{mode}'. Use 'replace' or 'append'."
            raise ValueError(msg)
    if reasoning_changed:
        if mode == "append":
            old_reasoning = _decrypt_content_field(
                cipher, row, "reasoning_encrypted", "reasoning_nonce"
            )
            if old_reasoning:
                new_reasoning = f"{old_reasoning}\n\n{reasoning}".strip()
            else:
                new_reasoning = reasoning
        else:
            new_reasoning = reasoning

    now = datetime_cls.now(UTC)

    # Build SET clause + parameter list dynamically.  Skip ciphertext columns
    # and indexed_at=NULL entirely when only date/tags changed so the row
    # stays in the indexed pool and we avoid a useless encrypt round-trip.
    set_parts: list[str] = ["updated_at = $1"]
    params: list[Any] = [now]

    if content_changed:
        assert new_content is not None  # noqa: S101 - narrowed by content_changed branch above
        new_content_ct, new_content_nonce = cipher.encrypt(new_content)
        set_parts.append(f"content_encrypted = ${len(params) + 1}")
        params.append(new_content_ct)
        set_parts.append(f"content_nonce = ${len(params) + 1}")
        params.append(new_content_nonce)
        set_parts.append(f"search_vector = to_tsvector('english', ${len(params) + 1})")
        params.append(new_content)
    if reasoning_changed:
        if new_reasoning is not None:
            new_reasoning_ct, new_reasoning_nonce = cipher.encrypt(new_reasoning)
        else:
            new_reasoning_ct = None
            new_reasoning_nonce = None
        set_parts.append(f"reasoning_encrypted = ${len(params) + 1}")
        params.append(new_reasoning_ct)
        set_parts.append(f"reasoning_nonce = ${len(params) + 1}")
        params.append(new_reasoning_nonce)
    if date_changed:
        assert date is not None  # noqa: S101 - narrowed by date_changed
        set_parts.append(f"date = ${len(params) + 1}")
        params.append(date_cls.fromisoformat(date))
    if tags_changed:
        set_parts.append(f"tags = ${len(params) + 1}")
        params.append(list(tags) if tags is not None else [])
    # indexed_at = NULL only when the embedded text actually changed.
    # date/tag-only updates leave indexed_at intact so the reindex worker
    # does not pick up rows whose embedding is still correct.
    if content_changed or reasoning_changed:
        set_parts.append("indexed_at = NULL")

    set_clause = ", ".join(set_parts)
    entry_param = f"${len(params) + 1}"
    params.append(entry_id)

    # CTE: update entry + update topic timestamp in one round-trip.
    await conn.execute(
        f"""
        WITH updated AS (
            UPDATE entries
            SET {set_clause}
            WHERE id = {entry_param}
            RETURNING topic_id
        )
        UPDATE topics SET updated_at = $1
        FROM updated WHERE topics.id = updated.topic_id
        """,  # noqa: S608 - set_clause built from literal column names + numbered params
        *params,
    )

    # When content or reasoning changed, search_vector was rewritten and
    # indexed_at cleared above -- but the existing entry_embeddings row
    # still reflects the OLD content. journal_search merges FTS + semantic
    # (gubbi/tools/search.py:69-105) and FTS now misses the old tokens, but
    # the stale vector would still surface the row to a query for the old
    # content. DELETE atomically (caller wraps in conn.transaction()) so no
    # reader can observe the (new content + stale embedding) state.
    #
    # The tool layer (gubbi/tools/entries.py) attempts a best-effort
    # re-embed after this UPDATE commits. If that step fails, the entry
    # stays semantic-blind (FTS-only) until a future reindex heals it --
    # preferable to a stale match that silently leaks old-content keywords
    # to a search-only-permissioned reader.
    if content_changed or reasoning_changed:
        await conn.execute(
            "DELETE FROM entry_embeddings WHERE entry_id = $1",
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
    """Stamp indexed_at = now() for a batch of entries in one query.

    Requires a BYPASSRLS connection (e.g. the ``admin_pool``). The UPDATE
    spans rows owned by potentially many users -- the reindex worker is
    a cross-tenant operation -- and a user-scoped (RLS-enforced)
    connection would silently match only the rows whose ``user_id``
    equals the current ``app.current_user_id`` GUC. Setting that GUC at
    call time is insufficient: there is no single user_id valid for a
    batch sourced from the cross-user reindex queue.
    """
    if not entry_ids:
        return
    await conn.execute(
        "UPDATE entries SET indexed_at = now() WHERE id = ANY($1)",
        entry_ids,
    )


async def reset_indexed_at(conn: asyncpg.Connection) -> None:
    """Clear indexed_at on all non-deleted entries so reindex re-embeds everything.

    Requires a BYPASSRLS connection (e.g. the ``admin_pool``). The UPDATE
    spans every tenant; a user-scoped (RLS-enforced) connection would
    restrict the rowcount to the current ``app.current_user_id`` GUC.
    Setting that GUC is insufficient: there is no single user_id valid
    for "every non-deleted row in the table".
    """
    await conn.execute("UPDATE entries SET indexed_at = NULL WHERE deleted_at IS NULL")


async def reset_indexed_at_for_ids(conn: asyncpg.Connection, entry_ids: Sequence[int]) -> None:
    """Clear indexed_at for a specific set of entries (compensating reset).

    Used by ``_run_reindex`` to roll back the claim stamp when encode or
    save fails for a subset of the claimed batch, so a subsequent reindex
    pass picks them up again.

    Requires a BYPASSRLS connection (e.g. the ``admin_pool``). The
    failed-id set is sourced from a cross-tenant reindex batch; a
    user-scoped connection would silently drop ids whose ``user_id``
    differs from the current ``app.current_user_id`` GUC and strand
    those ids in the claimed-but-never-processed state. Setting the
    GUC per-call would require iterating per-user, which defeats the
    point of batching the reset.
    """
    if not entry_ids:
        return
    await conn.execute(
        "UPDATE entries SET indexed_at = NULL WHERE id = ANY($1) AND deleted_at IS NULL",
        list(entry_ids),
    )


async def get_by_date_range(
    conn: asyncpg.Connection,
    cipher: ContentCipher | None,
    date_from: str,
    date_to: str,
    limit: int | None = None,
    ascending: bool = True,
    offset: int = 0,
    title_only: bool = False,
) -> list[dict[str, Any]]:
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

    results: list[dict[str, Any]] = []
    for r in rows:
        if title_only:
            results.append(_to_title_only_row(r, cipher))
        else:
            assert cipher is not None  # noqa: S101 - narrowed by guard above
            results.append(_to_full_row(r, cipher))
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


def _to_title_only_row(r: asyncpg.Record, cipher: ContentCipher | None) -> dict[str, Any]:
    """Shape a UNION ALL row into a title-only result dict.

    No decryption performed -- best-effort conversation title decode only.
    """
    if r["doc_type"] == "entry":
        return {
            "entry_id": r["doc_id"],
            "conversation_id": None,
            "doc_type": "entry",
            "topic": r["topic"],
            "topic_title": r["topic_title"],
            "title": _build_entry_title(r),
            "updated": r["date"],
            "tags": list(r["tags"] or []),
        }
    # conversation path
    return {
        "entry_id": None,
        "conversation_id": r["conv_id"],
        "doc_type": "conversation",
        "topic": r["topic"],
        "topic_title": r["topic_title"],
        "title": _decrypt_conv_title_or_none(cipher, r),
        "updated": r["date"],
        "tags": list(r["tags"] or []),
    }


def _to_full_row(r: asyncpg.Record, cipher: ContentCipher) -> dict[str, Any]:
    """Shape a UNION ALL row into a full-content result dict.

    Decrypts content/title/summary as needed. Raises ``RuntimeError`` on
    schema-invariant violations (None decryption results).
    """
    if r["doc_type"] == "entry":
        decrypted = _decrypt_content_field(cipher, r, "content_encrypted", "content_nonce")
        if decrypted is None:
            raise RuntimeError(
                f"Entry {r['doc_id']}: content decrypted to None; schema invariant violated"
            )
        content = decrypted
        title_text = content.split("\n", 1)[0][:80]
        return {
            "entry_id": r["doc_id"],
            "conversation_id": None,
            "doc_type": "entry",
            "topic": r["topic"],
            "title": title_text if title_text else r["topic_title"],
            "description": content[:SNIPPET_PREVIEW_LEN],
            "tags": list(r["tags"] or []),
            "updated": r["date"],
        }
    # conversation path
    conv_title = _decrypt_content_field(cipher, r, "title_encrypted", "title_nonce")
    summary = _decrypt_content_field(cipher, r, "summary_encrypted", "summary_nonce")
    if conv_title is None or summary is None:
        raise RuntimeError(
            "Conversation title/summary decrypted to None; schema invariant violated"
        )
    return {
        "entry_id": None,
        "conversation_id": r["conv_id"],
        "doc_type": "conversation",
        "topic": r["topic"],
        "title": conv_title,
        "description": summary[:SNIPPET_PREVIEW_LEN],
        "tags": list(r["tags"] or []),
        "updated": r["date"],
    }


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
) -> list[dict[str, Any]]:
    """Return a cursor-paginated batch of entries needing semantic indexing.

    The inner subquery does ``FOR UPDATE SKIP LOCKED`` against the
    ``entries`` table only -- no JOIN inside the locking SELECT, so
    the row locks are scoped exactly to rows the worker is about to
    claim. The outer SELECT then joins ``topics`` for the path/title
    just for the rows the inner SELECT actually returned.

    Cursor semantics (``last_id``): the caller advances ``last_id``
    past rows it actually claimed and processed. Rows skipped because
    another worker held their lock stay below ``last_id`` and become
    visible again on the next pass once the holder commits/aborts. The
    outer ``ORDER BY e.id`` is what makes ``batch[-1]["id"]`` a safe
    cursor for the caller.

    Callers MUST run this inside an explicit transaction; the row locks
    are released on commit/rollback.
    """
    rows = await conn.fetch(
        """
        SELECT e.id, e.user_id, e.content_encrypted, e.content_nonce,
               e.tags, e.date::text AS date, t.path AS topic, t.title
        FROM entries e
        JOIN topics t ON t.id = e.topic_id
        WHERE e.id IN (
            SELECT id FROM entries
            WHERE deleted_at IS NULL
              AND indexed_at IS NULL
              AND id > $1
            ORDER BY id
            LIMIT $2
            FOR UPDATE SKIP LOCKED
        )
        ORDER BY e.id
        """,
        last_id,
        batch_size,
    )
    result: list[dict[str, Any]] = []
    for r in rows:
        decrypted = _decrypt_content_field(cipher, r, "content_encrypted", "content_nonce")
        if decrypted is None:
            raise RuntimeError(
                f"Entry {r['id']}: content decrypted to None; schema invariant violated"
            )
        # ``user_id`` is surfaced so ``_run_reindex`` can bind
        # ``app.current_user_id`` per row before issuing the embedding
        # UPSERT; without it, ``entry_embeddings.user_id`` would resolve
        # to NULL on the admin-pool write path (the GUC is unset on a
        # BYPASSRLS connection) and HNSW + RLS would later miss those
        # rows.  Keeping the propagation in ``_run_reindex`` (not here)
        # preserves the existing transaction-scoping contract.
        result.append(
            {
                "id": r["id"],
                "user_id": r["user_id"],
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
            await logger.warning(
                "Entry could not be decrypted; included with failure marker",
                entry_id=eid,
                error_type=type(exc).__name__,
            )
            result[eid] = ("[decryption-failed]", None)  # sentinel for search.py to surface
    return result


async def get_max_indexed_at(conn: asyncpg.Connection) -> datetime_cls | None:
    """Return the most recent indexed_at timestamp across all active entries, or None."""
    return await conn.fetchval(  # type: ignore[no-any-return]
        "SELECT MAX(indexed_at) FROM entries WHERE deleted_at IS NULL AND indexed_at IS NOT NULL"
    )
