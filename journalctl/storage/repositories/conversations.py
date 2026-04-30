"""PostgreSQL conversation storage — module-level async functions.

All functions take an asyncpg.Connection as the first argument.
The ConversationMixin class is removed; DatabaseStorage inheritance is no longer needed.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC
from datetime import date as date_cls
from datetime import datetime as datetime_cls
from pathlib import Path
from typing import Any
from uuid import uuid4

import asyncpg

from journalctl.core.crypto import ContentCipher, DecryptionError, decrypt_or_raise
from journalctl.core.validation import slugify, validate_title, validate_topic
from journalctl.models.conversation import ConversationMeta, Message
from journalctl.storage.exceptions import ConversationNotFoundError
from journalctl.storage.repositories.base import _add_param, _escape_like
from journalctl.storage.repositories.topics import get_id as get_topic_id

logger = logging.getLogger(__name__)


def _parse_ts(ts: str | None) -> datetime_cls | None:
    """Convert an ISO 8601 timestamp string to a datetime object, or return None.

    Used when inserting messages into the TIMESTAMPTZ column so asyncpg
    receives a typed datetime rather than a plain string.
    """
    if ts is None:
        return None
    try:
        return datetime_cls.fromisoformat(ts)
    except ValueError:
        return None


# ── JSON archive ──────────────────────────────────────────────────────────────


def _write_conversation_json(
    conversations_json_dir: Path,
    file_id: str,
    meta: ConversationMeta,
    messages: list[Message],
) -> str:
    """Write conversation JSON archive. Returns the relative path string."""
    conversations_json_dir.mkdir(parents=True, exist_ok=True)
    out_path = conversations_json_dir / f"{file_id}.json"
    payload = {
        "meta": meta.model_dump(exclude={"id"}),
        "messages": [m.model_dump() for m in messages],
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return f"conversations_json/{file_id}.json"


def _decrypt_content_field(
    cipher: ContentCipher,
    row: Any,
    encrypted_key: str,
    nonce_key: str,
) -> str | None:
    ct = row[encrypted_key]
    nonce = row[nonce_key]
    if ct is not None and nonce is not None:
        return decrypt_or_raise(cipher, bytes(ct), bytes(nonce))
    if ct is None and nonce is None:
        return None
    raise DecryptionError("encrypted column and nonce must both be present")


def _row_to_meta(cipher: ContentCipher, row: asyncpg.Record) -> ConversationMeta:
    title = _decrypt_content_field(cipher, row, "title_encrypted", "title_nonce")
    summary = _decrypt_content_field(cipher, row, "summary_encrypted", "summary_nonce")
    if title is None or summary is None:
        raise RuntimeError(
            "Conversation title/summary decrypted to None; schema invariant violated"
        )
    return ConversationMeta(
        id=row["id"],
        source=row["source"],
        title=title,
        topic=row["topic"],
        tags=list(row["tags"] or []),
        created=row["created_at"].date().isoformat(),
        updated=row["updated_at"].date().isoformat(),
        summary=summary,
        participants=list(row["participants"] or []),
        message_count=row["message_count"],
    )


def _decrypt_message_content(
    cipher: ContentCipher,
    row: Any,
) -> str:
    """Return decrypted message content.

    Requires both ciphertext and nonce to be present. ``messages.content_encrypted``
    and ``messages.content_nonce`` are NOT NULL post-0008, so both-NULL and
    half-NULL alike indicate row corruption and raise ``DecryptionError``.
    """
    ct = row["content_encrypted"]
    nonce = row["content_nonce"]
    if ct is not None and nonce is not None:
        return decrypt_or_raise(cipher, bytes(ct), bytes(nonce))
    raise DecryptionError("message content_encrypted and content_nonce must both be present")


# ── Save ──────────────────────────────────────────────────────────────────────


async def save_conversation(
    conn: asyncpg.Connection,
    cipher: ContentCipher,
    conversations_json_dir: Path,
    topic: str,
    title: str,
    messages: list[Message],
    summary: str,
    source: str = "claude",
    tags: list[str] | None = None,
    date: str | None = None,
) -> tuple[int, str, bool, int]:
    """Save a conversation. Idempotent — same topic+title overwrites.

    The caller MUST supply ``conn`` already inside a transaction — e.g. from
    ``core.db_context.user_scoped_connection`` — because this function issues
    multiple writes (upsert conversation, delete/insert messages, upsert
    linked entry, update topic) that only stay consistent when grouped into
    one atomic commit.

    Returns (conversation_id, summary, is_update, linked_entry_id).

    Design: JSON archive is written to disk BEFORE the DB writes using a
    UUID filename. That way ``json_path`` is known at INSERT time and can be
    committed atomically with the rest of the row — no separate UPDATE needed.

    Failure modes:
    - File write fails → DB writes never run, clean state.
    - Caller's transaction rolls back → orphan UUID file on disk, harmless
      (nothing points to it).
    """
    topic = validate_topic(topic)
    title = validate_title(title)
    slug = slugify(title)
    conversation_date = date or date_cls.today().isoformat()
    now = datetime_cls.now(UTC)
    participants = sorted({m.role for m in messages})

    # Pre-check: validate topic exists early and get canonical_created for re-saves.
    topic_id = await get_topic_id(conn, topic)
    existing_row = await conn.fetchrow(
        "SELECT created_at FROM conversations WHERE topic_id = $1 AND slug = $2",
        topic_id,
        slug,
    )
    canonical_created = (
        existing_row["created_at"].date().isoformat() if existing_row else conversation_date
    )

    # --- Phase 1: Write JSON archive BEFORE transaction ---
    meta = ConversationMeta(
        source=source,
        title=title,
        topic=topic,
        tags=tags or [],
        created=canonical_created,
        updated=now.date().isoformat(),
        summary=summary,
        participants=participants,
        message_count=len(messages),
    )
    json_path = _write_conversation_json(conversations_json_dir, str(uuid4()), meta, messages)

    # --- Phase 2: All DB writes in a single transaction, json_path included ---
    # topic_id already verified and fetched in the pre-check above — no need to re-query.
    conv_id, is_update, existing_msg_count = await _upsert_conversation_record(
        conn,
        cipher,
        topic_id,
        title,
        slug,
        source,
        summary,
        tags or [],
        participants,
        messages,
        conversation_date,
        json_path,
    )

    # Skip delete+reinsert when only metadata changed (same message count).
    # This avoids deleting and re-inserting potentially thousands of rows
    # when the caller is just updating the summary or tags.
    if not is_update or existing_msg_count != len(messages):
        await conn.execute("DELETE FROM messages WHERE conversation_id = $1", conv_id)
        await _insert_messages(conn, cipher, conv_id, messages)
    linked_entry_id = await _upsert_linked_entry(
        conn, cipher, topic_id, conv_id, title, summary, conversation_date, now
    )

    await conn.execute(
        "UPDATE topics SET updated_at = $1 WHERE id = $2",
        now,
        topic_id,
    )

    return conv_id, summary, is_update, linked_entry_id


async def _upsert_conversation_record(
    conn: asyncpg.Connection,
    cipher: ContentCipher,
    topic_id: int,
    title: str,
    slug: str,
    source: str,
    summary: str,
    tags: list[str],
    participants: list[str],
    messages: list[Message],
    conversation_date: str,
    json_path: str,
) -> tuple[int, bool, int]:
    """Insert or update the conversations row.

    Returns (conversation_id, is_update, existing_msg_count).

    Uses ON CONFLICT DO UPDATE (race-safe upsert).
    is_update is detected via a pre-check SELECT — avoids relying on the
    undocumented xmax implementation detail.
    created_at is preserved on conflict (not included in DO UPDATE).
    """
    existing = await conn.fetchrow(
        "SELECT id, message_count FROM conversations WHERE topic_id = $1 AND slug = $2",
        topic_id,
        slug,
    )
    is_update = existing is not None
    existing_msg_count = int(existing["message_count"]) if existing else 0

    now = datetime_cls.now(UTC)
    title_ct, title_nonce = cipher.encrypt(title)
    summary_ct, summary_nonce = cipher.encrypt(summary)
    vector_text = f"{title} {summary}".strip()
    row = await conn.fetchrow(
        """
        INSERT INTO conversations
            (topic_id, title_encrypted, title_nonce, slug, source,
             summary_encrypted, summary_nonce, tags, participants,
             message_count, user_id, created_at, updated_at, json_path, search_vector)
        VALUES (
            $1, $2, $3, $4, $5,
            $6, $7, $8, $9, $10,
            (SELECT NULLIF(current_setting('app.current_user_id', true), '')::uuid),
            $11, $12, $13, to_tsvector('english', $14)
        )
        ON CONFLICT (topic_id, slug) DO UPDATE
            SET source        = excluded.source,
                title_encrypted = excluded.title_encrypted,
                title_nonce   = excluded.title_nonce,
                summary_encrypted = excluded.summary_encrypted,
                summary_nonce = excluded.summary_nonce,
                tags          = excluded.tags,
                participants  = excluded.participants,
                message_count = excluded.message_count,
                updated_at    = excluded.updated_at,
                json_path     = excluded.json_path,
                search_vector = excluded.search_vector
        RETURNING id
        """,
        topic_id,
        title_ct,
        title_nonce,
        slug,
        source,
        summary_ct,
        summary_nonce,
        tags,
        participants,
        len(messages),
        datetime_cls.fromisoformat(conversation_date).replace(tzinfo=UTC),
        now,
        json_path,
        vector_text,
    )
    if row is None:
        raise RuntimeError("INSERT/UPDATE conversations failed: no row returned")

    conv_id = int(row["id"])
    return conv_id, is_update, existing_msg_count


async def _insert_messages(
    conn: asyncpg.Connection,
    cipher: ContentCipher,
    conv_id: int,
    messages: list[Message],
) -> None:
    """Insert all messages for a conversation."""
    rows: list[tuple[int, str, datetime_cls | None, int, bytes, bytes, str]] = []
    for i, m in enumerate(messages):
        content_ct, content_nonce = cipher.encrypt(m.content)
        rows.append(
            (
                conv_id,
                m.role,
                _parse_ts(m.timestamp),
                i,
                content_ct,
                content_nonce,
                m.content,
            )
        )
    await conn.executemany(
        """
        INSERT INTO messages
            (conversation_id, role, timestamp, position,
             content_encrypted, content_nonce, search_vector, user_id)
        VALUES (
            $1, $2, $3, $4, $5, $6, to_tsvector('english', $7),
            (SELECT NULLIF(current_setting('app.current_user_id', true), '')::uuid)
        )
        """,
        rows,
    )


async def _upsert_linked_entry(
    conn: asyncpg.Connection,
    cipher: ContentCipher,
    topic_id: int,
    conv_id: int,
    title: str,
    summary: str,
    entry_date: str,
    now: datetime_cls,
) -> int:
    """Upsert a linked entry so the conversation appears in journal_read_topic + timeline.

    Returns the entry_id so callers can embed it after the transaction commits.
    """
    content = f"Conversation saved: {title}\n\n{summary}"
    content_ct, content_nonce = cipher.encrypt(content)
    entry_date_val = date_cls.fromisoformat(entry_date)
    existing = await conn.fetchrow(
        "SELECT id FROM entries WHERE conversation_id = $1",
        conv_id,
    )

    if existing:
        # reasoning_encrypted/reasoning_nonce reset to NULL: linked entries
        # never have reasoning, and clearing them explicitly prevents stale
        # ciphertext from persisting if that invariant is ever relaxed.
        await conn.execute(
            "UPDATE entries SET content_encrypted = $1, content_nonce = $2,"
            " search_vector = to_tsvector('english', $3), reasoning_encrypted = NULL,"
            " reasoning_nonce = NULL, updated_at = $4, date = $5, indexed_at = NULL"
            " WHERE id = $6",
            content_ct,
            content_nonce,
            content,
            now,
            entry_date_val,
            int(existing["id"]),
        )
        return int(existing["id"])
    row = await conn.fetchrow(
        """
        INSERT INTO entries
            (topic_id, date, content_encrypted, content_nonce, search_vector,
             conversation_id, tags, user_id, created_at, updated_at)
        VALUES (
            $1, $2, $3, $4, to_tsvector('english', $5), $6, $7,
            (SELECT NULLIF(current_setting('app.current_user_id', true), '')::uuid),
            $8, $8
        )
        RETURNING id
        """,
        topic_id,
        entry_date_val,
        content_ct,
        content_nonce,
        content,
        conv_id,
        ["conversation"],
        now,
    )
    if row is None:
        raise RuntimeError("INSERT linked entry failed: no row returned")
    return int(row["id"])


# ── List / Read ───────────────────────────────────────────────────────────────


async def count_conversations(
    conn: asyncpg.Connection,
    topic_prefix: str | None = None,
) -> int:
    """Return total conversation count, optionally filtered by topic prefix."""
    params: list[Any] = []
    where = ""
    if topic_prefix:
        topic_prefix = validate_topic(topic_prefix)
        where = (
            f"WHERE t.path LIKE {_add_param(params, _escape_like(topic_prefix) + '%')} ESCAPE '!'"
        )
    sql = f"SELECT COUNT(*) FROM conversations c JOIN topics t ON t.id = c.topic_id {where}"  # noqa: S608 — safe: topic_prefix is validated by validate_topic(); all user values go through _add_param()
    return int(await conn.fetchval(sql, *params) or 0)


async def list_conversations(
    conn: asyncpg.Connection,
    cipher: ContentCipher,
    topic_prefix: str | None = None,
    limit: int | None = None,
    offset: int = 0,
) -> tuple[list[ConversationMeta], int]:
    """List conversations, optionally filtered by topic prefix.

    Returns (conversations, total_count).
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
        SELECT c.id, c.title_encrypted, c.title_nonce, c.slug, c.source,
               c.summary_encrypted, c.summary_nonce, c.tags,
               c.participants, c.message_count,
               c.created_at, c.updated_at, t.path AS topic,
               COUNT(*) OVER() AS total_count
        FROM conversations c
        JOIN topics t ON t.id = c.topic_id
        {where}
        ORDER BY c.created_at DESC
        {pagination}
    """
    rows = await conn.fetch(sql, *params)
    total = int(rows[0]["total_count"]) if rows else 0
    return [_row_to_meta(cipher, r) for r in rows], total


async def read_conversation(
    conn: asyncpg.Connection,
    cipher: ContentCipher,
    topic: str,
    title: str,
) -> tuple[ConversationMeta, list[Message]]:
    """Read a conversation by topic + title slug.

    Returns (ConversationMeta, messages). Raises ConversationNotFoundError if not found.
    """
    topic = validate_topic(topic)
    slug = slugify(title)

    row = await conn.fetchrow(
        """
        SELECT c.id, c.title_encrypted, c.title_nonce, c.slug, c.source,
               c.summary_encrypted, c.summary_nonce, c.tags,
               c.participants, c.message_count,
               c.created_at, c.updated_at, t.path AS topic
        FROM conversations c
        JOIN topics t ON t.id = c.topic_id
        WHERE t.path = $1 AND c.slug = $2
        """,
        topic,
        slug,
    )
    if not row:
        msg = f"Conversation '{title}' not found under '{topic}'"
        raise ConversationNotFoundError(msg)

    meta = _row_to_meta(cipher, row)
    msg_rows = await conn.fetch(
        "SELECT role, content_encrypted, content_nonce, timestamp FROM messages"
        " WHERE conversation_id = $1 ORDER BY position ASC",
        int(row["id"]),
    )
    return meta, [
        Message(
            role=r["role"],
            content=_decrypt_message_content(cipher, r),
            timestamp=r["timestamp"].isoformat() if r["timestamp"] else None,
        )
        for r in msg_rows
    ]


async def read_conversation_by_id(
    conn: asyncpg.Connection,
    cipher: ContentCipher,
    conversation_id: int,
    preview: bool = False,
) -> tuple[ConversationMeta, list[Message], int]:
    """Read a conversation by its stable integer ID.

    Args:
        conversation_id: Database primary key.
        preview: If True, return only first 3 and last 3 messages.

    Returns (ConversationMeta, messages, total_messages).
    Raises ConversationNotFoundError if not found.
    """
    row = await conn.fetchrow(
        """
        SELECT c.id, c.title_encrypted, c.title_nonce, c.slug, c.source,
               c.summary_encrypted, c.summary_nonce, c.tags,
               c.participants, c.message_count,
               c.created_at, c.updated_at, t.path AS topic
        FROM conversations c
        JOIN topics t ON t.id = c.topic_id
        WHERE c.id = $1
        """,
        conversation_id,
    )
    if not row:
        msg = f"Conversation id {conversation_id} not found"
        raise ConversationNotFoundError(msg)

    meta = _row_to_meta(cipher, row)
    total_messages = int(row["message_count"]) if row["message_count"] is not None else 0

    msg_rows = await conn.fetch(
        "SELECT role, content_encrypted, content_nonce, timestamp FROM messages"
        " WHERE conversation_id = $1 ORDER BY position ASC",
        conversation_id,
    )
    messages = [
        Message(
            role=r["role"],
            content=_decrypt_message_content(cipher, r),
            timestamp=r["timestamp"].isoformat() if r["timestamp"] else None,
        )
        for r in msg_rows
    ]
    if preview and len(messages) > 6:
        messages = messages[:3] + messages[-3:]
    return meta, messages, total_messages


async def read_conversation_by_id_paginated(
    conn: asyncpg.Connection,
    cipher: ContentCipher,
    conversation_id: int,
    messages_limit: int,
    messages_offset: int = 0,
) -> tuple[ConversationMeta, list[Message], int]:
    """Read a conversation with pagination on messages (no preview).

    Pushes LIMIT and OFFSET into the SQL query so only the requested
    page of messages is decrypted and returned.

    Args:
        conversation_id: Database primary key.
        messages_limit: Max messages to return.
        messages_offset: Messages to skip before returning.

    Returns (ConversationMeta, paged_messages, total_messages).
    Raises ConversationNotFoundError if not found.
    """
    row = await conn.fetchrow(
        """
        SELECT c.id, c.title_encrypted, c.title_nonce, c.slug, c.source,
               c.summary_encrypted, c.summary_nonce, c.tags,
               c.participants, c.message_count,
               c.created_at, c.updated_at, t.path AS topic
        FROM conversations c
        JOIN topics t ON t.id = c.topic_id
        WHERE c.id = $1
        """,
        conversation_id,
    )
    if not row:
        msg = f"Conversation id {conversation_id} not found"
        raise ConversationNotFoundError(msg)

    meta = _row_to_meta(cipher, row)
    total_messages = int(row["message_count"]) if row["message_count"] is not None else 0

    msg_rows = await conn.fetch(  # noqa: S608
        "SELECT role, content_encrypted, content_nonce, timestamp FROM messages"
        " WHERE conversation_id = $1 ORDER BY position ASC"
        " LIMIT $2 OFFSET $3",
        conversation_id,
        messages_limit,
        messages_offset,
    )
    messages = [
        Message(
            role=r["role"],
            content=_decrypt_message_content(cipher, r),
            timestamp=r["timestamp"].isoformat() if r["timestamp"] else None,
        )
        for r in msg_rows
    ]
    return meta, messages, total_messages


async def get_title_summary(
    conn: asyncpg.Connection,
    cipher: ContentCipher,
    conversation_id: int,
) -> tuple[str, str] | None:
    row = await conn.fetchrow(
        "SELECT title_encrypted, title_nonce, summary_encrypted, summary_nonce "
        "FROM conversations WHERE id = $1",
        conversation_id,
    )
    if not row:
        return None
    title = _decrypt_content_field(cipher, row, "title_encrypted", "title_nonce")
    summary = _decrypt_content_field(cipher, row, "summary_encrypted", "summary_nonce")
    if title is None or summary is None:
        raise RuntimeError(
            "Conversation title/summary decrypted to None; schema invariant violated"
        )
    return title, summary


async def get_titles_summaries(
    conn: asyncpg.Connection,
    cipher: ContentCipher,
    conversation_ids: list[int],
) -> dict[int, tuple[str, str]]:
    """Return {conv_id: (title, summary)} for all conversations in conversation_ids.

    Missing ids are silently omitted.
    Uses WHERE id = ANY($1) for a single round-trip.
    """
    if not conversation_ids:
        return {}
    rows = await conn.fetch(
        "SELECT id,"
        " title_encrypted, title_nonce, summary_encrypted, summary_nonce "
        "FROM conversations WHERE id = ANY($1)",
        conversation_ids,
    )
    result: dict[int, tuple[str, str]] = {}
    for r in rows:
        cid = int(r["id"])
        try:
            title = _decrypt_content_field(cipher, r, "title_encrypted", "title_nonce")
            summary = _decrypt_content_field(cipher, r, "summary_encrypted", "summary_nonce")
            if title is None or summary is None:
                logger.warning(
                    "Skipping conversation %d: title/summary decrypted to None",
                    cid,
                )
                continue
            result[cid] = (title, summary)
        except DecryptionError as exc:
            logger.warning(
                "Skipping conversation %d: decryption failed (%s)",
                cid,
                type(exc).__name__,
                exc_info=True,
            )
    return result
