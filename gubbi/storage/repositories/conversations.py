"""PostgreSQL conversation storage -- module-level async functions.

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
from typing import TYPE_CHECKING, Any, NamedTuple
from uuid import UUID, uuid4

import structlog

from gubbi.crypto.cipher import (
    ContentCipher,
    DecryptionError,
    decrypt_or_raise,
)
from gubbi.crypto.cipher import (
    decrypt_content_field as _decrypt_content_field,
)
from gubbi.models.conversation import ConversationMeta, Message
from gubbi.storage.exceptions import ConversationNotFoundError
from gubbi.storage.repositories.base import _add_param, _escape_like
from gubbi.storage.repositories.topics import get_id as get_topic_id
from gubbi.validation import slugify, validate_title, validate_topic

if TYPE_CHECKING:
    from collections.abc import Sequence

    import asyncpg

__all__: list[str] = [
    "DECRYPTION_FAILED_SENTINEL",
    "SaveConversationResult",
    "count_conversations",
    "delete_superseded_json_archive",
    "exists_by_platform_id",
    "get_conversation",
    "get_processed_at",
    "get_title_summary",
    "get_titles_summaries",
    "list_conversations",
    "mark_processed",
    "read_conversation",
    "read_conversation_by_id",
    "read_conversation_by_id_paginated",
    "save_conversation",
    "set_platform_metadata",
]

# ``logger`` is the canonical async-context logger (used inside async
# repository functions). ``_sync_log`` covers the sync archive-cleanup
# helper (``delete_superseded_json_archive``); ``structlog.AsyncBoundLogger``
# emits return coroutines that cannot be used from sync callers.
logger = structlog.get_logger(__name__)
_sync_log = logging.getLogger(__name__)


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


# -- JSON archive --------------------------------------------------------------


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


def _row_to_meta(
    cipher: ContentCipher, row: asyncpg.Record, *, soft_fail: bool = False
) -> ConversationMeta:
    """Map a conversation row to ``ConversationMeta``, decrypting title/summary.

    ``soft_fail=False`` (default, MCP path): a decryption failure raises
    ``RuntimeError`` / ``DecryptionError`` so a corrupt row surfaces loudly.

    ``soft_fail=True`` (web read path): a decryption failure yields the
    ``DECRYPTION_FAILED_SENTINEL`` in place of plaintext and sets
    ``decryption_failed=True`` on the returned meta, so one corrupt row does not
    fail the whole response.
    """
    if soft_fail:
        soft_title, title_failed = _decrypt_field_soft(
            cipher, row, "title_encrypted", "title_nonce"
        )
        soft_summary, summary_failed = _decrypt_field_soft(
            cipher, row, "summary_encrypted", "summary_nonce"
        )
        return ConversationMeta(
            id=row["id"],
            source=row["source"],
            title=soft_title,
            topic=row["topic"],
            tags=list(row["tags"] or []),
            created=row["created_at"].date().isoformat(),
            updated=row["updated_at"].date().isoformat(),
            summary=soft_summary,
            participants=list(row["participants"] or []),
            message_count=row["message_count"],
            decryption_failed=title_failed or summary_failed,
        )
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


# Surfaced in place of plaintext on the soft-fail (web) read path when a row
# cannot be decrypted. Matches the web decryption helper's sentinel so the
# contract is uniform across resources.
DECRYPTION_FAILED_SENTINEL: str = "[decryption failed]"


def _decrypt_field_soft(
    cipher: ContentCipher,
    row: asyncpg.Record,
    encrypted_key: str,
    nonce_key: str,
) -> tuple[str, bool]:
    """Decrypt a title/summary column pair without raising on bad data.

    Returns ``(plaintext, False)`` on success (treating a legitimately-empty
    column as ``""``) and ``(DECRYPTION_FAILED_SENTINEL, True)`` when the stored
    value cannot be decrypted.
    """
    try:
        value = _decrypt_content_field(cipher, row, encrypted_key, nonce_key)
        return (value or "", False)
    except DecryptionError:
        return (DECRYPTION_FAILED_SENTINEL, True)


def _decrypt_message_content_soft(cipher: ContentCipher, row: Any) -> tuple[str, bool]:
    """Decrypt message content without raising; returns ``(content, failed)``.

    On the soft-fail (web) read path a corrupt message yields the sentinel and
    ``failed=True`` so one bad message does not fail the whole transcript page.
    """
    try:
        return (_decrypt_message_content(cipher, row), False)
    except DecryptionError:
        return (DECRYPTION_FAILED_SENTINEL, True)


# -- Save ----------------------------------------------------------------------


class SaveConversationResult(NamedTuple):
    """Result of ``save_conversation``.

    ``superseded_json_path`` is the **previous** ``json_path`` of an
    updated conversation whose archive file has been replaced by a new
    one, else ``None``. The caller is responsible for deleting the
    superseded file via ``delete_superseded_json_archive`` AFTER the
    enclosing transaction has committed -- see ``save_conversation`` for
    the full contract.
    """

    conversation_id: int
    summary: str
    is_update: bool
    linked_entry_id: int
    superseded_json_path: str | None


async def save_conversation(
    conn: asyncpg.Connection,
    cipher: ContentCipher,
    conversations_json_dir: Path,
    topic: str,
    title: str,
    messages: list[Message],
    summary: str,
    source: str = "claude",
    tags: Sequence[str] | None = None,
    date: str | None = None,
) -> SaveConversationResult:
    """Save a conversation. Idempotent -- same topic+title overwrites.

    Transaction contract: the caller MUST wrap any multi-statement
    write method in ``conn.transaction()``. Multi-statement methods
    assert ``conn.is_in_transaction()`` at entry and raise
    AssertionError in non-prod if called outside a transaction.
    Single-statement read methods do not require a transaction.

    Returns a ``SaveConversationResult`` named tuple. The
    ``superseded_json_path`` field is the **previous** ``json_path`` of
    an updated conversation that has been replaced by a new archive
    file, else ``None``.

    The caller is responsible for deleting the superseded file via
    ``delete_superseded_json_archive`` AFTER the enclosing transaction has
    committed. If the transaction rolls back, the conversations row
    reverts to the previous ``json_path`` -- so the file MUST still exist
    on disk at that point. Performing the delete inside the transaction
    body would leave a dangling reference on rollback.

    Design: JSON archive is written to disk BEFORE the DB writes using a
    UUID filename. That way ``json_path`` is known at INSERT time and can be
    committed atomically with the rest of the row -- no separate UPDATE needed.

    Failure modes:
    - File write fails -> DB writes never run, clean state.
    - Caller's transaction rolls back -> the new UUID archive file is
      unreferenced (no row points to it); harmless on disk.
    - Caller's transaction rolls back on a re-save -> the previous
      archive file survives because ``save_conversation`` never deletes
      it; the row reverts to that path and remains internally
      consistent.
    """
    transaction_required = "conversations.save_conversation: caller must wrap in conn.transaction()"
    assert conn.is_in_transaction(), transaction_required  # noqa: S101
    topic = validate_topic(topic)
    title = validate_title(title)
    slug = slugify(title)
    conversation_date = date or date_cls.today().isoformat()
    now = datetime_cls.now(UTC)
    participants = sorted({m.role for m in messages})

    # Pre-check: validate topic exists early and get canonical_created + json_path for re-saves.
    topic_id = await get_topic_id(conn, topic)
    existing_row = await conn.fetchrow(
        "SELECT created_at, json_path FROM conversations WHERE topic_id = $1 AND slug = $2",
        topic_id,
        slug,
    )
    canonical_created = (
        existing_row["created_at"].date().isoformat() if existing_row else conversation_date
    )
    old_json_path: str | None = existing_row["json_path"] if existing_row else None

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
    # topic_id already verified and fetched in the pre-check above -- no need to re-query.
    conv_id, is_update = await _upsert_conversation_record(
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

    # Always rewrite messages on save (insert or update).
    # DELETE is a no-op for new conversations (no messages yet) and ensures
    # updated conversations reflect the current message content regardless
    # of whether the count changed.
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

    superseded_json_path = (
        old_json_path if is_update and old_json_path and old_json_path != json_path else None
    )
    return SaveConversationResult(
        conversation_id=conv_id,
        summary=summary,
        is_update=is_update,
        linked_entry_id=linked_entry_id,
        superseded_json_path=superseded_json_path,
    )


def delete_superseded_json_archive(conversations_json_dir: Path, json_path: str) -> None:
    """Delete a superseded conversation JSON archive. Best-effort.

    Callers MUST invoke this only AFTER the database transaction that
    updated the row's ``json_path`` has committed. Deleting earlier
    would leave a dangling reference if the transaction rolls back.

    Only the basename of ``json_path`` is used; any directory components
    are stripped via ``Path(json_path).name``. Symlinks are skipped (the
    function is a no-op if the resolved candidate is a symlink) so a
    deliberately-symlinked archive is never silently followed and
    unlinked. ``OSError`` from ``unlink`` is logged and swallowed --
    archive cleanup must not fail the surrounding request.
    """
    try:
        candidate = Path(conversations_json_dir / Path(json_path).name)
        if not candidate.is_symlink():
            candidate.unlink(missing_ok=True)
    except OSError:
        _sync_log.exception("Failed to delete superseded JSON archive: %s", json_path)


async def _upsert_conversation_record(
    conn: asyncpg.Connection,
    cipher: ContentCipher,
    topic_id: int,
    title: str,
    slug: str,
    source: str,
    summary: str,
    tags: Sequence[str],
    participants: Sequence[str],
    messages: list[Message],
    conversation_date: str,
    json_path: str,
) -> tuple[int, bool]:
    """Insert or update the conversations row.

    Returns (conversation_id, is_update).

    Uses ON CONFLICT DO UPDATE (race-safe upsert).
    is_update is detected via (xmax != 0) in the RETURNING clause.
    created_at is preserved on conflict (not included in DO UPDATE).
    """
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
        RETURNING id, (xmax != 0) AS was_update
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
    is_update = bool(row["was_update"]) if row["was_update"] is not None else False
    return conv_id, is_update


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


# -- List / Read ---------------------------------------------------------------


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
    sql = f"SELECT COUNT(*) FROM conversations c JOIN topics t ON t.id = c.topic_id {where}"  # noqa: S608 -- safe: topic_prefix is validated by validate_topic(); all user values go through _add_param()
    return int(await conn.fetchval(sql, *params) or 0)


async def list_conversations(
    conn: asyncpg.Connection,
    cipher: ContentCipher,
    topic_prefix: str | None = None,
    limit: int | None = None,
    offset: int = 0,
    *,
    soft_fail: bool = False,
) -> tuple[list[ConversationMeta], int]:
    """List conversations, optionally filtered by topic prefix.

    Returns (conversations, total_count).
    total_count reflects the full filtered set before LIMIT -- use for pagination.

    ``soft_fail=False`` (default, MCP path) raises on a corrupt row.
    ``soft_fail=True`` (web read path) surfaces the decryption sentinel +
    ``decryption_failed`` per meta instead, so one bad row does not fail the
    whole list.
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
        ORDER BY c.created_at DESC, c.id DESC
        {pagination}
    """
    rows = await conn.fetch(sql, *params)
    if rows:
        total = int(rows[0]["total_count"])
    else:
        # Empty page (e.g. offset past the end): COUNT(*) OVER() yields no row,
        # so fall back to the dedicated COUNT over the same prefix filter.
        total = await count_conversations(conn, topic_prefix=topic_prefix)
    return [_row_to_meta(cipher, r, soft_fail=soft_fail) for r in rows], total


async def get_conversation(
    conn: asyncpg.Connection,
    cipher: ContentCipher,
    topic: str,
    title: str,
) -> tuple[ConversationMeta, list[Message]]:
    """Read a conversation by topic + title slug.

    Returns (ConversationMeta, messages). Raises ConversationNotFoundError if not found.
    """
    transaction_required = "conversations.get_conversation: caller must wrap in conn.transaction()"
    assert conn.is_in_transaction(), transaction_required  # noqa: S101
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


# Deprecated alias kept for one release during the code-org rename.
# `get_conversation` is the canonical name; `read_conversation` was inconsistent
# with the `get_*` verb used elsewhere in this module.
read_conversation = get_conversation


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
    transaction_required = (
        "conversations.read_conversation_by_id: caller must wrap in conn.transaction()"
    )
    assert conn.is_in_transaction(), transaction_required  # noqa: S101
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
    *,
    soft_fail: bool = False,
) -> tuple[ConversationMeta, list[Message], int]:
    """Read a conversation with pagination on messages (no preview).

    Pushes LIMIT and OFFSET into the SQL query so only the requested
    page of messages is decrypted and returned.

    Args:
        conversation_id: Database primary key.
        messages_limit: Max messages to return.
        messages_offset: Messages to skip before returning.
        soft_fail: When False (default, MCP path) a corrupt title/summary or
            message raises. When True (web read path) corruption surfaces the
            decryption sentinel + per-row ``decryption_failed`` instead, so one
            bad row does not fail the whole response.

    Returns (ConversationMeta, paged_messages, total_messages).
    Raises ConversationNotFoundError if not found.
    """
    transaction_required = (
        "conversations.read_conversation_by_id_paginated: caller must wrap in conn.transaction()"
    )
    assert conn.is_in_transaction(), transaction_required  # noqa: S101
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

    meta = _row_to_meta(cipher, row, soft_fail=soft_fail)
    total_messages = int(row["message_count"]) if row["message_count"] is not None else 0

    msg_rows = await conn.fetch(
        "SELECT role, content_encrypted, content_nonce, timestamp FROM messages"
        " WHERE conversation_id = $1 ORDER BY position ASC"
        " LIMIT $2 OFFSET $3",
        conversation_id,
        messages_limit,
        messages_offset,
    )
    messages = [_message_from_row(cipher, r, soft_fail=soft_fail) for r in msg_rows]
    return meta, messages, total_messages


def _message_from_row(cipher: ContentCipher, row: Any, *, soft_fail: bool) -> Message:
    """Build a ``Message`` from a row, loud or soft-fail per ``soft_fail``."""
    timestamp = row["timestamp"].isoformat() if row["timestamp"] else None
    if soft_fail:
        content, failed = _decrypt_message_content_soft(cipher, row)
        return Message(
            role=row["role"], content=content, timestamp=timestamp, decryption_failed=failed
        )
    return Message(
        role=row["role"], content=_decrypt_message_content(cipher, row), timestamp=timestamp
    )


async def get_title_summary(
    conn: asyncpg.Connection,
    cipher: ContentCipher,
    conversation_id: int,
) -> tuple[str, str] | None:
    """Decrypt and return (title, summary) for a conversation, or None if not found."""
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


async def exists_by_platform_id(
    conn: asyncpg.Connection,
    user_id: UUID,
    platform: str,
    platform_id: str,
) -> bool:
    """Return True iff a conversation row matches (user_id, platform, platform_id)."""
    return bool(
        await conn.fetchval(
            "SELECT 1 FROM conversations"
            " WHERE user_id = $1 AND platform = $2 AND platform_id = $3",
            user_id,
            platform,
            platform_id,
        )
    )


async def set_platform_metadata(
    conn: asyncpg.Connection,
    conversation_id: int,
    platform: str,
    platform_id: str,
) -> None:
    """Set the platform + platform_id columns on a conversations row."""
    await conn.execute(
        "UPDATE conversations SET platform = $1, platform_id = $2 WHERE id = $3",
        platform,
        platform_id,
        conversation_id,
    )


async def get_processed_at(
    conn: asyncpg.Connection,
    conversation_id: int,
) -> datetime_cls | None:
    """Return the conversations.processed_at timestamp, or None if NULL."""
    return await conn.fetchval(  # type: ignore[no-any-return]
        "SELECT processed_at FROM conversations WHERE id = $1",
        conversation_id,
    )


async def mark_processed(
    conn: asyncpg.Connection,
    conversation_id: int,
) -> None:
    """Set conversations.processed_at = now() WHERE id = $conversation_id AND processed_at IS NULL.

    Idempotent under retry: if processed_at is already set this becomes a no-op (UPDATE 0).
    Safe to call from concurrent workers -- the last-write-wins race is prevented by the
    WHERE predicate; only one winner will observe rowcount == 1.
    """
    await conn.execute(
        "UPDATE conversations SET processed_at = now() WHERE id = $1 AND processed_at IS NULL",
        conversation_id,
    )


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
                await logger.warning(
                    "Skipping conversation: title/summary decrypted to None",
                    conversation_id=cid,
                )
                continue
            result[cid] = (title, summary)
        except DecryptionError as exc:
            await logger.warning(
                "Skipping conversation: decryption failed",
                conversation_id=cid,
                error_type=type(exc).__name__,
            )
    return result
