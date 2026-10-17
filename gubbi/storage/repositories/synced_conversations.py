"""Merge a conversation synced from a source platform into stored history.

A synced conversation is identified by ``(user_id, platform, platform_id)``
and each of its messages by ``source_message_id``. A merge never deletes a
stored message: re-sending a thread appends the messages not yet stored and
replaces the text of stored ones in place, so an abandoned branch stays.
Messages without a source id only extend a stored thread whose text they
repeat exactly; anything else writes nothing and is reported as a conflict.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import asyncpg

from gubbi.crypto.cipher import decrypt_or_raise
from gubbi.storage.repositories.conversations import upsert_linked_entry
from gubbi.storage.repositories.topics import get_id as get_topic_id
from gubbi.validation import slugify, truncate_synced_message, validate_title

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gubbi.crypto.cipher import ContentCipher

__all__: list[str] = [
    "MergeResult",
    "SyncedConversation",
    "SyncedMessage",
    "merge_synced_conversation",
]


@dataclass(frozen=True, slots=True)
class SyncedMessage:
    """One message as the source platform sent it."""

    role: str
    content: str
    timestamp: datetime | None = None
    message_id: str | None = None


@dataclass(frozen=True, slots=True)
class SyncedConversation:
    """A conversation as the source platform sent it, in payload order."""

    platform: str
    platform_id: str
    title: str
    topic: str
    created_at: datetime
    messages: tuple[SyncedMessage, ...]


@dataclass(frozen=True, slots=True)
class MergeResult:
    """What one merge wrote.

    ``truncated`` counts written messages (appended or replaced) whose text was
    cut to the length limit. ``duplicate_ids`` counts payload messages dropped
    because an earlier message in the same payload carried the same id. When
    ``legacy_identity_conflict`` is set, nothing was written, and
    ``conversation_id`` is None if the payload mixed id and id-less messages.
    """

    conversation_id: int | None
    created: bool
    appended: int
    replaced: int
    truncated: int
    duplicate_ids: int
    legacy_identity_conflict: bool

    @property
    def has_new_messages(self) -> bool:
        """True when messages were appended and the thread needs extraction."""
        return self.appended > 0


@dataclass(frozen=True, slots=True)
class _Incoming:
    role: str
    text: str
    was_cut: bool
    timestamp: datetime | None
    message_id: str | None


@dataclass(frozen=True, slots=True)
class _Stored:
    row_id: int
    position: int
    source_message_id: str | None
    content_encrypted: bytes
    content_nonce: bytes


@dataclass(frozen=True, slots=True)
class _Plan:
    appends: tuple[_Incoming, ...] = ()
    replacements: tuple[tuple[int, _Incoming], ...] = ()
    duplicate_ids: int = 0
    conflict: bool = False


async def merge_synced_conversation(
    conn: asyncpg.Connection,
    cipher: ContentCipher,
    conversation: SyncedConversation,
) -> MergeResult:
    """Insert or extend one synced conversation inside the caller's transaction.

    Each attempt runs in a savepoint, so a unique violation from a racing
    writer rolls back only this merge; it is retried once from a fresh read,
    and a second violation propagates.
    """
    if not conn.is_in_transaction():
        raise RuntimeError("merge_synced_conversation: caller must wrap in conn.transaction()")
    if not conversation.messages:
        raise ValueError("a synced conversation needs at least one message")
    incoming = tuple(_prepare(m) for m in conversation.messages)
    if len({m.message_id is None for m in incoming}) > 1:
        return MergeResult(None, False, 0, 0, 0, 0, legacy_identity_conflict=True)
    try:
        async with conn.transaction():
            return await _merge_once(conn, cipher, conversation, incoming)
    except asyncpg.UniqueViolationError:
        async with conn.transaction():
            return await _merge_once(conn, cipher, conversation, incoming)


def _prepare(message: SyncedMessage) -> _Incoming:
    text, was_cut = truncate_synced_message(message.content)
    return _Incoming(message.role, text, was_cut, message.timestamp, message.message_id)


async def _merge_once(
    conn: asyncpg.Connection,
    cipher: ContentCipher,
    conversation: SyncedConversation,
    incoming: tuple[_Incoming, ...],
) -> MergeResult:
    now = datetime.now(UTC)
    title = validate_title(conversation.title)
    topic_id = await get_topic_id(conn, conversation.topic)
    created = await _insert_conversation(conn, cipher, topic_id, title, conversation, now)
    conv_id, stored_topic_id, participants = await _lock_conversation(conn, conversation)
    stored = await _load_stored(conn, conv_id)

    if created:
        plan = _plan_new(incoming)
    elif incoming[0].message_id is not None:
        plan = _plan_by_id(cipher, incoming, stored)
    else:
        plan = _plan_by_prefix(cipher, incoming, stored)

    if plan.conflict:
        return MergeResult(conv_id, created, 0, 0, 0, 0, legacy_identity_conflict=True)

    await _replace_texts(conn, cipher, plan.replacements)
    next_position = max((s.position for s in stored), default=-1) + 1
    await _insert_messages(conn, cipher, conv_id, next_position, plan.appends)
    if plan.appends:
        roles = sorted({*participants, *(m.role for m in plan.appends)})
        await _mark_appended(conn, conv_id, stored_topic_id, len(stored) + len(plan.appends), roles)
    if created:
        entry_date = conversation.created_at.astimezone(UTC).date().isoformat()
        await upsert_linked_entry(conn, cipher, topic_id, conv_id, title, "", entry_date, now)

    written = (*plan.appends, *(m for _, m in plan.replacements))
    return MergeResult(
        conversation_id=conv_id,
        created=created,
        appended=len(plan.appends),
        replaced=len(plan.replacements),
        truncated=sum(1 for m in written if m.was_cut),
        duplicate_ids=plan.duplicate_ids,
        legacy_identity_conflict=False,
    )


def _drop_duplicate_ids(incoming: Sequence[_Incoming]) -> tuple[list[_Incoming], int]:
    seen: set[str] = set()
    kept: list[_Incoming] = []
    for message in incoming:
        if message.message_id is not None:
            if message.message_id in seen:
                continue
            seen.add(message.message_id)
        kept.append(message)
    return kept, len(incoming) - len(kept)


def _plan_new(incoming: Sequence[_Incoming]) -> _Plan:
    kept, duplicates = _drop_duplicate_ids(incoming)
    return _Plan(appends=tuple(kept), duplicate_ids=duplicates)


def _plan_by_id(
    cipher: ContentCipher, incoming: Sequence[_Incoming], stored: Sequence[_Stored]
) -> _Plan:
    # Stored rows without a source id cannot be matched to an id, so every id
    # would read as unseen and the whole thread would be appended a second time.
    if any(s.source_message_id is None for s in stored):
        return _Plan(conflict=True)
    by_id = {s.source_message_id: s for s in stored}
    kept, duplicates = _drop_duplicate_ids(incoming)
    appends: list[_Incoming] = []
    replacements: list[tuple[int, _Incoming]] = []
    for message in kept:
        match = by_id.get(message.message_id)
        if match is None:
            appends.append(message)
        elif _decrypt(cipher, match) != message.text:
            replacements.append((match.row_id, message))
    return _Plan(tuple(appends), tuple(replacements), duplicates)


def _plan_by_prefix(
    cipher: ContentCipher, incoming: Sequence[_Incoming], stored: Sequence[_Stored]
) -> _Plan:
    # A thread that already holds source ids is matched by id only; id-less
    # rows appended to it would make every later id payload unmatchable.
    if any(s.source_message_id is not None for s in stored):
        return _Plan(conflict=True)
    overlap = min(len(incoming), len(stored))
    for message, row in zip(incoming[:overlap], stored[:overlap], strict=True):
        if _decrypt(cipher, row) != message.text:
            return _Plan(conflict=True)
    return _Plan(appends=tuple(incoming[overlap:]))


def _decrypt(cipher: ContentCipher, row: _Stored) -> str:
    return decrypt_or_raise(cipher, row.content_encrypted, row.content_nonce)


async def _insert_conversation(
    conn: asyncpg.Connection,
    cipher: ContentCipher,
    topic_id: int,
    title: str,
    conversation: SyncedConversation,
    now: datetime,
) -> bool:
    title_ct, title_nonce = cipher.encrypt(title)
    summary_ct, summary_nonce = cipher.encrypt("")
    row_id = await conn.fetchval(
        """
        INSERT INTO conversations
            (topic_id, title_encrypted, title_nonce, slug, source,
             summary_encrypted, summary_nonce, participants, message_count,
             user_id, created_at, updated_at, search_vector, platform, platform_id)
        VALUES (
            $1, $2, $3, $4, $5, $6, $7, '{}', 0,
            (SELECT NULLIF(current_setting('app.current_user_id', true), '')::uuid),
            $8, $9, to_tsvector('english', $10), $11, $12
        )
        ON CONFLICT (user_id, platform, platform_id) WHERE platform_id IS NOT NULL
        DO NOTHING
        RETURNING id
        """,
        topic_id,
        title_ct,
        title_nonce,
        slugify(title),
        conversation.platform,
        summary_ct,
        summary_nonce,
        conversation.created_at.astimezone(UTC),
        now,
        title,
        conversation.platform,
        conversation.platform_id,
    )
    return row_id is not None


async def _lock_conversation(
    conn: asyncpg.Connection, conversation: SyncedConversation
) -> tuple[int, int, list[str]]:
    row = await conn.fetchrow(
        """
        SELECT id, topic_id, participants FROM conversations
        WHERE user_id = (SELECT NULLIF(current_setting('app.current_user_id', true), '')::uuid)
          AND platform = $1 AND platform_id = $2
        FOR UPDATE
        """,
        conversation.platform,
        conversation.platform_id,
    )
    if row is None:
        raise RuntimeError("synced conversation row missing after insert")
    return int(row["id"]), int(row["topic_id"]), list(row["participants"] or [])


async def _load_stored(conn: asyncpg.Connection, conv_id: int) -> list[_Stored]:
    rows = await conn.fetch(
        "SELECT id, position, source_message_id, content_encrypted, content_nonce"
        " FROM messages WHERE conversation_id = $1 ORDER BY position, id",
        conv_id,
    )
    return [
        _Stored(
            row_id=int(r["id"]),
            position=int(r["position"]),
            source_message_id=r["source_message_id"],
            content_encrypted=bytes(r["content_encrypted"]),
            content_nonce=bytes(r["content_nonce"]),
        )
        for r in rows
    ]


async def _replace_texts(
    conn: asyncpg.Connection,
    cipher: ContentCipher,
    replacements: Sequence[tuple[int, _Incoming]],
) -> None:
    if not replacements:
        return
    sealed = [cipher.encrypt(message.text) for _, message in replacements]
    await conn.execute(
        """
        UPDATE messages AS m
        SET content_encrypted = r.ct, content_nonce = r.nonce,
            search_vector = to_tsvector('english', r.body)
        FROM unnest($1::integer[], $2::bytea[], $3::bytea[], $4::text[])
            AS r(row_id, ct, nonce, body)
        WHERE m.id = r.row_id
        """,
        [row_id for row_id, _ in replacements],
        [ct for ct, _ in sealed],
        [nonce for _, nonce in sealed],
        [message.text for _, message in replacements],
    )


async def _insert_messages(
    conn: asyncpg.Connection,
    cipher: ContentCipher,
    conv_id: int,
    first_position: int,
    messages: Sequence[_Incoming],
) -> None:
    if not messages:
        return
    rows = []
    for offset, message in enumerate(messages):
        content_ct, content_nonce = cipher.encrypt(message.text)
        rows.append(
            (
                conv_id,
                message.role,
                message.timestamp,
                first_position + offset,
                content_ct,
                content_nonce,
                message.text,
                message.message_id,
            )
        )
    await conn.executemany(
        """
        INSERT INTO messages
            (conversation_id, role, timestamp, position, content_encrypted,
             content_nonce, search_vector, source_message_id, user_id)
        VALUES (
            $1, $2, $3, $4, $5, $6, to_tsvector('english', $7), $8,
            (SELECT NULLIF(current_setting('app.current_user_id', true), '')::uuid)
        )
        """,
        rows,
    )


async def _mark_appended(
    conn: asyncpg.Connection,
    conv_id: int,
    topic_id: int,
    message_count: int,
    participants: list[str],
) -> None:
    # Read the clock after the row lock so a writer that waited never goes back.
    await conn.execute(
        "UPDATE conversations SET message_count = $1, participants = $2,"
        " updated_at = GREATEST(updated_at, clock_timestamp()), processed_at = NULL"
        " WHERE id = $3",
        message_count,
        participants,
        conv_id,
    )
    await conn.execute(
        "UPDATE topics SET updated_at = GREATEST(updated_at, clock_timestamp()) WHERE id = $1",
        topic_id,
    )
