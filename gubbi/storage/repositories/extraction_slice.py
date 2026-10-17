"""The part of a conversation an extraction run reads, and its watermark.

``conversations.extracted_through_position`` is the highest message position
already extracted (NULL: never extracted). A run loads the messages above it
plus a short tail of already-extracted messages as context, reads the entries
earlier runs drew from the same conversation, and on success moves the
watermark forward with a compare-and-set on the value it loaded.

All functions take an RLS-scoped ``asyncpg.Connection``; another user's rows
are invisible.

Public surface
--------------
load_extraction_slice  -- tail + new messages of a conversation, decrypted, with positions
list_thread_entries    -- newest live entries drawn from a conversation, oldest first
advance_watermark      -- move the watermark from a known base to a higher target
SliceMessage           -- one decrypted message with its position
ExtractionSlice        -- what load_extraction_slice returns
ThreadEntry            -- one entry returned by list_thread_entries
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import structlog

from gubbi.constants import EXTRACTION_TAIL_MESSAGES, EXTRACTION_THREAD_ENTRIES_LIMIT
from gubbi.crypto.cipher import ContentCipher, DecryptionError, decrypt_or_raise
from gubbi.crypto.cipher import decrypt_content_field as _decrypt_content_field
from gubbi.storage.exceptions import ConversationNotFoundError

if TYPE_CHECKING:
    from datetime import date

    import asyncpg

__all__: list[str] = [
    "ExtractionSlice",
    "SliceMessage",
    "ThreadEntry",
    "advance_watermark",
    "list_thread_entries",
    "load_extraction_slice",
]

logger = structlog.get_logger(__name__)

_UNEXTRACTED = -1

_SLICE_MESSAGES_SQL = """
(SELECT position, role, content_encrypted, content_nonce, timestamp
 FROM messages
 WHERE conversation_id = $1 AND position <= $2
 ORDER BY position DESC
 LIMIT $3)
UNION ALL
(SELECT position, role, content_encrypted, content_nonce, timestamp
 FROM messages
 WHERE conversation_id = $1 AND position > $2)
ORDER BY position
"""


@dataclass(frozen=True)
class SliceMessage:
    """One decrypted message; ``timestamp`` is ISO 8601 like ``Message.timestamp``."""

    position: int
    role: str
    content: str
    timestamp: str | None


@dataclass(frozen=True)
class ExtractionSlice:
    """Messages one extraction run reads.

    ``base`` is the watermark the slice was loaded against, the value
    ``advance_watermark`` must be given. ``tail`` holds up to
    ``EXTRACTION_TAIL_MESSAGES`` messages at or below it, ``new`` every message
    above it; both are in position order.
    """

    base: int | None
    tail: tuple[SliceMessage, ...]
    new: tuple[SliceMessage, ...]

    @property
    def messages(self) -> tuple[SliceMessage, ...]:
        """Tail followed by new messages."""
        return (*self.tail, *self.new)

    @property
    def context_count(self) -> int:
        """Number of leading ``messages`` that are already-extracted context."""
        return len(self.tail)

    @property
    def target(self) -> int | None:
        """Highest position loaded above the watermark; None when nothing is above it."""
        return self.new[-1].position if self.new else None


@dataclass(frozen=True)
class ThreadEntry:
    """An entry earlier extraction runs drew from the conversation."""

    content: str
    entry_date: date


async def load_extraction_slice(
    conn: asyncpg.Connection,
    cipher: ContentCipher,
    conversation_id: int,
    tail_size: int = EXTRACTION_TAIL_MESSAGES,
) -> ExtractionSlice:
    """Load the messages above the conversation's watermark plus a tail before them.

    A NULL watermark loads every message with an empty tail. Messages are not
    role-filtered, so ``target`` covers every stored position up to the newest.
    Raises ConversationNotFoundError when the conversation is missing or
    another user's, and DecryptionError when a message cannot be decrypted.
    """
    row = await conn.fetchrow(
        "SELECT extracted_through_position FROM conversations WHERE id = $1",
        conversation_id,
    )
    if row is None:
        raise ConversationNotFoundError(f"Conversation id {conversation_id} not found")
    base: int | None = row["extracted_through_position"]
    floor = _UNEXTRACTED if base is None else base
    rows = await conn.fetch(_SLICE_MESSAGES_SQL, conversation_id, floor, tail_size)
    loaded = [_slice_message(cipher, r) for r in rows]
    return ExtractionSlice(
        base=base,
        tail=tuple(m for m in loaded if m.position <= floor),
        new=tuple(m for m in loaded if m.position > floor),
    )


def _slice_message(cipher: ContentCipher, row: Any) -> SliceMessage:
    ct = row["content_encrypted"]
    nonce = row["content_nonce"]
    if ct is None or nonce is None:
        raise DecryptionError("message content_encrypted and content_nonce must both be present")
    timestamp = row["timestamp"]
    return SliceMessage(
        position=int(row["position"]),
        role=row["role"],
        content=decrypt_or_raise(cipher, bytes(ct), bytes(nonce)),
        timestamp=timestamp.isoformat() if timestamp else None,
    )


async def list_thread_entries(
    conn: asyncpg.Connection,
    cipher: ContentCipher,
    conversation_id: int,
    limit: int = EXTRACTION_THREAD_ENTRIES_LIMIT,
) -> list[ThreadEntry]:
    """Return the newest ``limit`` live entries drawn from the conversation, oldest first.

    Newest is by event day, then by id. The conversation's own summary entry
    is left out. Rows that fail to decrypt are logged and skipped.
    """
    rows = await conn.fetch(
        """
        SELECT id, date, content_encrypted, content_nonce
        FROM entries
        WHERE source_conversation_id = $1
          AND deleted_at IS NULL
          AND conversation_id IS DISTINCT FROM $1
        ORDER BY date DESC, id DESC
        LIMIT $2
        """,
        conversation_id,
        limit,
    )
    found: list[ThreadEntry] = []
    for r in reversed(rows):
        try:
            content = _decrypt_content_field(cipher, r, "content_encrypted", "content_nonce")
        except DecryptionError as exc:
            await logger.warning(
                "thread_entry_decrypt_failed",
                entry_id=r["id"],
                error_type=type(exc).__name__,
            )
            continue
        if content is None:
            raise RuntimeError(
                f"Entry {r['id']}: content decrypted to None; schema invariant violated"
            )
        found.append(ThreadEntry(content=content, entry_date=r["date"]))
    return found


async def advance_watermark(
    conn: asyncpg.Connection,
    conversation_id: int,
    base: int | None,
    target: int,
) -> bool:
    """Move the watermark from ``base`` to ``target``.

    Returns True iff the row was updated: the stored watermark still equals
    ``base`` (NULL matching NULL) and ``target`` is above it. A concurrent
    advance, a stale ``base`` or a ``target`` at or below the watermark
    returns False and writes nothing, so the watermark never moves backward.
    """
    result = await conn.execute(
        "UPDATE conversations SET extracted_through_position = $3"
        " WHERE id = $1"
        " AND extracted_through_position IS NOT DISTINCT FROM $2::integer"
        " AND $3::integer > COALESCE($2::integer, $4::integer)",
        conversation_id,
        base,
        target,
        _UNEXTRACTED,
    )
    return str(result) == "UPDATE 1"
