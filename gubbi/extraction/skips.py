"""Rule-based extraction skips and the conversation content hash.

Pure functions only: no I/O, no logging, no provider calls. Both operate on
the role-filtered transcript (user/assistant turns with non-blank content), so
system prompts and tool output never trigger a re-extraction or count as text.
"""

import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol

from gubbi.constants import (
    AUTOMATED_EXTRACTION_SOURCES,
    EXTRACTION_MIN_USER_CHARS,
    EXTRACTION_VERSION,
)
from gubbi.models.conversation import keep_conversation_roles

USER_ROLE = "user"


class SkippableMessage(Protocol):
    """A transcript message; ``timestamp`` may be a datetime or an ISO string."""

    @property
    def role(self) -> str:
        """Speaker role, e.g. ``user`` or ``assistant``."""
        ...

    @property
    def content(self) -> str:
        """Message text."""
        ...

    @property
    def timestamp(self) -> datetime | str | None:
        """When the message was sent, if known."""
        ...


class SkipReason(StrEnum):
    """Why a conversation is not sent to the extraction model."""

    NO_TEXT = "no_text"
    TRIVIAL_USER_TEXT = "trivial_user_text"
    UNCHANGED = "unchanged"
    AUTOMATED_SOURCE = "automated_source"


def is_automated_source(source: str | None) -> bool:
    """Return True when ``source`` names an automated or sub-agent session."""
    return source is not None and source in AUTOMATED_EXTRACTION_SOURCES


def content_hash(messages: Sequence[SkippableMessage]) -> str:
    """Return the sha256 hex digest of the role-filtered transcript.

    Each kept message contributes ``[role, content, timestamp]`` in order, as
    canonical JSON. Timestamps are normalized so the same instant hashes the
    same whether it arrives as a datetime or as an ISO string.
    """
    canonical = [
        [m.role, m.content, _canonical_timestamp(m.timestamp)]
        for m in keep_conversation_roles(messages)
    ]
    payload = json.dumps(canonical, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def evaluate_skip(
    messages: Sequence[SkippableMessage],
    source: str | None,
    stored_hash: str | None,
    stored_version: int | None,
) -> SkipReason | None:
    """Return why extraction should be skipped, or None to extract.

    ``stored_hash`` and ``stored_version`` are the values recorded by the last
    completed extraction; UNCHANGED requires both to match the current
    transcript and ``EXTRACTION_VERSION``.
    """
    if is_automated_source(source):
        return SkipReason.AUTOMATED_SOURCE

    kept = keep_conversation_roles(messages)
    if not kept:
        return SkipReason.NO_TEXT

    user_chars = sum(len(m.content.strip()) for m in kept if m.role == USER_ROLE)
    if user_chars < EXTRACTION_MIN_USER_CHARS:
        return SkipReason.TRIVIAL_USER_TEXT

    if stored_version == EXTRACTION_VERSION and stored_hash == content_hash(kept):
        return SkipReason.UNCHANGED

    return None


def evaluate_slice_skip(
    new_messages: Sequence[SkippableMessage],
    source: str | None,
    watermark: int | None,
    stored_hash: str | None,
    stored_version: int | None,
) -> SkipReason | None:
    """Return why extracting the messages above the watermark should be skipped, or None.

    ``new_messages`` are the messages above ``watermark``. A NULL watermark
    (never extracted) applies :func:`evaluate_skip` unchanged. With a
    watermark, nothing left after the role filter is UNCHANGED, and the stored
    hash is not consulted: edits to text below the watermark never re-extract.
    """
    if watermark is None:
        return evaluate_skip(new_messages, source, stored_hash, stored_version)
    if is_automated_source(source):
        return SkipReason.AUTOMATED_SOURCE
    if not keep_conversation_roles(new_messages):
        return SkipReason.UNCHANGED
    return evaluate_skip(new_messages, source, None, None)


def _canonical_timestamp(value: datetime | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return _iso_utc(value)
    try:
        return _iso_utc(datetime.fromisoformat(value.strip()))
    except ValueError:
        return value


def _iso_utc(value: datetime) -> str:
    if value.tzinfo is None:
        return value.isoformat()
    return value.astimezone(UTC).isoformat()
