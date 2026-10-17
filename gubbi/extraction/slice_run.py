"""What one extraction run reads from a conversation, and the watermark move on success.

A run reads the messages above the conversation's extraction watermark, a
short tail of already-extracted messages before them as context, and the
entries earlier runs drew from the same conversation. Skip rules look at the
new messages only. When the run succeeds, the watermark moves from the value
it was loaded against to the highest position loaded, in the transaction
that completes the run; a watermark another run moved first fails this one,
so it writes nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from gubbi.extraction.skips import SkipReason, content_hash, evaluate_slice_skip
from gubbi.models.conversation import keep_conversation_roles
from gubbi.storage.repositories import extraction_slice

if TYPE_CHECKING:
    import asyncpg

    from gubbi.crypto.cipher import ContentCipher
    from gubbi.storage.repositories.conversations import ExtractionState
    from gubbi.storage.repositories.extraction_slice import SliceMessage, ThreadEntry

__all__: list[str] = [
    "RunSlice",
    "WatermarkMovedError",
    "advance_or_raise",
    "load_run_slice",
    "skip_reason",
]


class WatermarkMovedError(RuntimeError):
    """The conversation's watermark no longer equals the value the run was loaded against."""


@dataclass(frozen=True)
class RunSlice:
    """The role-filtered messages one run reads, and where its watermark moves on success.

    ``tail`` and ``new`` keep only conversation turns; ``target`` is the
    highest stored position above ``base`` whatever its role, None when no
    message is above it.
    """

    base: int | None
    target: int | None
    tail: tuple[SliceMessage, ...]
    new: tuple[SliceMessage, ...]
    prior_entries: tuple[ThreadEntry, ...]

    @property
    def messages(self) -> tuple[SliceMessage, ...]:
        """Context tail followed by the new messages: the index space of the pipeline."""
        return (*self.tail, *self.new)

    @property
    def context_count(self) -> int:
        """Number of leading ``messages`` that are already-extracted context."""
        return len(self.tail)

    @property
    def content_hash(self) -> str:
        """Hash of the new messages; with no watermark, of the whole transcript."""
        return content_hash(self.new)


async def load_run_slice(
    conn: asyncpg.Connection, cipher: ContentCipher, conversation_id: int
) -> RunSlice:
    """Load the run's slice and the conversation's earlier entries on ``conn``."""
    loaded = await extraction_slice.load_extraction_slice(conn, cipher, conversation_id)
    prior = await extraction_slice.list_thread_entries(conn, cipher, conversation_id)
    return RunSlice(
        base=loaded.base,
        target=loaded.target,
        tail=tuple(keep_conversation_roles(loaded.tail)),
        new=tuple(keep_conversation_roles(loaded.new)),
        prior_entries=tuple(prior),
    )


def skip_reason(run_slice: RunSlice, state: ExtractionState) -> SkipReason | None:
    """Why the run should not call the model, judged on the new messages only."""
    return evaluate_slice_skip(
        run_slice.new,
        state.source,
        run_slice.base,
        state.extracted_content_hash,
        state.extraction_version,
    )


async def advance_or_raise(
    conn: asyncpg.Connection, conversation_id: int, base: int | None, target: int | None
) -> None:
    """Move the watermark from ``base`` to ``target``; nothing to move when ``target`` is None.

    Raises WatermarkMovedError when the stored watermark is no longer
    ``base``, so the caller's transaction rolls back and the run writes nothing.
    """
    if target is None:
        return
    if not await extraction_slice.advance_watermark(conn, conversation_id, base, target):
        msg = f"conversation {conversation_id} watermark moved since the run loaded it"
        raise WatermarkMovedError(msg)
