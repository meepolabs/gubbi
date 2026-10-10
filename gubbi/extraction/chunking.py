"""Split a conversation transcript into LLM-sized chunks.

Pure functions only: no I/O, no logging, no provider calls. Token counts
are estimates (``ceil(chars / CHARS_PER_TOKEN)``), deliberately
conservative so a chunk never exceeds the model's real context budget.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol

from gubbi.constants import (
    CHARS_PER_TOKEN,
    CHUNK_OVERLAP_TOKENS,
    CHUNK_TARGET_TOKENS,
    PRIOR_ENTRIES_MAX_TOKENS,
)

UNDATED_LABEL = "[undated]"
PART_SEPARATOR = "\n"


class ChunkableMessage(Protocol):
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


class PriorEntry(Protocol):
    """An entry already extracted from earlier chunks."""

    @property
    def content(self) -> str:
        """Entry headline."""
        ...

    @property
    def entry_date(self) -> date | str | None:
        """Event day of the entry, if known."""
        ...


@dataclass(frozen=True)
class ChunkPart:
    """One serialized message, or one piece of a message larger than the target."""

    index: int
    header: str
    body: str

    @property
    def text(self) -> str:
        """Serialized form sent to the model."""
        return f"{self.header}{self.body}"


@dataclass(frozen=True)
class Chunk:
    """An ordered run of parts that fits one model call."""

    parts: tuple[ChunkPart, ...]

    @property
    def text(self) -> str:
        """Newline-joined serialized parts."""
        return PART_SEPARATOR.join(part.text for part in self.parts)

    @property
    def indices(self) -> tuple[int, ...]:
        """Sorted distinct message indices present in this chunk."""
        return tuple(sorted({part.index for part in self.parts}))

    @property
    def token_estimate(self) -> int:
        """Estimated tokens of :attr:`text`."""
        return estimate_tokens(self.text)


def estimate_tokens(text: str) -> int:
    """Conservative token estimate: ``ceil(len(text) / CHARS_PER_TOKEN)``."""
    return _tokens_for_chars(len(text))


def _tokens_for_chars(char_count: int) -> int:
    return math.ceil(char_count / CHARS_PER_TOKEN)


def _format_day(value: date | datetime | str | None) -> str:
    if value is None:
        return UNDATED_LABEL
    if isinstance(value, datetime):
        return f"[{value.date().isoformat()}]"
    if isinstance(value, date):
        return f"[{value.isoformat()}]"
    try:
        return f"[{datetime.fromisoformat(value.strip()).date().isoformat()}]"
    except ValueError:
        return UNDATED_LABEL


def _header(index: int, message: ChunkableMessage) -> str:
    return f"[{index}] {_format_day(message.timestamp)} {message.role}: "


def _max_body_chars(header: str, target_tokens: int) -> int:
    return max(1, math.floor(target_tokens * CHARS_PER_TOKEN) - len(header))


def _serialize(index: int, message: ChunkableMessage, target_tokens: int) -> list[ChunkPart]:
    header = _header(index, message)
    body = message.content
    if estimate_tokens(header + body) <= target_tokens:
        return [ChunkPart(index=index, header=header, body=body)]
    step = _max_body_chars(header, target_tokens)
    return [
        ChunkPart(index=index, header=header, body=body[start : start + step])
        for start in range(0, max(len(body), 1), step)
    ]


def _joined_chars(parts: Sequence[ChunkPart]) -> int:
    if not parts:
        return 0
    return sum(len(part.text) for part in parts) + len(PART_SEPARATOR) * (len(parts) - 1)


def _fits(parts: Sequence[ChunkPart], budget_tokens: int) -> bool:
    return _tokens_for_chars(_joined_chars(parts)) <= budget_tokens


def _overlap_tail(parts: Sequence[ChunkPart], overlap_tokens: int) -> list[ChunkPart]:
    """Longest run of whole trailing parts within the overlap budget.

    Never the whole chunk, so every following chunk adds new material.
    """
    tail: list[ChunkPart] = []
    for part in reversed(parts[1:]):
        candidate = [part, *tail]
        if not _fits(candidate, overlap_tokens):
            break
        tail = candidate
    return tail


@dataclass(frozen=True)
class _PackConfig:
    target_tokens: int
    overlap_tokens: int


def _start_next(
    previous: Sequence[ChunkPart], part: ChunkPart, cfg: _PackConfig
) -> list[ChunkPart]:
    seed = _overlap_tail(previous, cfg.overlap_tokens)
    while seed and not _fits([*seed, part], cfg.target_tokens):
        seed = seed[1:]
    return [*seed, part]


def _pack(parts: Sequence[ChunkPart], cfg: _PackConfig) -> tuple[Chunk, ...]:
    chunks: list[Chunk] = []
    current: list[ChunkPart] = []
    current_chars = 0
    for part in parts:
        grown_chars = current_chars + len(PART_SEPARATOR) + len(part.text)
        if current and _tokens_for_chars(grown_chars) <= cfg.target_tokens:
            current.append(part)
            current_chars = grown_chars
            continue
        if current:
            chunks.append(Chunk(parts=tuple(current)))
            current = _start_next(current, part, cfg)
        else:
            current = [part]
        current_chars = _joined_chars(current)
    if current:
        chunks.append(Chunk(parts=tuple(current)))
    return tuple(chunks)


def chunk_messages(
    messages: Sequence[ChunkableMessage],
    target_tokens: int = CHUNK_TARGET_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> tuple[Chunk, ...]:
    """Pack messages into chunks of at most ``target_tokens`` estimated tokens.

    Messages stay whole; only a single message larger than the target is cut
    into pieces, each carrying the message's header. Each chunk after the
    first repeats whole trailing parts of its predecessor, up to
    ``overlap_tokens``, when they fit beside the next new part.
    """
    if target_tokens < 1:
        raise ValueError("target_tokens must be positive")
    if not 0 <= overlap_tokens < target_tokens:
        raise ValueError("overlap_tokens must be in [0, target_tokens)")
    parts = [
        part
        for index, message in enumerate(messages)
        for part in _serialize(index, message, target_tokens)
    ]
    return _pack(parts, _PackConfig(target_tokens=target_tokens, overlap_tokens=overlap_tokens))


def _balanced_cut(parts: Sequence[ChunkPart]) -> int:
    total = _joined_chars(parts)
    return min(
        range(1, len(parts)),
        key=lambda cut: abs(2 * _joined_chars(parts[:cut]) - total),
    )


def split_in_two(chunk: Chunk) -> tuple[Chunk, Chunk]:
    """Split a chunk into two halves of roughly equal size, without overlap.

    Cuts at the message boundary nearest the midpoint; a chunk holding a
    single part has that part's body cut in half instead.
    """
    parts = chunk.parts
    if len(parts) >= 2:
        cut = _balanced_cut(parts)
        return Chunk(parts=parts[:cut]), Chunk(parts=parts[cut:])
    if not parts:
        raise ValueError("cannot split an empty chunk")
    only = parts[0]
    mid = len(only.body) // 2
    return (
        Chunk(parts=(ChunkPart(index=only.index, header=only.header, body=only.body[:mid]),)),
        Chunk(parts=(ChunkPart(index=only.index, header=only.header, body=only.body[mid:]),)),
    )


def render_prior_entries(
    entries: Sequence[PriorEntry],
    max_tokens: int = PRIOR_ENTRIES_MAX_TOKENS,
) -> str:
    """Render entries as ``- [YYYY-MM-DD] content`` lines within ``max_tokens``.

    ``entries`` is ordered oldest to newest; the newest contiguous run that
    fits the budget is kept, in original order. Returns ``""`` when nothing fits.
    """
    kept: list[str] = []
    for entry in reversed(entries):
        candidate = [f"- {_format_day(entry.entry_date)} {entry.content}", *kept]
        if estimate_tokens(PART_SEPARATOR.join(candidate)) > max_tokens:
            break
        kept = candidate
    return PART_SEPARATOR.join(kept)
