"""Deterministic dedupe of extracted entries against each other and stored entries.

A candidate is compared with stored entries of the same event day whose client
is the job's client or unknown (``NULL``), and with earlier candidates of the
same job that are going to be inserted. Two texts match exactly when their
normalized forms are equal or their embeddings reach ``DEDUPE_EXACT_COSINE``;
they are near when the embeddings reach ``DEDUPE_NEAR_COSINE``. Stored entries
always win over candidates and nothing is ever deleted: an exact match
suppresses the candidate, a near match inserts it flagged with what it
resembles.

``classify`` is pure: no I/O, no clock, and the same inputs in the same order
always produce the same decisions. ``embed_texts`` is the only async helper.
"""

from __future__ import annotations

import asyncio
import math
import unicodedata
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

from gubbi.constants import DEDUPE_EXACT_COSINE, DEDUPE_NEAR_COSINE

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import date

__all__: list[str] = [
    "DedupeAction",
    "DedupeCandidate",
    "DedupeDecision",
    "ExistingEntry",
    "TextEncoder",
    "classify",
    "embed_texts",
    "normalize_text",
]

_TRAILING_PUNCTUATION = ".!?;:,"


class DedupeAction(StrEnum):
    """What to do with one extracted candidate."""

    INSERT = "insert"
    INSERT_NEAR = "insert_near"
    SUPPRESS = "suppress"


@dataclass(frozen=True)
class DedupeCandidate:
    """An extracted entry awaiting a dedupe decision."""

    content: str
    event_date: date
    embedding: Sequence[float] | None = None


class ExistingEntry(Protocol):
    """A stored entry that a candidate may repeat."""

    @property
    def id(self) -> int:
        """Entry primary key."""
        ...

    @property
    def content(self) -> str:
        """Decrypted entry headline."""
        ...

    @property
    def date(self) -> date:
        """Event day of the entry."""
        ...

    @property
    def client(self) -> str | None:
        """AI client the entry came from; ``None`` when unknown."""
        ...

    @property
    def embedding(self) -> Sequence[float] | None:
        """Stored embedding, or ``None`` when the entry is not indexed yet."""
        ...


class TextEncoder(Protocol):
    """Anything that turns text into an embedding vector synchronously."""

    def encode(self, text: str) -> list[float]:
        """Return the embedding of ``text``."""
        ...


@dataclass(frozen=True)
class DedupeDecision:
    """Outcome for one candidate.

    For ``INSERT_NEAR`` and ``SUPPRESS`` exactly one of ``existing_entry_id``
    (a stored entry) or ``earlier_candidate`` (index of an earlier candidate in
    the same ``classify`` call) names the match; for ``INSERT`` both are None.
    """

    action: DedupeAction
    existing_entry_id: int | None = None
    earlier_candidate: int | None = None


@dataclass(frozen=True)
class _Match:
    score: float
    existing_entry_id: int | None = None
    earlier_candidate: int | None = None


def normalize_text(text: str) -> str:
    """Case-, width- and whitespace-insensitive form used for exact text matching."""
    folded = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(folded.split()).rstrip(_TRAILING_PUNCTUATION).rstrip()


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"embedding dimensions differ: {len(a)} != {len(b)}")
    norm = math.sqrt(math.sumprod(a, a)) * math.sqrt(math.sumprod(b, b))
    if norm == 0.0:
        return 0.0
    return math.sumprod(a, b) / norm


def _similarity(
    text: str,
    embedding: Sequence[float] | None,
    other_text: str,
    other_embedding: Sequence[float] | None,
) -> float | None:
    """1.0 for equal normalized text, else cosine; None when not comparable."""
    if text == other_text:
        return 1.0
    if embedding is None or other_embedding is None:
        return None
    return _cosine(embedding, other_embedding)


def _best_existing(
    candidate: DedupeCandidate,
    normalized: str,
    existing: Sequence[tuple[ExistingEntry, str]],
    client: str | None,
) -> _Match | None:
    best: _Match | None = None
    for entry, entry_normalized in existing:
        if entry.date != candidate.event_date:
            continue
        if entry.client is not None and entry.client != client:
            continue
        score = _similarity(normalized, candidate.embedding, entry_normalized, entry.embedding)
        if score is not None and (best is None or score > best.score):
            best = _Match(score=score, existing_entry_id=entry.id)
    return best


def _best_earlier(
    candidate: DedupeCandidate,
    normalized: str,
    kept: Sequence[tuple[int, DedupeCandidate, str]],
) -> _Match | None:
    best: _Match | None = None
    for index, earlier, earlier_normalized in kept:
        if earlier.event_date != candidate.event_date:
            continue
        score = _similarity(normalized, candidate.embedding, earlier_normalized, earlier.embedding)
        if score is not None and (best is None or score > best.score):
            best = _Match(score=score, earlier_candidate=index)
    return best


def _decide(existing_match: _Match | None, earlier_match: _Match | None) -> DedupeDecision:
    """Exact before near; at each level a stored entry before an earlier candidate."""
    for threshold, action in (
        (DEDUPE_EXACT_COSINE, DedupeAction.SUPPRESS),
        (DEDUPE_NEAR_COSINE, DedupeAction.INSERT_NEAR),
    ):
        for match in (existing_match, earlier_match):
            if match is not None and match.score >= threshold:
                return DedupeDecision(
                    action=action,
                    existing_entry_id=match.existing_entry_id,
                    earlier_candidate=match.earlier_candidate,
                )
    return DedupeDecision(action=DedupeAction.INSERT)


def classify(
    candidates: Sequence[DedupeCandidate],
    existing: Sequence[ExistingEntry],
    client: str | None,
) -> list[DedupeDecision]:
    """Return one decision per candidate, in candidate order.

    ``client`` is the client of the job producing the candidates; a stored
    entry is a neighbour only if its client equals it or is unknown. Equal
    scores resolve to the first match in input order. A suppressed candidate
    is never itself a match target for later candidates: whatever it repeated
    is compared with them directly.
    """
    existing_normalized = [(entry, normalize_text(entry.content)) for entry in existing]
    kept: list[tuple[int, DedupeCandidate, str]] = []
    decisions: list[DedupeDecision] = []
    for index, candidate in enumerate(candidates):
        normalized = normalize_text(candidate.content)
        decision = _decide(
            _best_existing(candidate, normalized, existing_normalized, client),
            _best_earlier(candidate, normalized, kept),
        )
        if decision.action is not DedupeAction.SUPPRESS:
            kept.append((index, candidate, normalized))
        decisions.append(decision)
    return decisions


async def embed_texts(embedding_service: TextEncoder, texts: Sequence[str]) -> list[list[float]]:
    """Embed ``texts`` in order off the event loop, in one worker thread."""
    if not texts:
        return []
    items = list(texts)
    return await asyncio.to_thread(lambda: [embedding_service.encode(text) for text in items])
