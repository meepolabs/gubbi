"""Tests for deterministic dedupe of extracted entries."""

from __future__ import annotations

import math
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import pytest

from gubbi.constants import DEDUPE_EXACT_COSINE, DEDUPE_NEAR_COSINE
from gubbi.extraction.dedupe import (
    DedupeAction,
    DedupeCandidate,
    DedupeDecision,
    classify,
    embed_texts,
    normalize_text,
)
from gubbi.storage.repositories.entries import DedupeNeighbour

DAY = date(2026, 6, 1)
OTHER_DAY = date(2026, 6, 2)
CLIENT = "claude"
BASE = (1.0, 0.0)


def _at_cosine(cosine: float) -> tuple[float, float]:
    """Unit vector whose cosine with ``BASE`` is ``cosine``."""
    return (cosine, math.sqrt(1.0 - cosine * cosine))


def _existing(
    content: str = "Signed the lease",
    *,
    entry_id: int = 7,
    day: date = DAY,
    client: str | None = CLIENT,
    embedding: tuple[float, ...] | None = BASE,
) -> DedupeNeighbour:
    return DedupeNeighbour(
        id=entry_id, content=content, date=day, client=client, embedding=embedding
    )


def _candidate(
    content: str = "Moved to a new flat",
    *,
    day: date = DAY,
    embedding: Sequence[float] | None = BASE,
) -> DedupeCandidate:
    return DedupeCandidate(content=content, event_date=day, embedding=embedding)


SUPPRESSED_BY_EXISTING = DedupeDecision(DedupeAction.SUPPRESS, existing_entry_id=7)
NEAR_EXISTING = DedupeDecision(DedupeAction.INSERT_NEAR, existing_entry_id=7)
INSERT = DedupeDecision(DedupeAction.INSERT)


@pytest.mark.parametrize(
    ("candidate", "existing", "expected"),
    [
        pytest.param(
            _candidate("  signed THE lease. ", embedding=None),
            _existing("Signed the lease", embedding=None),
            SUPPRESSED_BY_EXISTING,
            id="exact-normalized-text-without-embeddings",
        ),
        pytest.param(
            _candidate(embedding=_at_cosine(DEDUPE_EXACT_COSINE)),
            _existing(),
            SUPPRESSED_BY_EXISTING,
            id="cosine-at-exact-boundary",
        ),
        pytest.param(
            _candidate(embedding=_at_cosine(DEDUPE_EXACT_COSINE - 0.001)),
            _existing(),
            NEAR_EXISTING,
            id="cosine-just-below-exact-boundary",
        ),
        pytest.param(
            _candidate(embedding=_at_cosine(DEDUPE_NEAR_COSINE)),
            _existing(),
            NEAR_EXISTING,
            id="cosine-at-near-boundary",
        ),
        pytest.param(
            _candidate(embedding=_at_cosine(DEDUPE_NEAR_COSINE - 0.001)),
            _existing(),
            INSERT,
            id="cosine-below-near-boundary",
        ),
        pytest.param(
            _candidate(embedding=BASE),
            _existing(day=OTHER_DAY),
            INSERT,
            id="different-day",
        ),
        pytest.param(
            _candidate(embedding=BASE),
            _existing(client="chatgpt"),
            INSERT,
            id="different-client",
        ),
        pytest.param(
            _candidate(embedding=BASE),
            _existing(client=None),
            SUPPRESSED_BY_EXISTING,
            id="null-client-legacy-entry",
        ),
        pytest.param(
            _candidate("Moved to a new flat", embedding=None),
            _existing("Signed the lease", embedding=BASE),
            INSERT,
            id="unembedded-candidate-with-different-text",
        ),
    ],
)
def test_classify_against_existing_entry(
    candidate: DedupeCandidate, existing: DedupeNeighbour, expected: DedupeDecision
) -> None:
    assert classify([candidate], [existing], CLIENT) == [expected]


def test_classify_suppresses_within_job_repeat_by_earlier_candidate_index() -> None:
    candidates = [
        _candidate("Booked flights to Lisbon", embedding=BASE),
        _candidate("Unrelated", embedding=(0.0, 1.0)),
        _candidate("booked flights to lisbon!", embedding=None),
        _candidate("Booked plane tickets to Lisbon", embedding=_at_cosine(0.93)),
    ]

    decisions = classify(candidates, [], CLIENT)

    assert decisions == [
        INSERT,
        INSERT,
        DedupeDecision(DedupeAction.SUPPRESS, earlier_candidate=0),
        DedupeDecision(DedupeAction.INSERT_NEAR, earlier_candidate=0),
    ]


def test_classify_within_job_repeat_on_another_day_is_inserted() -> None:
    candidates = [_candidate(embedding=BASE), _candidate(day=OTHER_DAY, embedding=BASE)]

    assert classify(candidates, [], CLIENT) == [INSERT, INSERT]


def test_classify_existing_entry_wins_over_earlier_candidate() -> None:
    candidates = [
        _candidate("Signed the lease", embedding=BASE),
        _candidate("signed the lease", embedding=BASE),
    ]

    decisions = classify(candidates, [_existing()], CLIENT)

    assert decisions == [SUPPRESSED_BY_EXISTING, SUPPRESSED_BY_EXISTING]


def test_classify_existing_near_match_wins_over_closer_earlier_near_match() -> None:
    existing = _existing(embedding=(1.0, 0.0, 0.0))
    first = _candidate("first", embedding=(0.75, 0.645, math.sqrt(1 - 0.75**2 - 0.645**2)))
    second = _candidate("second", embedding=(0.91, math.sqrt(1 - 0.91**2), 0.0))

    decisions = classify([first, second], [existing], CLIENT)

    assert decisions == [INSERT, NEAR_EXISTING]


def test_classify_exact_earlier_candidate_beats_near_existing_entry() -> None:
    near = _at_cosine(0.93)
    candidates = [_candidate("first", embedding=near), _candidate("second", embedding=near)]
    existing = _existing(embedding=BASE)

    decisions = classify(candidates, [existing], CLIENT)

    assert decisions == [NEAR_EXISTING, DedupeDecision(DedupeAction.SUPPRESS, earlier_candidate=0)]


def test_classify_suppressed_candidate_is_not_a_match_target() -> None:
    existing = _existing(embedding=(1.0, 0.0, 0.0))
    suppressed = _candidate("first", embedding=(0.98, math.sqrt(1 - 0.98**2), 0.0))
    later = _candidate("second", embedding=(0.93, math.sqrt(1 - 0.93**2), 0.0))

    decisions = classify([suppressed, later], [existing], CLIENT)

    assert decisions == [SUPPRESSED_BY_EXISTING, NEAR_EXISTING]


def test_classify_picks_most_similar_existing_entry() -> None:
    existing = [
        _existing(entry_id=1, embedding=_at_cosine(0.91)),
        _existing(entry_id=2, embedding=_at_cosine(0.95)),
        _existing(entry_id=3, embedding=_at_cosine(0.92)),
    ]

    decisions = classify([_candidate(embedding=BASE)], existing, CLIENT)

    assert decisions == [DedupeDecision(DedupeAction.INSERT_NEAR, existing_entry_id=2)]


def test_classify_none_client_matches_only_null_client_entries() -> None:
    existing = [_existing(entry_id=1, client=CLIENT), _existing(entry_id=2, client=None)]

    decisions = classify([_candidate(embedding=BASE)], existing, None)

    assert decisions == [DedupeDecision(DedupeAction.SUPPRESS, existing_entry_id=2)]


def test_classify_is_deterministic_for_same_input_order() -> None:
    candidates = [
        _candidate("a", embedding=BASE),
        _candidate("b", embedding=_at_cosine(0.95)),
        _candidate("c", embedding=_at_cosine(0.98)),
        _candidate("A.", embedding=None),
    ]
    existing = [
        _existing(entry_id=1, embedding=_at_cosine(0.95)),
        _existing(entry_id=2, embedding=_at_cosine(0.95)),
    ]

    first = classify(candidates, existing, CLIENT)

    assert all(classify(candidates, existing, CLIENT) == first for _ in range(5))
    assert first[1] == DedupeDecision(DedupeAction.SUPPRESS, existing_entry_id=1)


def test_classify_does_not_mutate_inputs() -> None:
    candidates = (_candidate(embedding=BASE), _candidate(embedding=BASE))
    existing = (_existing(),)
    snapshot = (candidates, existing)

    classify(candidates, existing, CLIENT)

    assert (candidates, existing) == snapshot


def test_classify_rejects_mismatched_embedding_dimensions() -> None:
    with pytest.raises(ValueError, match="dimensions differ"):
        classify([_candidate(embedding=(1.0, 0.0, 0.0))], [_existing()], CLIENT)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("  Signed\tthe   LEASE. ", "signed the lease", id="case-space-punct"),
        pytest.param("\uff33igned", "signed", id="fullwidth-folds"),
        pytest.param("Done?!", "done", id="multiple-trailing-marks"),
    ],
)
def test_normalize_text(text: str, expected: str) -> None:
    assert normalize_text(text) == expected


@dataclass
class _RecordingEncoder:
    thread_ids: list[int]

    def encode(self, text: str) -> list[float]:
        self.thread_ids.append(threading.get_ident())
        return [float(len(text))]


async def test_embed_texts_encodes_in_order_off_the_event_loop_thread() -> None:
    encoder = _RecordingEncoder(thread_ids=[])

    vectors = await embed_texts(encoder, ["a", "bbb", "cc"])

    assert vectors == [[1.0], [3.0], [2.0]]
    assert threading.get_ident() not in encoder.thread_ids


async def test_embed_texts_empty_input_does_not_call_encoder() -> None:
    encoder = _RecordingEncoder(thread_ids=[])

    assert await embed_texts(encoder, []) == []
    assert encoder.thread_ids == []
