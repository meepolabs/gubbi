"""Behavior tests for the pure transcript chunker."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from itertools import pairwise

import pytest

from gubbi.constants import (
    CHARS_PER_TOKEN,
    CHUNK_OVERLAP_TOKENS,
    CHUNK_TARGET_TOKENS,
    PRIOR_ENTRIES_MAX_TOKENS,
)
from gubbi.extraction.chunking import (
    Chunk,
    chunk_messages,
    estimate_tokens,
    render_prior_entries,
    split_in_two,
)
from gubbi.models.conversation import Message


@dataclass(frozen=True)
class _Msg:
    role: str
    content: str
    timestamp: datetime | str | None = None


@dataclass(frozen=True)
class _Entry:
    content: str
    entry_date: date | str | None


def _dense_conversation(count: int, chars: int) -> list[_Msg]:
    return [
        _Msg(
            role="user" if i % 2 == 0 else "assistant",
            content=f"m{i} " + "x" * chars,
            timestamp=datetime(2026, 1, 1 + i % 28, 12, tzinfo=UTC),
        )
        for i in range(count)
    ]


def _covered(chunks: tuple[Chunk, ...]) -> set[int]:
    return {index for chunk in chunks for index in chunk.indices}


@pytest.mark.parametrize(
    ("timestamp", "expected"),
    [
        (datetime(2026, 3, 4, 23, 59, tzinfo=UTC), "[0] [2026-03-04] user: hi"),
        ("2026-03-04T10:00:00+00:00", "[0] [2026-03-04] user: hi"),
        ("2026-03-04", "[0] [2026-03-04] user: hi"),
        (None, "[0] [undated] user: hi"),
        ("not a date", "[0] [undated] user: hi"),
    ],
    ids=["datetime", "iso-datetime-str", "iso-date-str", "none", "unparseable-str"],
)
def test_serializes_message_with_index_day_and_role(
    timestamp: datetime | str | None, expected: str
) -> None:
    chunks = chunk_messages([_Msg(role="user", content="hi", timestamp=timestamp)])

    assert chunks[0].text == expected


def test_accepts_stored_message_model() -> None:
    message = Message(role="assistant", content="ok", timestamp="2026-05-06T01:02:03")

    chunks = chunk_messages([message])

    assert chunks[0].text == "[0] [2026-05-06] assistant: ok"


@pytest.mark.parametrize(
    ("text", "expected"),
    [("", 0), ("a", 1), ("a" * 7, 2), ("a" * 8, 3)],
)
def test_estimates_tokens_as_ceiling_of_chars_over_ratio(text: str, expected: int) -> None:
    assert CHARS_PER_TOKEN == 3.5
    assert estimate_tokens(text) == expected


def test_dense_conversation_packs_within_target_covering_every_index_with_overlap() -> None:
    messages = _dense_conversation(count=400, chars=1_400)
    total_tokens = sum(estimate_tokens(m.content) for m in messages)
    assert total_tokens > 100_000

    largest_message_tokens = max(c.token_estimate for c in chunk_messages(messages[:1]))

    chunks = chunk_messages(messages)

    assert len(chunks) >= 3
    assert all(chunk.token_estimate <= CHUNK_TARGET_TOKENS for chunk in chunks)
    assert _covered(chunks) == set(range(len(messages)))
    for previous, following in pairwise(chunks):
        shared = set(previous.indices) & set(following.indices)
        shared_parts = [p for p in following.parts if p.index in shared]
        overlap_tokens = estimate_tokens("\n".join(p.text for p in shared_parts))
        assert shared, "neighbouring chunks share no message"
        assert overlap_tokens <= CHUNK_OVERLAP_TOKENS
        assert overlap_tokens >= CHUNK_OVERLAP_TOKENS - largest_message_tokens - 1
        assert max(shared) == previous.indices[-1]
        assert following.indices[0] == min(shared)


def test_no_overlap_when_overlap_budget_is_zero() -> None:
    messages = _dense_conversation(count=50, chars=350)

    chunks = chunk_messages(messages, target_tokens=1_000, overlap_tokens=0)

    indices = [i for chunk in chunks for i in chunk.indices]
    assert indices == list(range(len(messages)))


def test_whole_messages_are_never_split_when_they_fit() -> None:
    messages = _dense_conversation(count=60, chars=700)

    chunks = chunk_messages(messages, target_tokens=2_000, overlap_tokens=300)

    for chunk in chunks:
        for part in chunk.parts:
            assert part.body == messages[part.index].content


def test_single_message_over_target_splits_into_parts_within_target() -> None:
    target_chars = int(CHUNK_TARGET_TOKENS * CHARS_PER_TOKEN * 2.5)
    body = "".join(f"{n:07d}" for n in range(target_chars // 7))
    messages = [_Msg("user", "before"), _Msg("assistant", body), _Msg("user", "after")]

    chunks = chunk_messages(messages)

    assert len(chunks) >= 3
    assert all(chunk.token_estimate <= CHUNK_TARGET_TOKENS for chunk in chunks)
    pieces = [p.body for c in chunks for p in c.parts if p.index == 1]
    assert "".join(dict.fromkeys(pieces)) == body
    assert _covered(chunks) == {0, 1, 2}


@pytest.mark.parametrize("length", [0, 1, 2, 3, 4, 5, 7, 50, 99, 100, 101, 1_000])
def test_any_message_length_chunks_within_target(length: int) -> None:
    target = 30
    messages = [_Msg("user", "z" * length), _Msg("assistant", "w" * (length // 2))]

    chunks = chunk_messages(messages, target_tokens=target, overlap_tokens=5)

    assert all(chunk.token_estimate <= target for chunk in chunks)
    assert _covered(chunks) == {0, 1}


def test_empty_conversation_yields_no_chunks() -> None:
    assert chunk_messages([]) == ()


@pytest.mark.parametrize(
    ("target", "overlap"),
    [(0, 0), (-1, 0), (10, 10), (10, -1)],
)
def test_rejects_invalid_budgets(target: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_messages([_Msg("user", "a")], target_tokens=target, overlap_tokens=overlap)


def test_split_in_two_cuts_multi_message_chunk_at_boundary() -> None:
    (chunk,) = chunk_messages(_dense_conversation(count=9, chars=100))

    first, second = split_in_two(chunk)

    assert first.parts
    assert second.parts
    assert first.parts + second.parts == chunk.parts
    assert abs(len(first.text) - len(second.text)) <= max(len(p.text) for p in chunk.parts) + 1


def test_split_in_two_cuts_single_message_body() -> None:
    (chunk,) = chunk_messages([_Msg("user", "abcdefghij")])

    first, second = split_in_two(chunk)

    assert first.parts[0].body == "abcde"
    assert second.parts[0].body == "fghij"
    assert first.parts[0].header == second.parts[0].header == chunk.parts[0].header


def test_split_in_two_rejects_empty_chunk() -> None:
    with pytest.raises(ValueError):
        split_in_two(Chunk(parts=()))


def test_render_prior_entries_formats_date_and_content() -> None:
    entries = [_Entry("old", date(2026, 1, 2)), _Entry("new", "2026-01-03"), _Entry("nd", None)]

    rendered = render_prior_entries(entries)

    assert rendered == "- [2026-01-02] old\n- [2026-01-03] new\n- [undated] nd"


def test_render_prior_entries_keeps_newest_within_budget() -> None:
    entries = [_Entry(f"e{i:03d} " + "q" * 66, date(2026, 2, 1)) for i in range(400)]
    budget = 1_000

    rendered = render_prior_entries(entries, max_tokens=budget)

    lines = rendered.split("\n")
    assert estimate_tokens(rendered) <= budget
    assert 1 < len(lines) < len(entries)
    kept = [int(line.split(" ")[2][1:]) for line in lines]
    assert kept == list(range(len(entries) - len(lines), len(entries)))


def test_render_prior_entries_default_budget_bounds_output() -> None:
    entries = [_Entry("r" * 500, None) for _ in range(500)]

    rendered = render_prior_entries(entries)

    assert 0 < estimate_tokens(rendered) <= PRIOR_ENTRIES_MAX_TOKENS


@pytest.mark.parametrize(
    "entries",
    [[], [_Entry("s" * 10_000, None)]],
    ids=["no-entries", "newest-alone-over-budget"],
)
def test_render_prior_entries_empty_when_nothing_fits(entries: list[_Entry]) -> None:
    assert render_prior_entries(entries, max_tokens=100) == ""
