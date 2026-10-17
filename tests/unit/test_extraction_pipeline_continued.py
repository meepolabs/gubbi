"""Pipeline behavior on a continued thread: a context tail and earlier entries."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import pytest

from gubbi.constants import PRIOR_ENTRIES_MAX_TOKENS
from gubbi.extraction.chunking import chunk_messages, estimate_tokens
from tests.fixtures.extraction_pipeline import (
    PipelineProvider,
    Step,
    category_output,
    entries_output,
    entry,
    large_conversation,
    output,
    run,
    small_conversation,
)

pytestmark = pytest.mark.unit

BOUNDARY_PREFIX = "Boundary: transcript messages [0] through"
_LAST_MESSAGE = len(large_conversation()) - 1


@dataclass(frozen=True)
class _Prior:
    content: str
    entry_date: str | None = "2025-12-01"


def _extract_calls(mode: str, chunk_count: int) -> range:
    first = 1 if mode == "two_call" else 0
    return range(first, first + chunk_count)


def _decision(i: int) -> dict[str, Any]:
    return entry(f"decision-{i}", source=_LAST_MESSAGE)


def _per_chunk_steps(mode: str, chunk_count: int) -> list[Step]:
    if mode == "two_call":
        return [category_output(), *(entries_output(_decision(i)) for i in range(chunk_count))]
    return [output(entries=[_decision(i)]) for i in range(chunk_count)]


@pytest.mark.parametrize(
    ("source", "context_count", "kept"),
    [
        pytest.param(1, 2, False, id="index context_count-1 dropped"),
        pytest.param(2, 2, True, id="index context_count kept"),
        pytest.param(None, 2, False, id="null index with context dropped"),
        pytest.param(None, 0, True, id="null index without context kept"),
    ],
)
async def test_candidates_drawn_from_the_context_tail_are_dropped(
    source: int | None, context_count: int, kept: bool
) -> None:
    provider = PipelineProvider([output(entries=[entry("tail check", None, source)])])

    outcome = await run(provider, small_conversation(), context_count=context_count)

    assert [c.content for c in outcome.candidates] == (["tail check"] if kept else [])


@pytest.mark.parametrize(("source", "day"), [(1, date(2026, 2, 2)), (2, date(2026, 2, 3))])
async def test_candidate_after_the_tail_is_dated_from_its_own_message(
    source: int, day: date
) -> None:
    provider = PipelineProvider([output(entries=[entry("dated", None, source)])])

    outcome = await run(provider, small_conversation(), context_count=1)

    assert [c.entry_date for c in outcome.candidates] == [day]


@pytest.mark.parametrize("mode", ["single", "two_call"])
async def test_thread_entries_reach_every_extract_call_before_the_runs_candidates(
    mode: str,
) -> None:
    messages = large_conversation()
    chunk_count = len(chunk_messages(messages))
    provider = PipelineProvider(_per_chunk_steps(mode, chunk_count))
    priors = [_Prior("thread-old"), _Prior("thread-new", None)]

    await run(provider, messages, mode=mode, prior_entries=priors)

    for chunk_index, call in enumerate(_extract_calls(mode, chunk_count)):
        lines = provider.prior_section_of(call).split("\n")
        assert lines[:2] == ["- [2025-12-01] thread-old", "- [undated] thread-new"]
        assert [line.rsplit(" ", 1)[1] for line in lines[2:]] == [
            f"decision-{i}" for i in range(chunk_index)
        ]


@pytest.mark.parametrize("context_count", [0, 3])
async def test_thread_entries_are_cut_oldest_first_to_the_prior_budget(
    context_count: int,
) -> None:
    priors = [_Prior(f"thread-{i:03d} " + "y" * 200) for i in range(120)]
    messages = large_conversation()
    chunk_count = len(chunk_messages(messages))
    provider = PipelineProvider(_per_chunk_steps("single", chunk_count))

    await run(provider, messages, context_count=context_count, prior_entries=priors)

    for call in range(chunk_count):
        section = provider.prior_section_of(call)
        assert estimate_tokens(section) <= PRIOR_ENTRIES_MAX_TOKENS
        assert "thread-000" not in section
        assert "thread-119" in section
        assert section.endswith("thread-119 " + "y" * 200 if call == 0 else f"decision-{call - 1}")


@pytest.mark.parametrize("mode", ["single", "two_call"])
async def test_boundary_note_names_the_context_range_in_every_extract_call(mode: str) -> None:
    messages = large_conversation()
    chunk_count = len(chunk_messages(messages))
    provider = PipelineProvider(_per_chunk_steps(mode, chunk_count))

    await run(provider, messages, mode=mode, context_count=4)

    note = (
        f"{BOUNDARY_PREFIX} [3] are earlier context whose entries were already recorded; "
        "read them for context only and record entries only from message [4] onward."
    )
    calls = _extract_calls(mode, chunk_count)
    assert [provider.prior_section_of(call).split("\n")[0] for call in calls] == [note] * len(calls)


async def test_no_boundary_note_without_context_messages() -> None:
    messages = large_conversation()
    chunk_count = len(chunk_messages(messages))
    provider = PipelineProvider(_per_chunk_steps("single", chunk_count))

    await run(provider, messages)

    assert provider.prior_section_of(0) == "(none)"
    assert not any(BOUNDARY_PREFIX in content for content in provider.user_contents)
