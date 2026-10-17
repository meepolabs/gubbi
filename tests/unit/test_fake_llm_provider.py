"""Behavior tests for the test-only FakeLLMProvider canned shapes and markers."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import pytest

from gubbi.extraction.chunk_extraction import CHUNK_ENTRIES_SCHEMA
from gubbi.extraction.llm.fake_provider import FakeLLMProvider
from gubbi.extraction.llm.provider import (
    LLMMessage,
    LLMPermanentError,
    LLMProviderError,
    LLMRefusalError,
    LLMTransientError,
    LLMTruncatedError,
    LLMUsage,
)
from gubbi.extraction.service import ExtractionService

pytestmark = pytest.mark.unit

_CANNED_USAGE = LLMUsage(input=100, output=50)


def _msg(content: str) -> list[LLMMessage]:
    return [{"role": "user", "content": content}]


async def test_chunk_schema_yields_one_entry_under_testbench_topic() -> None:
    service = ExtractionService(FakeLLMProvider())

    result = await service.extract_chunk("[0] [undated] user: hello", [], "")

    assert result.topic_path == "testbench/fake"
    assert len(result.entries) == 1
    assert result.usage == _CANNED_USAGE


async def test_two_call_schemas_yield_testbench_topic_and_one_entry() -> None:
    service = ExtractionService(FakeLLMProvider())

    categorized = await service.categorize_chunk("[0] [undated] user: hello", [])
    extracted = await service.extract_chunk_entries(
        "[0] [undated] user: hello", categorized.topic_path, ""
    )

    assert categorized.topic_path == "testbench/fake"
    assert len(extracted.entries) == 1
    assert extracted.usage == _CANNED_USAGE


async def test_triage_schema_says_worth_recording() -> None:
    verdict = await ExtractionService(FakeLLMProvider()).triage("[0] [undated] user: hello")

    assert verdict.worth_recording is True
    assert verdict.usage == _CANNED_USAGE


@pytest.mark.parametrize(
    ("marker", "error_class", "expected_usage"),
    [
        pytest.param("FAKE_LLM_REFUSE", LLMRefusalError, _CANNED_USAGE, id="refuse"),
        pytest.param("FAKE_LLM_TRUNCATE", LLMTruncatedError, _CANNED_USAGE, id="truncate"),
        pytest.param("FAKE_LLM_FAIL_PERMANENT", LLMPermanentError, None, id="permanent"),
        pytest.param("FAKE_LLM_FAIL", LLMTransientError, None, id="transient"),
    ],
)
async def test_marker_raises_matching_error(
    marker: str,
    error_class: type[LLMProviderError],
    expected_usage: LLMUsage | None,
) -> None:
    with pytest.raises(LLMProviderError) as caught:
        await FakeLLMProvider().complete(
            _msg(f"before {marker} after"), "sys", CHUNK_ENTRIES_SCHEMA
        )

    assert type(caught.value) is error_class
    assert caught.value.usage == expected_usage


async def _entries_via_single_call(transcript: str) -> list[tuple[str, int | None]]:
    result = await ExtractionService(FakeLLMProvider()).extract_chunk(transcript, [], "")
    return [(e.content, e.source_message_index) for e in result.entries]


async def _entries_via_two_calls(transcript: str) -> list[tuple[str, int | None]]:
    result = await ExtractionService(FakeLLMProvider()).extract_chunk_entries(
        transcript, "testbench/fake", ""
    )
    return [(e.content, e.source_message_index) for e in result.entries]


_CANNED_ENTRY = ("fake decision marker from FakeLLMProvider", 0)


@pytest.mark.parametrize(
    "extract",
    [
        pytest.param(_entries_via_single_call, id="single-call"),
        pytest.param(_entries_via_two_calls, id="two-call"),
    ],
)
@pytest.mark.parametrize(
    ("transcript", "expected"),
    [
        pytest.param(
            "[4] [undated] user: hello\n[5] [undated] assistant: hi",
            [_CANNED_ENTRY],
            id="no-marker-keeps-canned-entry",
        ),
        pytest.param(
            "[4] [undated] user: hello\n[5] [undated] assistant: FAKE_LLM_ENTRY:tail-1 noted",
            [("fake entry tail-1", 5)],
            id="one-marker",
        ),
        pytest.param(
            "[2] [undated] user: FAKE_LLM_ENTRY:a1\n"
            "[3] [undated] assistant: ok\n"
            "[7] [undated] user: and FAKE_LLM_ENTRY:b2",
            [("fake entry a1", 2), ("fake entry b2", 7)],
            id="two-markers-on-different-lines",
        ),
        pytest.param(
            "[8] [undated] user: first line\nsecond line\nthird FAKE_LLM_ENTRY:deep\n"
            "[9] [undated] assistant: done",
            [("fake entry deep", 8)],
            id="marker-inside-multi-line-message",
        ),
    ],
)
async def test_entry_marker_yields_one_entry_per_marker_at_its_message_index(
    extract: Callable[[str], Awaitable[list[tuple[str, int | None]]]],
    transcript: str,
    expected: list[tuple[str, int | None]],
) -> None:
    assert await extract(transcript) == expected


@pytest.mark.parametrize(
    ("marker", "error_class"),
    [
        pytest.param("FAKE_LLM_REFUSE", LLMRefusalError, id="refuse"),
        pytest.param("FAKE_LLM_TRUNCATE", LLMTruncatedError, id="truncate"),
        pytest.param("FAKE_LLM_FAIL_PERMANENT", LLMPermanentError, id="permanent"),
        pytest.param("FAKE_LLM_FAIL", LLMTransientError, id="transient"),
    ],
)
async def test_failure_marker_takes_precedence_over_entry_marker(
    marker: str, error_class: type[LLMProviderError]
) -> None:
    service = ExtractionService(FakeLLMProvider())

    with pytest.raises(LLMProviderError) as caught:
        await service.extract_chunk_entries(
            f"[0] [undated] user: FAKE_LLM_ENTRY:x\n[1] [undated] user: {marker}",
            "testbench/fake",
            "",
        )

    assert type(caught.value) is error_class
