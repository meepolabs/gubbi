"""Behavior tests for the test-only FakeLLMProvider canned shapes and markers."""

from __future__ import annotations

import pytest

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
from gubbi.extraction.service import (
    _CATEGORIZE_SCHEMA,
    _EXTRACT_SCHEMA,
    ExtractionService,
)

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


async def test_categorize_schema_returns_unchanged_canned_topic() -> None:
    response = await FakeLLMProvider().complete(_msg("hi"), "sys", _CATEGORIZE_SCHEMA)

    assert response.content == {
        "topic_path": "testbench/fake",
        "topic_title": "Testbench Fake Topic",
        "summary": "Canned categorization summary from FakeLLMProvider.",
        "confidence": 0.9,
    }


async def test_extract_entries_schema_returns_unchanged_canned_entry() -> None:
    response = await FakeLLMProvider().complete(_msg("hi"), "sys", _EXTRACT_SCHEMA)

    assert response.content == {
        "entries": [
            {
                "content": "fake decision marker from FakeLLMProvider",
                "reasoning": "canned reasoning from FakeLLMProvider",
                "tags": ["fake", "testbench"],
            }
        ]
    }


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
        await FakeLLMProvider().complete(_msg(f"before {marker} after"), "sys", _EXTRACT_SCHEMA)

    assert type(caught.value) is error_class
    assert caught.value.usage == expected_usage
