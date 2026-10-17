"""Behavior tests for per-chunk extraction calls: schemas, parsing, service calls."""

from __future__ import annotations

import json
import traceback
from collections.abc import Iterator, Mapping
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from gubbi.extraction.chunk_extraction import (
    CHUNK_CATEGORIZE_SCHEMA,
    CHUNK_ENTRIES_SCHEMA,
    CHUNK_EXTRACTION_SCHEMA,
    TRIAGE_SCHEMA,
    ChunkEntry,
    ChunkExtraction,
    TriageVerdict,
    parse_chunk_categorization,
    parse_chunk_entries,
    parse_chunk_extraction,
    parse_triage,
)
from gubbi.extraction.llm.provider import LLMResponse, LLMSchemaError, LLMUsage
from gubbi.extraction.service import ExtractionService

pytestmark = pytest.mark.unit

_USAGE = LLMUsage(input=120, output=40, cached_input=30, reasoning=7)
_SECRET = "zz-private-response-text-zz"


def _valid_output() -> dict[str, Any]:
    return {
        "topic_path": "health/fitness",
        "topic_title": "Fitness",
        "summary": "User planned a training block.",
        "entries": [
            {
                "content": "Decided to run a half marathon in spring",
                "why": "Wants a concrete goal",
                "tags": ["running", "goals"],
                "event_date": "2026-03-01",
                "source_message_index": 2,
            },
            {
                "content": "Prefers morning workouts",
                "why": "",
                "tags": [],
                "event_date": None,
                "source_message_index": None,
            },
        ],
    }


def _object_nodes(node: Any) -> Iterator[Mapping[str, Any]]:
    if isinstance(node, Mapping):
        if node.get("type") == "object":
            yield node
        for value in node.values():
            yield from _object_nodes(value)
    elif isinstance(node, list):
        for item in node:
            yield from _object_nodes(item)


@pytest.mark.parametrize(
    ("schema", "object_count"),
    [
        pytest.param(CHUNK_EXTRACTION_SCHEMA, 2, id="single-call"),
        pytest.param(CHUNK_CATEGORIZE_SCHEMA, 1, id="categorize"),
        pytest.param(CHUNK_ENTRIES_SCHEMA, 2, id="entries"),
        pytest.param(TRIAGE_SCHEMA, 1, id="triage"),
    ],
)
def test_schema_every_object_is_closed_and_requires_all_properties(
    schema: Mapping[str, Any], object_count: int
) -> None:
    objects = list(_object_nodes(schema))

    assert len(objects) == object_count
    for node in objects:
        assert node["additionalProperties"] is False
        assert sorted(node["required"]) == sorted(node["properties"])


def test_schema_declares_nullable_entry_fields_as_any_of_with_null() -> None:
    entry_props = CHUNK_EXTRACTION_SCHEMA["properties"]["entries"]["items"]["properties"]

    assert entry_props["event_date"] == {"anyOf": [{"type": "string"}, {"type": "null"}]}
    assert entry_props["source_message_index"] == {"anyOf": [{"type": "integer"}, {"type": "null"}]}


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(_valid_output(), id="dict"),
        pytest.param(json.dumps(_valid_output()), id="json-string"),
    ],
)
def test_valid_output_parses_to_chunk_extraction_with_usage(
    content: str | dict[str, Any],
) -> None:
    result = parse_chunk_extraction(content, _USAGE)

    assert result == ChunkExtraction(
        topic_path="health/fitness",
        topic_title="Fitness",
        summary="User planned a training block.",
        entries=(
            ChunkEntry(
                content="Decided to run a half marathon in spring",
                why="Wants a concrete goal",
                tags=("running", "goals"),
                event_date="2026-03-01",
                source_message_index=2,
            ),
            ChunkEntry(
                content="Prefers morning workouts",
                why="",
                tags=(),
                event_date=None,
                source_message_index=None,
            ),
        ),
        usage=_USAGE,
    )


def test_empty_entries_array_yields_zero_entries() -> None:
    output = {**_valid_output(), "entries": []}

    result = parse_chunk_extraction(output, _USAGE)

    assert result.entries == ()
    assert result.topic_path == "health/fitness"


def _leaky_output() -> dict[str, Any]:
    """A valid output whose free-text fields carry a marker that must never leak."""
    output = _valid_output()
    output["summary"] = _SECRET
    output["entries"][0]["why"] = _SECRET
    return output


def _without_top(key: str) -> dict[str, Any]:
    output = _leaky_output()
    del output[key]
    return output


def _with_top(key: str, value: Any) -> dict[str, Any]:
    return {**_leaky_output(), key: value}


def _without_entry(key: str) -> dict[str, Any]:
    output = _leaky_output()
    del output["entries"][0][key]
    return output


def _with_entry(key: str, value: Any) -> dict[str, Any]:
    output = _leaky_output()
    output["entries"][0][key] = value
    return output


@pytest.mark.parametrize(
    ("content", "location"),
    [
        pytest.param(_without_top("topic_path"), "topic_path", id="missing-topic-path"),
        pytest.param(_without_top("entries"), "entries", id="missing-entries"),
        pytest.param(_with_top("summary", 5), "summary", id="summary-not-string"),
        pytest.param(_with_top("entries", _SECRET), "entries", id="entries-not-list"),
        pytest.param(_with_top("confidence", 0.9), "<unknown>", id="extra-top-level-key"),
        pytest.param(_with_top(_SECRET, 1), "<unknown>", id="extra-key-named-like-content"),
        pytest.param(_without_entry("why"), "entries.0.why", id="missing-entry-why"),
        pytest.param(
            _without_entry("event_date"), "entries.0.event_date", id="missing-entry-nullable"
        ),
        pytest.param(_with_entry("tags", _SECRET), "entries.0.tags", id="tags-not-list"),
        pytest.param(
            _with_entry("source_message_index", "2"),
            "entries.0.source_message_index",
            id="index-as-string",
        ),
        pytest.param(
            _with_entry("source_message_index", True),
            "entries.0.source_message_index",
            id="index-as-bool",
        ),
        pytest.param(_with_entry("content", 42), "entries.0.content", id="content-not-string"),
    ],
)
def test_schema_violation_raises_schema_error_with_usage_and_no_content(
    content: dict[str, Any], location: str
) -> None:
    with pytest.raises(LLMSchemaError) as caught:
        parse_chunk_extraction(content, _USAGE)

    message = str(caught.value)
    assert caught.value.usage == _USAGE
    assert location in message
    assert _SECRET not in message
    assert _SECRET not in "".join(traceback.format_exception(caught.value))
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(f"not json {_SECRET}", id="not-json"),
        pytest.param(json.dumps([_SECRET]), id="json-array"),
        pytest.param(json.dumps(_SECRET), id="json-string"),
    ],
)
def test_non_object_output_raises_schema_error_with_usage_and_no_content(content: str) -> None:
    with pytest.raises(LLMSchemaError) as caught:
        parse_chunk_extraction(content, _USAGE)

    assert caught.value.usage == _USAGE
    assert _SECRET not in "".join(traceback.format_exception(caught.value))
    assert caught.value.__context__ is None


def _service_with(content: str | dict[str, Any]) -> tuple[ExtractionService, MagicMock]:
    llm = MagicMock()
    llm.complete = AsyncMock(
        return_value=LLMResponse(
            content=content,
            input_tokens=_USAGE.input,
            output_tokens=_USAGE.output,
            model="test-model",
            cached_input_tokens=_USAGE.cached_input,
            reasoning_tokens=_USAGE.reasoning,
        )
    )
    return ExtractionService(llm), llm


async def test_extract_chunk_sends_strict_schema_and_inputs_and_returns_usage() -> None:
    service, llm = _service_with(_valid_output())

    result = await service.extract_chunk(
        "[0] [2026-03-01] user: I will run a half marathon",
        ["health/fitness", "work/notes"],
        "- [2026-02-01] Started running",
    )

    assert result.usage == _USAGE
    assert len(result.entries) == 2
    messages, system_prompt, schema = llm.complete.await_args.args
    assert schema is CHUNK_EXTRACTION_SCHEMA
    assert "general-knowledge answers, coding help and small talk are not entries" in (
        system_prompt
    )
    (user_msg,) = messages
    assert user_msg["role"] == "user"
    for expected in (
        "[0] [2026-03-01] user: I will run a half marathon",
        "- health/fitness",
        "- work/notes",
        "- [2026-02-01] Started running",
    ):
        assert expected in user_msg["content"]


async def test_extract_chunk_raises_schema_error_carrying_call_usage() -> None:
    service, _ = _service_with({"topic_path": "x"})

    with pytest.raises(LLMSchemaError) as caught:
        await service.extract_chunk("[0] [undated] user: hi", [], "")

    assert caught.value.usage == _USAGE


# -- Two-call and triage parsing ---------------------------------------------------

_SIGNIFICANCE_RULE = "general-knowledge answers, coding help and small talk are not entries"


def test_categorization_parses_to_chunk_extraction_without_entries() -> None:
    output = {key: _valid_output()[key] for key in ("topic_path", "topic_title", "summary")}

    result = parse_chunk_categorization(output, _USAGE, attempts=2)

    assert result == ChunkExtraction(
        topic_path="health/fitness",
        topic_title="Fitness",
        summary="User planned a training block.",
        entries=(),
        usage=_USAGE,
        attempts=2,
    )


def test_entries_parse_to_chunk_extraction_under_the_given_topic() -> None:
    expected = parse_chunk_extraction(_valid_output(), _USAGE).entries

    result = parse_chunk_entries(
        {"entries": _valid_output()["entries"]}, _USAGE, topic_path="health/fitness"
    )

    assert result == ChunkExtraction(
        topic_path="health/fitness",
        topic_title="",
        summary="",
        entries=expected,
        usage=_USAGE,
    )


@pytest.mark.parametrize("worth_recording", [True, False])
def test_triage_parses_to_verdict_with_usage(worth_recording: bool) -> None:
    result = parse_triage(json.dumps({"worth_recording": worth_recording}), _USAGE, attempts=3)

    assert result == TriageVerdict(worth_recording=worth_recording, usage=_USAGE, attempts=3)


@pytest.mark.parametrize(
    ("parse", "content", "location"),
    [
        pytest.param(
            parse_chunk_categorization,
            {"topic_path": "x", "topic_title": "X", "summary": "s", "confidence": 0.9},
            "<unknown>",
            id="categorize-extra-key",
        ),
        pytest.param(
            lambda content, usage: parse_chunk_entries(content, usage, topic_path="x"),
            {"entries": [{**_valid_output()["entries"][0], "why": 1}]},
            "entries.0.why",
            id="entries-bad-why",
        ),
        pytest.param(
            parse_triage, {"worth_recording": "yes " + _SECRET}, "worth_recording", id="triage-str"
        ),
        pytest.param(parse_triage, f"not json {_SECRET}", "not a JSON object", id="triage-text"),
    ],
)
def test_two_call_and_triage_schema_violations_raise_schema_error_without_content(
    parse: Any, content: Any, location: str
) -> None:
    with pytest.raises(LLMSchemaError) as caught:
        parse(content, _USAGE)

    assert caught.value.usage == _USAGE
    assert location in str(caught.value)
    assert _SECRET not in "".join(traceback.format_exception(caught.value))
    assert caught.value.__context__ is None


async def test_categorize_chunk_sends_categorize_schema_topics_and_transcript() -> None:
    output = {key: _valid_output()[key] for key in ("topic_path", "topic_title", "summary")}
    service, llm = _service_with(output)

    result = await service.categorize_chunk("[0] [2026-03-01] user: hi", ["health/fitness"])

    assert result.topic_path == "health/fitness"
    assert result.usage == _USAGE
    (user_msg,), _, schema = llm.complete.await_args.args
    assert schema is CHUNK_CATEGORIZE_SCHEMA
    assert "- health/fitness" in user_msg["content"]
    assert user_msg["content"].endswith("## Transcript\n[0] [2026-03-01] user: hi")


async def test_extract_chunk_entries_sends_topic_prior_entries_and_significance_rule() -> None:
    service, llm = _service_with({"entries": _valid_output()["entries"]})

    result = await service.extract_chunk_entries(
        "[0] [2026-03-01] user: I will run", "health/fitness", "- [2026-02-01] Started running"
    )

    assert result.topic_path == "health/fitness"
    assert len(result.entries) == 2
    (user_msg,), system_prompt, schema = llm.complete.await_args.args
    assert schema is CHUNK_ENTRIES_SCHEMA
    assert _SIGNIFICANCE_RULE in system_prompt
    for expected in (
        "## Topic\nhealth/fitness\n",
        "- [2026-02-01] Started running",
        "[0] [2026-03-01] user: I will run",
    ):
        assert expected in user_msg["content"]


async def test_triage_sends_triage_schema_and_significance_rule() -> None:
    service, llm = _service_with({"worth_recording": False})

    result = await service.triage("[0] [undated] user: what is 2+2?")

    assert result.worth_recording is False
    assert result.usage == _USAGE
    (user_msg,), system_prompt, schema = llm.complete.await_args.args
    assert schema is TRIAGE_SCHEMA
    assert _SIGNIFICANCE_RULE in system_prompt
    assert user_msg["content"] == "## Transcript\n[0] [undated] user: what is 2+2?"
