"""Structured outputs of the per-chunk extraction calls.

The single-call schema returns topic, summary and entries in one object. The
two-call variant splits it into a categorize schema (topic and summary) and an
entries schema; both parse to :class:`ChunkExtraction` so the pipeline treats
the modes alike. The triage schema answers whether a chunk holds anything
worth recording.

The JSON schema is written in strict form (every property required,
``additionalProperties: false``, nullables as type unions with ``null``)
because strict structured-output modes reject anything looser. The model
output is validated with pydantic; a failure raises :class:`LLMSchemaError`
carrying the call's usage and naming only schema locations, never values.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from pydantic import BaseModel, ConfigDict, ValidationError

from gubbi.extraction.llm.provider import LLMSchemaError, LLMUsage

if TYPE_CHECKING:
    from pydantic_core import ErrorDetails

__all__: list[str] = [
    "CHUNK_CATEGORIZE_SCHEMA",
    "CHUNK_ENTRIES_SCHEMA",
    "CHUNK_EXTRACTION_SCHEMA",
    "TRIAGE_SCHEMA",
    "ChunkEntry",
    "ChunkExtraction",
    "TriageVerdict",
    "parse_chunk_categorization",
    "parse_chunk_entries",
    "parse_chunk_extraction",
    "parse_triage",
]

_ENTRY_SCHEMA: Final[Mapping[str, Any]] = {
    "type": "object",
    "properties": {
        "content": {"type": "string"},
        "why": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "event_date": {"type": ["string", "null"]},
        "source_message_index": {"type": ["integer", "null"]},
    },
    "required": ["content", "why", "tags", "event_date", "source_message_index"],
    "additionalProperties": False,
}

CHUNK_EXTRACTION_SCHEMA: Final[Mapping[str, Any]] = {
    "type": "object",
    "properties": {
        "topic_path": {"type": "string"},
        "topic_title": {"type": "string"},
        "summary": {"type": "string"},
        "entries": {"type": "array", "items": _ENTRY_SCHEMA},
    },
    "required": ["topic_path", "topic_title", "summary", "entries"],
    "additionalProperties": False,
}

CHUNK_CATEGORIZE_SCHEMA: Final[Mapping[str, Any]] = {
    "type": "object",
    "properties": {
        "topic_path": {"type": "string"},
        "topic_title": {"type": "string"},
        "summary": {"type": "string"},
    },
    "required": ["topic_path", "topic_title", "summary"],
    "additionalProperties": False,
}

CHUNK_ENTRIES_SCHEMA: Final[Mapping[str, Any]] = {
    "type": "object",
    "properties": {"entries": {"type": "array", "items": _ENTRY_SCHEMA}},
    "required": ["entries"],
    "additionalProperties": False,
}

TRIAGE_SCHEMA: Final[Mapping[str, Any]] = {
    "type": "object",
    "properties": {"worth_recording": {"type": "boolean"}},
    "required": ["worth_recording"],
    "additionalProperties": False,
}

_UNKNOWN_LOCATION = "<unknown>"


@dataclass(frozen=True)
class ChunkEntry:
    """One journal-worthy item the model found in a chunk."""

    content: str
    why: str
    tags: tuple[str, ...]
    event_date: str | None
    source_message_index: int | None


@dataclass(frozen=True)
class ChunkExtraction:
    """Topic, summary and entries for one chunk, plus the call's usage and outcome."""

    topic_path: str
    topic_title: str
    summary: str
    entries: tuple[ChunkEntry, ...]
    usage: LLMUsage
    finish_reason: str = "stop"
    attempts: int = 1
    unmetered_attempts: int = 0


@dataclass(frozen=True)
class TriageVerdict:
    """Whether a chunk holds anything worth recording, plus the call's usage and outcome."""

    worth_recording: bool
    usage: LLMUsage
    finish_reason: str = "stop"
    attempts: int = 1
    unmetered_attempts: int = 0


class _EntryOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    content: str
    why: str
    tags: list[str]
    event_date: str | None
    source_message_index: int | None


class _ChunkOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    topic_path: str
    topic_title: str
    summary: str
    entries: list[_EntryOutput]


class _CategoryOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    topic_path: str
    topic_title: str
    summary: str


class _EntriesOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    entries: list[_EntryOutput]


class _TriageOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    worth_recording: bool


_KNOWN_FIELDS: Final[frozenset[str]] = (
    frozenset(_ChunkOutput.model_fields)
    | frozenset(_EntryOutput.model_fields)
    | frozenset(_TriageOutput.model_fields)
)


def parse_chunk_extraction(
    content: str | Mapping[str, Any],
    usage: LLMUsage,
    *,
    finish_reason: str = "stop",
    attempts: int = 1,
    unmetered_attempts: int = 0,
) -> ChunkExtraction:
    """Validate model output against the chunk schema and return a ChunkExtraction.

    Raises:
        LLMSchemaError: the output is not a JSON object matching the schema.
    """
    output = _validate(
        _ChunkOutput, "chunk extraction", content, usage, attempts, unmetered_attempts
    )
    return ChunkExtraction(
        topic_path=output.topic_path,
        topic_title=output.topic_title,
        summary=output.summary,
        entries=tuple(_to_entry(entry) for entry in output.entries),
        usage=usage,
        finish_reason=finish_reason,
        attempts=attempts,
        unmetered_attempts=unmetered_attempts,
    )


def parse_chunk_categorization(
    content: str | Mapping[str, Any],
    usage: LLMUsage,
    *,
    finish_reason: str = "stop",
    attempts: int = 1,
    unmetered_attempts: int = 0,
) -> ChunkExtraction:
    """Validate categorize output; the result carries no entries.

    Raises:
        LLMSchemaError: the output is not a JSON object matching the schema.
    """
    output = _validate(
        _CategoryOutput, "chunk categorization", content, usage, attempts, unmetered_attempts
    )
    return ChunkExtraction(
        topic_path=output.topic_path,
        topic_title=output.topic_title,
        summary=output.summary,
        entries=(),
        usage=usage,
        finish_reason=finish_reason,
        attempts=attempts,
        unmetered_attempts=unmetered_attempts,
    )


def parse_chunk_entries(
    content: str | Mapping[str, Any],
    usage: LLMUsage,
    *,
    topic_path: str,
    finish_reason: str = "stop",
    attempts: int = 1,
    unmetered_attempts: int = 0,
) -> ChunkExtraction:
    """Validate entries output; the result carries ``topic_path`` and an empty title and summary.

    Raises:
        LLMSchemaError: the output is not a JSON object matching the schema.
    """
    output = _validate(
        _EntriesOutput, "chunk entries", content, usage, attempts, unmetered_attempts
    )
    return ChunkExtraction(
        topic_path=topic_path,
        topic_title="",
        summary="",
        entries=tuple(_to_entry(entry) for entry in output.entries),
        usage=usage,
        finish_reason=finish_reason,
        attempts=attempts,
        unmetered_attempts=unmetered_attempts,
    )


def parse_triage(
    content: str | Mapping[str, Any],
    usage: LLMUsage,
    *,
    finish_reason: str = "stop",
    attempts: int = 1,
    unmetered_attempts: int = 0,
) -> TriageVerdict:
    """Validate triage output and return a TriageVerdict.

    Raises:
        LLMSchemaError: the output is not a JSON object matching the schema.
    """
    output = _validate(_TriageOutput, "triage", content, usage, attempts, unmetered_attempts)
    return TriageVerdict(
        worth_recording=output.worth_recording,
        usage=usage,
        finish_reason=finish_reason,
        attempts=attempts,
        unmetered_attempts=unmetered_attempts,
    )


def _validate[M: BaseModel](
    model: type[M],
    label: str,
    content: str | Mapping[str, Any],
    usage: LLMUsage,
    attempts: int,
    unmetered_attempts: int,
) -> M:
    decoded = _decode(label, content, usage, attempts, unmetered_attempts)
    errors: list[ErrorDetails] = []
    try:
        return model.model_validate(decoded)
    except ValidationError as exc:
        errors = exc.errors(include_url=False, include_input=False, include_context=False)
    # Raised outside the except block so no exception context (which
    # embeds the offending values) is chained onto the schema error.
    raise LLMSchemaError(
        _describe(label, errors),
        usage=usage,
        attempts=attempts,
        unmetered_attempts=unmetered_attempts,
    )


def _decode(
    label: str,
    content: str | Mapping[str, Any],
    usage: LLMUsage,
    attempts: int,
    unmetered_attempts: int,
) -> Mapping[str, Any]:
    if isinstance(content, Mapping):
        return content
    try:
        decoded = json.loads(content)
    except json.JSONDecodeError:
        decoded = None
    if not isinstance(decoded, dict):
        raise LLMSchemaError(
            f"{label} output is not a JSON object",
            usage=usage,
            attempts=attempts,
            unmetered_attempts=unmetered_attempts,
        )
    return decoded


def _to_entry(entry: _EntryOutput) -> ChunkEntry:
    return ChunkEntry(
        content=entry.content,
        why=entry.why,
        tags=tuple(entry.tags),
        event_date=entry.event_date,
        source_message_index=entry.source_message_index,
    )


def _describe(label: str, errors: list[ErrorDetails]) -> str:
    problems = sorted({f"{_location(error['loc'])}: {error['type']}" for error in errors})
    return f"{label} output failed schema validation: " + "; ".join(problems)


def _location(loc: tuple[int | str, ...]) -> str:
    # Unknown keys come from model output; only schema field names are echoed.
    parts = [
        str(part) if isinstance(part, int) or part in _KNOWN_FIELDS else _UNKNOWN_LOCATION
        for part in loc
    ]
    return ".".join(parts) or _UNKNOWN_LOCATION
