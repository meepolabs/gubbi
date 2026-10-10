"""Structured output of one single-call extraction over a transcript chunk.

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
    "CHUNK_EXTRACTION_SCHEMA",
    "ChunkEntry",
    "ChunkExtraction",
    "parse_chunk_extraction",
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


_KNOWN_FIELDS: Final[frozenset[str]] = frozenset(_ChunkOutput.model_fields) | frozenset(
    _EntryOutput.model_fields
)


def parse_chunk_extraction(
    content: str | Mapping[str, Any],
    usage: LLMUsage,
    *,
    finish_reason: str = "stop",
    attempts: int = 1,
) -> ChunkExtraction:
    """Validate model output against the chunk schema and return a ChunkExtraction.

    Raises:
        LLMSchemaError: the output is not a JSON object matching the schema.
    """
    decoded = _decode(content, usage)
    errors: list[ErrorDetails] = []
    try:
        output = _ChunkOutput.model_validate(decoded)
    except ValidationError as exc:
        errors = exc.errors(include_url=False, include_input=False, include_context=False)
    if errors:
        # Raised outside the except block so no exception context (which
        # embeds the offending values) is chained onto the schema error.
        raise LLMSchemaError(_describe(errors), usage=usage)
    return ChunkExtraction(
        topic_path=output.topic_path,
        topic_title=output.topic_title,
        summary=output.summary,
        entries=tuple(_to_entry(entry) for entry in output.entries),
        usage=usage,
        finish_reason=finish_reason,
        attempts=attempts,
    )


def _decode(content: str | Mapping[str, Any], usage: LLMUsage) -> Mapping[str, Any]:
    if isinstance(content, Mapping):
        return content
    try:
        decoded = json.loads(content)
    except json.JSONDecodeError:
        decoded = None
    if not isinstance(decoded, dict):
        raise LLMSchemaError("chunk extraction output is not a JSON object", usage=usage)
    return decoded


def _to_entry(entry: _EntryOutput) -> ChunkEntry:
    return ChunkEntry(
        content=entry.content,
        why=entry.why,
        tags=tuple(entry.tags),
        event_date=entry.event_date,
        source_message_index=entry.source_message_index,
    )


def _describe(errors: list[ErrorDetails]) -> str:
    problems = sorted({f"{_location(error['loc'])}: {error['type']}" for error in errors})
    return "chunk extraction output failed schema validation: " + "; ".join(problems)


def _location(loc: tuple[int | str, ...]) -> str:
    # Unknown keys come from model output; only schema field names are echoed.
    parts = [
        str(part) if isinstance(part, int) or part in _KNOWN_FIELDS else _UNKNOWN_LOCATION
        for part in loc
    ]
    return ".".join(parts) or _UNKNOWN_LOCATION
