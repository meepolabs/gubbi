import json
import pathlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from gubbi.extraction.chunk_extraction import (
    CHUNK_CATEGORIZE_SCHEMA,
    CHUNK_ENTRIES_SCHEMA,
    CHUNK_EXTRACTION_SCHEMA,
    TRIAGE_SCHEMA,
    ChunkExtraction,
    TriageVerdict,
    parse_chunk_categorization,
    parse_chunk_entries,
    parse_chunk_extraction,
    parse_triage,
)
from gubbi.extraction.llm.provider import LLMMessage, LLMProvider, LLMResponse, LLMUsage


@dataclass
class CategorizationResult:
    topic_path: str
    topic_title: str
    summary: str
    confidence: float
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class ExtractedEntry:
    content: str
    reasoning: str
    tags: list[str]
    entry_date: str | None = None


@dataclass(frozen=True)
class ExtractionEntriesResult:
    """Wrapper around a list of ExtractedEntry that also carries token usage."""

    entries: tuple[ExtractedEntry, ...]
    input_tokens: int
    output_tokens: int


_CATEGORIZE_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "properties": {
        "topic_path": {"type": "string"},
        "topic_title": {"type": "string"},
        "summary": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["topic_path", "topic_title", "summary", "confidence"],
}

_CATEGORIZE_REQUIRED_FIELDS: frozenset[str] = frozenset(
    {"topic_path", "topic_title", "summary", "confidence"}
)


class ExtractionError(Exception):
    """Raised when an LLM extraction response fails contract validation."""


_EXTRACT_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "properties": {
        "entries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "reasoning": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "entry_date": {"type": "string"},
                },
                "required": ["content", "reasoning", "tags"],
            },
        },
    },
    "required": ["entries"],
}


class ExtractionService:
    def __init__(self, llm_provider: LLMProvider) -> None:
        self._llm = llm_provider
        self._prompts_dir = pathlib.Path(__file__).resolve().parent / "prompts"

    async def categorize_conversation(
        self,
        messages: list[LLMMessage],
        existing_topics: list[str],
    ) -> CategorizationResult:
        """Pick or coin a topic path for the conversation; return path + summary + confidence.

        The returned CategorizationResult includes input_tokens and output_tokens
        from the underlying LLM call so callers can accumulate cost.
        """
        system_prompt = self._read_prompt("categorize.md")
        user_content = json.dumps(
            {
                "messages": messages,
                "existing_topics": existing_topics,
            }
        )
        user_msg: LLMMessage = {"role": "user", "content": user_content}
        response = await self._llm.complete([user_msg], system_prompt, _CATEGORIZE_SCHEMA)

        parsed = self._parse_content(response.content)
        missing = _CATEGORIZE_REQUIRED_FIELDS - parsed.keys()
        if missing:
            raise ExtractionError(f"categorize response missing required fields: {sorted(missing)}")
        return CategorizationResult(
            topic_path=parsed["topic_path"],
            topic_title=parsed["topic_title"],
            summary=parsed["summary"],
            confidence=float(parsed["confidence"]),
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )

    async def extract_entries(
        self,
        messages: list[LLMMessage],
        topic: str,
    ) -> ExtractionEntriesResult:
        """Mine the conversation for journal-worthy entries (decisions, milestones, facts).

        Returns an ExtractionEntriesResult that bundles the entry list with the
        token counts from the LLM call so callers can accumulate cost.
        """
        system_prompt = self._read_prompt("extract_entries.md")
        user_content = json.dumps(
            {
                "messages": messages,
                "topic": topic,
            }
        )
        user_msg: LLMMessage = {"role": "user", "content": user_content}
        response = await self._llm.complete([user_msg], system_prompt, _EXTRACT_SCHEMA)

        parsed = self._parse_content(response.content)
        raw_entries = parsed.get("entries", [])
        entries = [
            ExtractedEntry(
                content=e["content"],
                reasoning=e["reasoning"],
                tags=e["tags"],
                entry_date=e.get("entry_date"),
            )
            for e in raw_entries
        ]
        return ExtractionEntriesResult(
            entries=tuple(entries),
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )

    async def extract_chunk(
        self,
        chunk_text: str,
        existing_topics: list[str],
        prior_entries_text: str,
    ) -> ChunkExtraction:
        """Pick a topic and extract significant entries from one transcript chunk.

        ``chunk_text`` is the serialized chunk (``[index] [date] role: text``
        lines); ``prior_entries_text`` lists entries already extracted from
        earlier chunks so the model does not repeat them. An empty
        ``entries`` tuple means the chunk holds nothing worth recording.

        Raises:
            LLMSchemaError: the model output does not match the chunk schema.
        """
        system_prompt = self._read_prompt("extract_single.md")
        user_msg: LLMMessage = {
            "role": "user",
            "content": _chunk_user_content(chunk_text, existing_topics, prior_entries_text),
        }
        response = await self._llm.complete([user_msg], system_prompt, CHUNK_EXTRACTION_SCHEMA)
        return parse_chunk_extraction(
            response.content,
            _usage_of(response),
            finish_reason=response.finish_reason,
            attempts=response.attempts,
        )

    async def categorize_chunk(
        self, chunk_text: str, existing_topics: list[str]
    ) -> ChunkExtraction:
        """Pick a topic and summarize the conversation from one transcript chunk.

        The first call of the two-call extraction; the result carries no entries.

        Raises:
            LLMSchemaError: the model output does not match the categorize schema.
        """
        system_prompt = self._read_prompt("categorize_chunk.md")
        user_msg: LLMMessage = {
            "role": "user",
            "content": _categorize_user_content(chunk_text, existing_topics),
        }
        response = await self._llm.complete([user_msg], system_prompt, CHUNK_CATEGORIZE_SCHEMA)
        return parse_chunk_categorization(
            response.content,
            _usage_of(response),
            finish_reason=response.finish_reason,
            attempts=response.attempts,
        )

    async def extract_chunk_entries(
        self,
        chunk_text: str,
        topic_path: str,
        prior_entries_text: str,
    ) -> ChunkExtraction:
        """Extract significant entries from one transcript chunk under a known topic.

        The per-chunk call of the two-call extraction; the result carries
        ``topic_path`` and an empty title and summary. Inputs are as for
        :meth:`extract_chunk`.

        Raises:
            LLMSchemaError: the model output does not match the entries schema.
        """
        system_prompt = self._read_prompt("extract_chunk_entries.md")
        user_msg: LLMMessage = {
            "role": "user",
            "content": _entries_user_content(chunk_text, topic_path, prior_entries_text),
        }
        response = await self._llm.complete([user_msg], system_prompt, CHUNK_ENTRIES_SCHEMA)
        return parse_chunk_entries(
            response.content,
            _usage_of(response),
            topic_path=topic_path,
            finish_reason=response.finish_reason,
            attempts=response.attempts,
        )

    async def triage(self, chunk_text: str) -> TriageVerdict:
        """Ask whether one transcript chunk holds anything worth recording.

        Raises:
            LLMSchemaError: the model output does not match the triage schema.
        """
        system_prompt = self._read_prompt("triage.md")
        user_msg: LLMMessage = {"role": "user", "content": f"{_TRANSCRIPT_HEADING}{chunk_text}"}
        response = await self._llm.complete([user_msg], system_prompt, TRIAGE_SCHEMA)
        return parse_triage(
            response.content,
            _usage_of(response),
            finish_reason=response.finish_reason,
            attempts=response.attempts,
        )

    @staticmethod
    def _parse_content(content: str | dict[str, Any]) -> dict[str, Any]:
        if isinstance(content, dict):
            return content
        try:
            return cast("dict[str, Any]", json.loads(content))
        except json.JSONDecodeError as e:
            # Do NOT interpolate the raw content into the exception
            # message: this exception's str() flows through OTel's
            # ``record_exception`` (as ``exception.message`` span event)
            # and into HyperDX, leaking the unparsed LLM response.
            # Operators who need the raw content for triage should fetch
            # it from the per-job audit row or the structured log line
            # captured at the call site.
            raise ValueError("LLM response content is not valid JSON") from e

    def _read_prompt(self, filename: str) -> str:
        path = self._prompts_dir / filename
        return path.read_text(encoding="utf-8")


_NONE_LABEL = "(none)"
_TRANSCRIPT_HEADING = "## Transcript\n"


def _usage_of(response: LLMResponse) -> LLMUsage:
    return LLMUsage(
        input=response.input_tokens,
        output=response.output_tokens,
        cached_input=response.cached_input_tokens,
        reasoning=response.reasoning_tokens,
    )


def _topics_section(existing_topics: list[str]) -> str:
    topics = "\n".join(f"- {topic}" for topic in existing_topics) or _NONE_LABEL
    return f"## Existing topics\n{topics}\n\n"


def _prior_section(prior_entries_text: str) -> str:
    prior = prior_entries_text or _NONE_LABEL
    return f"## Entries already recorded from earlier parts of this conversation\n{prior}\n\n"


def _chunk_user_content(
    chunk_text: str, existing_topics: list[str], prior_entries_text: str
) -> str:
    return (
        _topics_section(existing_topics)
        + _prior_section(prior_entries_text)
        + f"{_TRANSCRIPT_HEADING}{chunk_text}"
    )


def _categorize_user_content(chunk_text: str, existing_topics: list[str]) -> str:
    return _topics_section(existing_topics) + f"{_TRANSCRIPT_HEADING}{chunk_text}"


def _entries_user_content(chunk_text: str, topic_path: str, prior_entries_text: str) -> str:
    return (
        f"## Topic\n{topic_path}\n\n"
        + _prior_section(prior_entries_text)
        + f"{_TRANSCRIPT_HEADING}{chunk_text}"
    )
