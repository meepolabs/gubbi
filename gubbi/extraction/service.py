import json
import pathlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from gubbi.extraction.llm.provider import LLMMessage, LLMProvider


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

    @staticmethod
    def _parse_content(content: str | dict[str, Any]) -> dict[str, Any]:
        if isinstance(content, dict):
            return content
        try:
            return cast(dict[str, Any], json.loads(content))
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
