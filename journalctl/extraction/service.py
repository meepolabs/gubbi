import json
import pathlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from journalctl.extraction.llm.provider import LLMMessage, LLMProvider


@dataclass
class CategorizationResult:
    topic_path: str
    topic_title: str
    summary: str
    confidence: float


@dataclass
class ExtractedEntry:
    content: str
    reasoning: str
    tags: list[str]
    entry_date: str | None = None


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
        return CategorizationResult(
            topic_path=parsed["topic_path"],
            topic_title=parsed["topic_title"],
            summary=parsed["summary"],
            confidence=float(parsed["confidence"]),
        )

    async def extract_entries(
        self,
        messages: list[LLMMessage],
        topic: str,
    ) -> list[ExtractedEntry]:
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
        entries = parsed.get("entries", [])
        return [
            ExtractedEntry(
                content=e["content"],
                reasoning=e["reasoning"],
                tags=e["tags"],
                entry_date=e.get("entry_date"),
            )
            for e in entries
        ]

    @staticmethod
    def _parse_content(content: str | dict[str, Any]) -> dict[str, Any]:
        if isinstance(content, dict):
            return content
        try:
            return cast(dict[str, Any], json.loads(content))
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM response content is not valid JSON: {content!r}") from e

    def _read_prompt(self, filename: str) -> str:
        path = self._prompts_dir / filename
        return path.read_text(encoding="utf-8")
