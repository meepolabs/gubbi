"""Scripted provider, transcripts and a runner for ``run_pipeline`` unit tests.

The provider replays one scripted step per call and records every user
message, schema and time budget it receives, so tests can read the
transcript and prior-entries sections each call was sent.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any

import pytest

from gubbi.extraction.llm.provider import LLMMessage, LLMResponse, LLMUsage
from gubbi.extraction.pipeline import PipelineFailure, run_pipeline
from gubbi.extraction.service import ExtractionService

TODAY = date(2026, 6, 1)
CREATED = date(2026, 1, 5)
CALL_USAGE = LLMUsage(input=1000, output=200, cached_input=300, reasoning=50)
FAILED_USAGE = LLMUsage(input=900, output=16000, cached_input=0, reasoning=7)
TRANSCRIPT_TEXT_HEADING = "## Transcript\n"
PRIOR_HEADING = "## Entries already recorded from earlier parts of this conversation\n"
SECRET = "zz-private-transcript-text-zz"


@dataclass(frozen=True)
class Msg:
    role: str
    content: str
    timestamp: str | None = None


Step = Mapping[str, Any] | Exception


class PipelineProvider:
    """Replays one scripted step per call; records every user message it receives."""

    def __init__(
        self, steps: list[Step], *, attempts: int = 1, unmetered_attempts: int = 0
    ) -> None:
        self._steps = list(steps)
        self._attempts = attempts
        self._unmetered_attempts = unmetered_attempts
        self.user_contents: list[str] = []
        self.schemas: list[Mapping[str, Any] | None] = []
        self.time_budgets: list[float | None] = []

    async def complete(
        self,
        messages: list[LLMMessage],
        system_prompt: str,
        output_schema: Mapping[str, Any] | None = None,
        *,
        time_budget: float | None = None,
    ) -> LLMResponse:
        self.user_contents.append(messages[0]["content"])
        self.schemas.append(output_schema)
        self.time_budgets.append(time_budget)
        step = self._steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return LLMResponse(
            content=dict(step),
            input_tokens=CALL_USAGE.input,
            output_tokens=CALL_USAGE.output,
            model="scripted",
            cached_input_tokens=CALL_USAGE.cached_input,
            reasoning_tokens=CALL_USAGE.reasoning,
            attempts=self._attempts,
            unmetered_attempts=self._unmetered_attempts,
        )

    def estimate_cost_cents(self, input_tokens: int, output_tokens: int) -> float:
        return 0.0

    def transcript_of(self, call: int) -> str:
        return self.user_contents[call].split(TRANSCRIPT_TEXT_HEADING, 1)[1]

    def prior_section_of(self, call: int) -> str:
        after = self.user_contents[call].split(PRIOR_HEADING, 1)[1]
        return after.split("\n\n## Transcript", 1)[0]


def output(
    topic: str = "work/launch",
    summary: str = "Planned the launch.",
    entries: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "topic_path": topic,
        "topic_title": topic.title(),
        "summary": summary,
        "entries": entries or [],
    }


def entry(content: str, event_date: str | None = None, source: int | None = None) -> dict[str, Any]:
    return {
        "content": content,
        "why": f"why {content}",
        "tags": ["t"],
        "event_date": event_date,
        "source_message_index": source,
    }


def large_conversation() -> list[Msg]:
    # 40 x 10k chars ~ 114k estimated tokens: several chunks at the default target.
    return [
        Msg("user" if i % 2 == 0 else "assistant", f"m{i} " + "x" * 10_000, "2026-03-10T09:00:00")
        for i in range(40)
    ]


def small_conversation() -> list[Msg]:
    return [
        Msg("user", f"We picked the vendor. {SECRET}", "2026-02-01T10:00:00"),
        Msg("assistant", "Noted the vendor choice.", "2026-02-02T10:00:00"),
        Msg("user", "And we set the launch for spring.", "2026-02-03T10:00:00"),
    ]


def times(usage: LLMUsage, n: int) -> LLMUsage:
    return LLMUsage(
        input=usage.input * n,
        output=usage.output * n,
        cached_input=usage.cached_input * n,
        reasoning=usage.reasoning * n,
    )


def plus(a: LLMUsage, b: LLMUsage) -> LLMUsage:
    return LLMUsage(
        input=a.input + b.input,
        output=a.output + b.output,
        cached_input=a.cached_input + b.cached_input,
        reasoning=a.reasoning + b.reasoning,
    )


async def run(
    provider: PipelineProvider,
    messages: list[Msg],
    *,
    deadline: float = 100.0,
    clock: Callable[[], float] = lambda: 0.0,
    **modes: Any,
) -> Any:
    return await run_pipeline(
        ExtractionService(provider),
        messages,
        existing_topics=["work/launch"],
        conversation_created=CREATED,
        deadline=deadline,
        today=TODAY,
        clock=clock,
        **modes,
    )


async def expect_failure(provider: PipelineProvider, messages: list[Msg], **kwargs: Any) -> Any:
    with pytest.raises(PipelineFailure) as caught:
        await run(provider, messages, **kwargs)
    return caught.value


def category_output(
    topic: str = "work/launch", summary: str = "Planned the launch."
) -> dict[str, Any]:
    return {"topic_path": topic, "topic_title": topic.title(), "summary": summary}


def entries_output(*entries: dict[str, Any]) -> dict[str, Any]:
    return {"entries": list(entries)}
