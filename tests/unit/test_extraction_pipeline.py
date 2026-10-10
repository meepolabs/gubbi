"""Behavior tests for the LLM-phase extraction pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Any

import pytest

from gubbi.extraction.chunking import chunk_messages, split_in_two
from gubbi.extraction.llm.provider import (
    LLMMessage,
    LLMProviderError,
    LLMRefusalError,
    LLMResponse,
    LLMSchemaError,
    LLMTransientError,
    LLMTruncatedError,
    LLMUsage,
)
from gubbi.extraction.pipeline import (
    ExtractedCandidate,
    PipelineDeadlineError,
    PipelineFailure,
    run_pipeline,
)
from gubbi.extraction.service import ExtractionService
from gubbi.telemetry.attrs import SpanNames

if TYPE_CHECKING:
    from collections.abc import Callable

    from tests.conftest import InMemoryExporter
    from tests.fixtures.log_capture import LogCapture

pytestmark = pytest.mark.unit

TODAY = date(2026, 6, 1)
CREATED = date(2026, 1, 5)
CALL_USAGE = LLMUsage(input=1000, output=200, cached_input=300, reasoning=50)
FAILED_USAGE = LLMUsage(input=900, output=16000, cached_input=0, reasoning=7)
TRANSCRIPT_TEXT_HEADING = "## Transcript\n"
PRIOR_HEADING = "## Entries already recorded from earlier parts of this conversation\n"
SECRET = "zz-private-transcript-text-zz"


@dataclass(frozen=True)
class _Msg:
    role: str
    content: str
    timestamp: str | None = None


Step = Mapping[str, Any] | Exception


class _ScriptedProvider:
    """Replays one scripted step per call; records every user message it receives."""

    def __init__(self, steps: list[Step]) -> None:
        self._steps = list(steps)
        self.user_contents: list[str] = []

    async def complete(
        self,
        messages: list[LLMMessage],
        system_prompt: str,
        output_schema: Mapping[str, Any] | None = None,
    ) -> LLMResponse:
        self.user_contents.append(messages[0]["content"])
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
            attempts=2,
        )

    def estimate_cost_cents(self, input_tokens: int, output_tokens: int) -> float:
        return 0.0

    def transcript_of(self, call: int) -> str:
        return self.user_contents[call].split(TRANSCRIPT_TEXT_HEADING, 1)[1]

    def prior_section_of(self, call: int) -> str:
        after = self.user_contents[call].split(PRIOR_HEADING, 1)[1]
        return after.split("\n\n## Transcript", 1)[0]


def _output(
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


def _entry(
    content: str, event_date: str | None = None, source: int | None = None
) -> dict[str, Any]:
    return {
        "content": content,
        "why": f"why {content}",
        "tags": ["t"],
        "event_date": event_date,
        "source_message_index": source,
    }


def _large_conversation() -> list[_Msg]:
    # 40 x 10k chars ~ 114k estimated tokens: several chunks at the default target.
    return [
        _Msg("user" if i % 2 == 0 else "assistant", f"m{i} " + "x" * 10_000, "2026-03-10T09:00:00")
        for i in range(40)
    ]


def _small_conversation() -> list[_Msg]:
    return [
        _Msg("user", f"We picked the vendor. {SECRET}", "2026-02-01T10:00:00"),
        _Msg("assistant", "Noted the vendor choice.", "2026-02-02T10:00:00"),
        _Msg("user", "And we set the launch for spring.", "2026-02-03T10:00:00"),
    ]


def _times(usage: LLMUsage, n: int) -> LLMUsage:
    return LLMUsage(
        input=usage.input * n,
        output=usage.output * n,
        cached_input=usage.cached_input * n,
        reasoning=usage.reasoning * n,
    )


def _plus(a: LLMUsage, b: LLMUsage) -> LLMUsage:
    return LLMUsage(
        input=a.input + b.input,
        output=a.output + b.output,
        cached_input=a.cached_input + b.cached_input,
        reasoning=a.reasoning + b.reasoning,
    )


async def _run(
    provider: _ScriptedProvider,
    messages: list[_Msg],
    *,
    deadline: float = 100.0,
    clock: Callable[[], float] = lambda: 0.0,
) -> Any:
    return await run_pipeline(
        ExtractionService(provider),
        messages,
        existing_topics=["work/launch"],
        conversation_created=CREATED,
        deadline=deadline,
        today=TODAY,
        clock=clock,
    )


async def _failure(provider: _ScriptedProvider, messages: list[_Msg], **kwargs: Any) -> Any:
    with pytest.raises(PipelineFailure) as caught:
        await _run(provider, messages, **kwargs)
    return caught.value


# -- Multi-chunk ---------------------------------------------------------------


async def test_large_input_makes_one_call_per_chunk_and_sums_usage() -> None:
    messages = _large_conversation()
    chunks = chunk_messages(messages)
    provider = _ScriptedProvider([_output() for _ in chunks])

    outcome = await _run(provider, messages)

    assert len(chunks) >= 3
    assert len(provider.user_contents) == len(chunks)
    assert [provider.transcript_of(i) for i in range(len(chunks))] == [c.text for c in chunks]
    assert outcome.llm_calls == len(chunks)
    assert outcome.chunks == len(chunks)
    assert outcome.usage == _times(CALL_USAGE, len(chunks))


async def test_entries_from_earlier_chunks_reach_every_later_chunk() -> None:
    messages = _large_conversation()
    chunk_count = len(chunk_messages(messages))
    steps = [
        _output(entries=[_entry(f"decision-{i}", "2026-03-0" + str(i + 1))])
        for i in range(chunk_count)
    ]
    provider = _ScriptedProvider(steps)

    await _run(provider, messages)

    assert provider.prior_section_of(0) == "(none)"
    for call in range(1, chunk_count):
        prior = provider.prior_section_of(call)
        for earlier in range(call):
            assert f"decision-{earlier}" in prior
        assert f"decision-{call}" not in prior


async def test_topic_comes_from_first_chunk_and_later_topics_are_ignored() -> None:
    messages = _large_conversation()
    chunk_count = len(chunk_messages(messages))
    steps = [_output(topic="work/launch", summary="First part.")] + [
        _output(topic=f"other/topic-{i}", summary=f"Part {i}.") for i in range(1, chunk_count)
    ]
    provider = _ScriptedProvider(steps)

    outcome = await _run(provider, messages)

    assert outcome.topic_path == "work/launch"
    assert outcome.topic_title == "Work/Launch"
    assert outcome.summary.startswith("First part.")


# -- Candidates ----------------------------------------------------------------


async def test_candidates_carry_entry_fields_and_resolved_event_dates() -> None:
    provider = _ScriptedProvider(
        [
            _output(
                entries=[
                    _entry("picked vendor", "2026-01-20", 0),
                    _entry("launch in spring", None, 2),
                    _entry("future claim", "2099-01-01", 1),
                ]
            )
        ]
    )

    outcome = await _run(provider, _small_conversation())

    assert outcome.candidates == (
        ExtractedCandidate("picked vendor", "why picked vendor", ("t",), date(2026, 1, 20), 0),
        ExtractedCandidate("launch in spring", "why launch in spring", ("t",), date(2026, 2, 3), 2),
        ExtractedCandidate("future claim", "why future claim", ("t",), date(2026, 2, 2), 1),
    )


# -- Truncation ----------------------------------------------------------------


async def test_one_truncation_splits_the_chunk_once_and_recovers() -> None:
    messages = _small_conversation()
    (chunk,) = chunk_messages(messages)
    first_half, second_half = split_in_two(chunk)
    provider = _ScriptedProvider(
        [
            LLMTruncatedError("cap", usage=FAILED_USAGE),
            _output(entries=[_entry("from first half", None, 0)]),
            _output(topic="ignored/topic", entries=[_entry("from second half", None, 2)]),
        ]
    )

    outcome = await _run(provider, messages)

    assert len(provider.user_contents) == 3
    assert provider.transcript_of(1) == first_half.text
    assert provider.transcript_of(2) == second_half.text
    assert "from first half" in provider.prior_section_of(2)
    assert [c.content for c in outcome.candidates] == ["from first half", "from second half"]
    assert outcome.topic_path == "work/launch"
    assert outcome.llm_calls == 3
    assert outcome.chunks == 2
    assert outcome.usage == _plus(_times(CALL_USAGE, 2), FAILED_USAGE)


def _truncated() -> LLMTruncatedError:
    return LLMTruncatedError("cap", usage=FAILED_USAGE)


@pytest.mark.parametrize(
    ("script", "expected_usage"),
    [
        pytest.param(
            [_truncated(), _truncated()],
            _times(FAILED_USAGE, 2),
            id="first-half-truncated-again",
        ),
        pytest.param(
            [_truncated(), _output(), _truncated()],
            _plus(_times(FAILED_USAGE, 2), CALL_USAGE),
            id="second-half-truncated-again",
        ),
    ],
)
async def test_second_truncation_fails_with_summed_usage(
    script: list[Step], expected_usage: LLMUsage
) -> None:
    provider = _ScriptedProvider(script)

    failure = await _failure(provider, _small_conversation())

    assert isinstance(failure.error, LLMTruncatedError)
    assert len(provider.user_contents) == len(script)
    assert failure.llm_calls == len(script)
    assert failure.usage_so_far == expected_usage


# -- Other failures --------------------------------------------------------------


@pytest.mark.parametrize(
    ("step", "error_type", "failed_call_usage"),
    [
        pytest.param(
            LLMRefusalError("refused", usage=FAILED_USAGE),
            LLMRefusalError,
            FAILED_USAGE,
            id="refusal",
        ),
        pytest.param({"topic_path": "x"}, LLMSchemaError, CALL_USAGE, id="schema"),
        pytest.param(LLMTransientError("down"), LLMTransientError, LLMUsage(), id="no-usage"),
    ],
)
async def test_non_truncation_failure_is_not_retried_and_counts_the_failed_call(
    step: Step, error_type: type[LLMProviderError], failed_call_usage: LLMUsage
) -> None:
    messages = _large_conversation()
    provider = _ScriptedProvider([_output(), step])

    failure = await _failure(provider, messages)

    assert isinstance(failure.error, error_type)
    assert len(provider.user_contents) == 2
    assert failure.llm_calls == 2
    assert failure.usage_so_far == _plus(CALL_USAGE, failed_call_usage)
    assert failure.chunks == len(chunk_messages(messages))


async def test_empty_transcript_fails_without_calling_the_model() -> None:
    provider = _ScriptedProvider([])

    failure = await _failure(provider, [])

    assert isinstance(failure.error, ValueError)
    assert provider.user_contents == []


# -- Deadline --------------------------------------------------------------------


async def test_deadline_passed_before_next_call_fails_without_making_it() -> None:
    messages = _large_conversation()
    readings = iter([0.0, 10.0])
    provider = _ScriptedProvider([_output(), _output()])

    failure = await _failure(provider, messages, deadline=10.0, clock=lambda: next(readings))

    assert isinstance(failure.error, PipelineDeadlineError)
    assert len(provider.user_contents) == 1
    assert failure.llm_calls == 1
    assert failure.usage_so_far == CALL_USAGE


async def test_deadline_already_passed_makes_no_call() -> None:
    provider = _ScriptedProvider([_output()])

    failure = await _failure(provider, _small_conversation(), deadline=5.0, clock=lambda: 5.0)

    assert isinstance(failure.error, PipelineDeadlineError)
    assert provider.user_contents == []
    assert failure.usage_so_far == LLMUsage()


# -- Telemetry -------------------------------------------------------------------

EXPECTED_SPAN_KEYS = {
    "prompt_size",
    "completion_size",
    "cached_prompt_size",
    "reasoning_size",
    "finish_reason",
    "attempts",
    "chunk_index",
    "latency_ms",
}


def _llm_spans(exporter: InMemoryExporter) -> list[dict[str, Any]]:
    return [
        dict(span.attributes or {})
        for span in exporter.spans
        if span.name == SpanNames.EXTRACTION_LLM_CALL
    ]


async def test_successful_call_span_keeps_every_attribute_through_the_allowlist(
    in_memory_tracer: tuple[Any, InMemoryExporter],
) -> None:
    _, exporter = in_memory_tracer
    provider = _ScriptedProvider([_output()])

    await _run(provider, _small_conversation())

    (attrs,) = _llm_spans(exporter)
    assert set(attrs) == EXPECTED_SPAN_KEYS
    assert not [key for key in attrs if "token" in key.lower()]
    assert attrs["prompt_size"] == CALL_USAGE.input
    assert attrs["completion_size"] == CALL_USAGE.output
    assert attrs["cached_prompt_size"] == CALL_USAGE.cached_input
    assert attrs["reasoning_size"] == CALL_USAGE.reasoning
    assert attrs["finish_reason"] == "stop"
    assert attrs["attempts"] == 2
    assert attrs["chunk_index"] == 0


async def test_truncated_call_span_records_length_and_the_failed_call_usage(
    in_memory_tracer: tuple[Any, InMemoryExporter],
) -> None:
    _, exporter = in_memory_tracer
    provider = _ScriptedProvider(
        [LLMTruncatedError("cap", usage=FAILED_USAGE), _output(), _output()]
    )

    await _run(provider, _small_conversation())

    truncated, *halves = _llm_spans(exporter)
    assert truncated["finish_reason"] == "length"
    assert truncated["completion_size"] == FAILED_USAGE.output
    assert [half["chunk_index"] for half in halves] == [0, 0]


async def test_call_log_lines_carry_sizes_but_no_transcript_text(log_capture: LogCapture) -> None:
    provider = _ScriptedProvider([LLMRefusalError("refused", usage=FAILED_USAGE)])

    await _failure(provider, _small_conversation())
    provider = _ScriptedProvider([_output(summary=SECRET, entries=[_entry(SECRET)])])
    await _run(provider, _small_conversation())

    assert log_capture.text.count("extraction_llm_call") == 2
    assert '"prompt_size": 900' in log_capture.text
    assert SECRET not in log_capture.text
