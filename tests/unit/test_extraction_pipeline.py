"""Behavior tests for the LLM-phase extraction pipeline."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any

import pytest

from gubbi.extraction.chunk_extraction import (
    CHUNK_CATEGORIZE_SCHEMA,
    CHUNK_ENTRIES_SCHEMA,
    CHUNK_EXTRACTION_SCHEMA,
    TRIAGE_SCHEMA,
)
from gubbi.extraction.chunking import chunk_messages, split_in_two
from gubbi.extraction.llm.provider import (
    LLMDeadlineError,
    LLMProviderError,
    LLMRefusalError,
    LLMSchemaError,
    LLMTransientError,
    LLMTruncatedError,
    LLMUsage,
)
from gubbi.extraction.pipeline import (
    ExtractedCandidate,
    PipelineDeadlineError,
    PipelineFailure,
    PipelineSpend,
)
from gubbi.telemetry.attrs import SpanNames
from tests.fixtures.extraction_pipeline import (
    CALL_USAGE,
    FAILED_USAGE,
    SECRET,
    PipelineProvider,
    Step,
    category_output,
    entries_output,
    entry,
    expect_failure,
    large_conversation,
    output,
    plus,
    run,
    small_conversation,
    times,
)

if TYPE_CHECKING:
    from tests.conftest import InMemoryExporter
    from tests.fixtures.log_capture import LogCapture

pytestmark = pytest.mark.unit

# -- Multi-chunk ---------------------------------------------------------------


async def test_large_input_makes_one_call_per_chunk_and_sums_usage() -> None:
    messages = large_conversation()
    chunks = chunk_messages(messages)
    provider = PipelineProvider([output() for _ in chunks])

    outcome = await run(provider, messages)

    assert len(chunks) >= 3
    assert len(provider.user_contents) == len(chunks)
    assert [provider.transcript_of(i) for i in range(len(chunks))] == [c.text for c in chunks]
    assert outcome.llm_calls == len(chunks)
    assert outcome.chunks == len(chunks)
    assert outcome.usage == times(CALL_USAGE, len(chunks))


async def test_entries_from_earlier_chunks_reach_every_later_chunk() -> None:
    messages = large_conversation()
    chunk_count = len(chunk_messages(messages))
    steps = [
        output(entries=[entry(f"decision-{i}", "2026-03-0" + str(i + 1))])
        for i in range(chunk_count)
    ]
    provider = PipelineProvider(steps)

    await run(provider, messages)

    assert provider.prior_section_of(0) == "(none)"
    for call in range(1, chunk_count):
        prior = provider.prior_section_of(call)
        for earlier in range(call):
            assert f"decision-{earlier}" in prior
        assert f"decision-{call}" not in prior


async def test_topic_comes_from_first_chunk_and_later_topics_are_ignored() -> None:
    messages = large_conversation()
    chunk_count = len(chunk_messages(messages))
    steps = [output(topic="work/launch", summary="First part.")] + [
        output(topic=f"other/topic-{i}", summary=f"Part {i}.") for i in range(1, chunk_count)
    ]
    provider = PipelineProvider(steps)

    outcome = await run(provider, messages)

    assert outcome.topic_path == "work/launch"
    assert outcome.topic_title == "Work/Launch"
    assert outcome.summary.startswith("First part.")


# -- Candidates ----------------------------------------------------------------


async def test_candidates_carry_entry_fields_and_resolved_event_dates() -> None:
    provider = PipelineProvider(
        [
            output(
                entries=[
                    entry("picked vendor", "2026-01-20", 0),
                    entry("launch in spring", None, 2),
                    entry("future claim", "2099-01-01", 1),
                ]
            )
        ]
    )

    outcome = await run(provider, small_conversation())

    assert outcome.candidates == (
        ExtractedCandidate("picked vendor", "why picked vendor", ("t",), date(2026, 1, 20), 0),
        ExtractedCandidate("launch in spring", "why launch in spring", ("t",), date(2026, 2, 3), 2),
        ExtractedCandidate("future claim", "why future claim", ("t",), date(2026, 2, 2), 1),
    )


# -- Truncation ----------------------------------------------------------------


async def test_one_truncation_splits_the_chunk_once_and_recovers() -> None:
    messages = small_conversation()
    (chunk,) = chunk_messages(messages)
    first_half, second_half = split_in_two(chunk)
    provider = PipelineProvider(
        [
            LLMTruncatedError("cap", usage=FAILED_USAGE),
            output(entries=[entry("from first half", None, 0)]),
            output(topic="ignored/topic", entries=[entry("from second half", None, 2)]),
        ]
    )

    outcome = await run(provider, messages)

    assert len(provider.user_contents) == 3
    assert provider.transcript_of(1) == first_half.text
    assert provider.transcript_of(2) == second_half.text
    assert "from first half" in provider.prior_section_of(2)
    assert [c.content for c in outcome.candidates] == ["from first half", "from second half"]
    assert outcome.topic_path == "work/launch"
    assert outcome.llm_calls == 3
    assert outcome.chunks == 2
    assert outcome.usage == plus(times(CALL_USAGE, 2), FAILED_USAGE)


def _truncated() -> LLMTruncatedError:
    return LLMTruncatedError("cap", usage=FAILED_USAGE)


@pytest.mark.parametrize(
    ("script", "expected_usage"),
    [
        pytest.param(
            [_truncated(), _truncated()],
            times(FAILED_USAGE, 2),
            id="first-half-truncated-again",
        ),
        pytest.param(
            [_truncated(), output(), _truncated()],
            plus(times(FAILED_USAGE, 2), CALL_USAGE),
            id="second-half-truncated-again",
        ),
    ],
)
async def test_second_truncation_fails_with_summed_usage(
    script: list[Step], expected_usage: LLMUsage
) -> None:
    provider = PipelineProvider(script)

    failure = await expect_failure(provider, small_conversation())

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
    messages = large_conversation()
    provider = PipelineProvider([output(), step])

    failure = await expect_failure(provider, messages)

    assert isinstance(failure.error, error_type)
    assert len(provider.user_contents) == 2
    assert failure.llm_calls == 2
    assert failure.usage_so_far == plus(CALL_USAGE, failed_call_usage)
    assert failure.chunks == len(chunk_messages(messages))


async def test_llm_calls_count_provider_retries_on_success() -> None:
    provider = PipelineProvider([output()], attempts=3)

    outcome = await run(provider, small_conversation())

    assert outcome.llm_calls == 3


@pytest.mark.parametrize(
    ("error", "expected_calls"),
    [
        pytest.param(LLMTransientError("down", attempts=3), 3, id="exhausted-retries"),
        pytest.param(LLMTransientError("down"), 1, id="attempts-unknown"),
    ],
)
async def test_llm_calls_count_the_failed_calls_requests(
    error: LLMProviderError, expected_calls: int
) -> None:
    provider = PipelineProvider([error])

    failure = await expect_failure(provider, small_conversation())

    assert failure.llm_calls == expected_calls


async def test_schema_failure_counts_the_responses_requests() -> None:
    provider = PipelineProvider([{"topic_path": "x"}], attempts=2)

    failure = await expect_failure(provider, small_conversation())

    assert isinstance(failure.error, LLMSchemaError)
    assert failure.llm_calls == 2


async def test_running_spend_is_reported_after_every_call_failed_ones_included() -> None:
    messages = large_conversation()
    chunk_count = len(chunk_messages(messages))
    timed_out = LLMTransientError("down", usage=FAILED_USAGE, attempts=3, unmetered_attempts=2)
    provider = PipelineProvider([output(), timed_out])
    reports: list[PipelineSpend] = []

    await expect_failure(provider, messages, on_spend=reports.append)

    assert reports == [
        PipelineSpend(usage=LLMUsage(), llm_calls=0, chunks=chunk_count, unmetered_calls=0),
        PipelineSpend(usage=CALL_USAGE, llm_calls=1, chunks=chunk_count, unmetered_calls=0),
        PipelineSpend(
            usage=plus(CALL_USAGE, FAILED_USAGE),
            llm_calls=4,
            chunks=chunk_count,
            unmetered_calls=2,
        ),
    ]


async def test_a_truncation_split_reports_the_extra_chunk_before_the_next_call() -> None:
    messages = small_conversation()
    truncated = LLMTruncatedError("cap", usage=FAILED_USAGE)
    provider = PipelineProvider([truncated, output(), output()])
    reports: list[PipelineSpend] = []
    clock_readings = iter([0.0, 200.0])

    await expect_failure(
        provider, messages, on_spend=reports.append, clock=lambda: next(clock_readings)
    )

    assert reports[-1] == PipelineSpend(
        usage=FAILED_USAGE, llm_calls=1, chunks=2, unmetered_calls=0
    )


@pytest.mark.parametrize(
    ("step", "expected_unmetered"),
    [
        pytest.param(output(), 1, id="success-after-a-timed-out-attempt"),
        pytest.param(
            LLMDeadlineError("budget spent", attempts=2, unmetered_attempts=2),
            2,
            id="deadline-after-timed-out-attempts",
        ),
    ],
)
async def test_unmetered_attempts_are_counted_on_the_outcome_and_the_failure(
    step: Step, expected_unmetered: int
) -> None:
    provider = PipelineProvider([step], unmetered_attempts=1)

    try:
        result: Any = await run(provider, small_conversation())
    except PipelineFailure as failure:
        result = failure

    assert result.unmetered_calls == expected_unmetered


async def test_empty_transcript_fails_without_calling_the_model() -> None:
    provider = PipelineProvider([])

    failure = await expect_failure(provider, [])

    assert isinstance(failure.error, ValueError)
    assert provider.user_contents == []


# -- Deadline --------------------------------------------------------------------


async def test_deadline_passed_before_next_call_fails_without_making_it() -> None:
    messages = large_conversation()
    readings = iter([0.0, 10.0])
    provider = PipelineProvider([output(), output()])

    failure = await expect_failure(provider, messages, deadline=10.0, clock=lambda: next(readings))

    assert isinstance(failure.error, PipelineDeadlineError)
    assert len(provider.user_contents) == 1
    assert failure.llm_calls == 1
    assert failure.usage_so_far == CALL_USAGE


async def test_deadline_already_passed_makes_no_call() -> None:
    provider = PipelineProvider([output()])

    failure = await expect_failure(provider, small_conversation(), deadline=5.0, clock=lambda: 5.0)

    assert isinstance(failure.error, PipelineDeadlineError)
    assert provider.user_contents == []
    assert failure.usage_so_far == LLMUsage()


async def test_each_call_gets_the_time_left_before_the_deadline_as_its_budget() -> None:
    messages = large_conversation()
    chunks = chunk_messages(messages)
    readings = iter(float(10 * i) for i in range(len(chunks)))
    provider = PipelineProvider([output() for _ in chunks])

    await run(provider, messages, deadline=1000.0, clock=lambda: next(readings))

    assert provider.time_budgets == [1000.0 - 10 * i for i in range(len(chunks))]


async def test_provider_running_out_of_budget_fails_as_deadline_with_usage() -> None:
    provider = PipelineProvider([LLMDeadlineError("budget spent", usage=FAILED_USAGE, attempts=2)])

    failure = await expect_failure(provider, small_conversation())

    assert isinstance(failure.error, PipelineDeadlineError)
    assert failure.usage_so_far == FAILED_USAGE
    assert failure.llm_calls == 2


# -- Telemetry -------------------------------------------------------------------

EXPECTED_SPAN_KEYS = {
    "provider_name",
    "model_name",
    "prompt_size",
    "completion_size",
    "cached_prompt_size",
    "reasoning_size",
    "finish_reason",
    "attempts",
    "chunk_index",
    "stage",
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
    provider = PipelineProvider([output()], attempts=2)

    await run(provider, small_conversation())

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
    assert attrs["provider_name"] == "PipelineProvider"
    assert attrs["model_name"] == "unknown"


async def test_truncated_call_span_records_length_and_the_failed_call_usage(
    in_memory_tracer: tuple[Any, InMemoryExporter],
) -> None:
    _, exporter = in_memory_tracer
    provider = PipelineProvider([LLMTruncatedError("cap", usage=FAILED_USAGE), output(), output()])

    await run(provider, small_conversation())

    truncated, *halves = _llm_spans(exporter)
    assert truncated["finish_reason"] == "length"
    assert truncated["completion_size"] == FAILED_USAGE.output
    assert [half["chunk_index"] for half in halves] == [0, 0]


async def test_call_log_lines_carry_sizes_but_no_transcript_text(log_capture: LogCapture) -> None:
    provider = PipelineProvider([LLMRefusalError("refused", usage=FAILED_USAGE)])

    await expect_failure(provider, small_conversation())
    provider = PipelineProvider([output(summary=SECRET, entries=[entry(SECRET)])])
    await run(provider, small_conversation())

    assert log_capture.text.count("extraction_llm_call") == 2
    assert '"prompt_size": 900' in log_capture.text
    assert SECRET not in log_capture.text


# -- Default mode ----------------------------------------------------------------


async def test_default_arguments_make_only_single_call_extractions() -> None:
    messages = large_conversation()
    chunk_count = len(chunk_messages(messages))
    provider = PipelineProvider([output() for _ in range(chunk_count)])

    await run(provider, messages)

    assert provider.schemas == [CHUNK_EXTRACTION_SCHEMA] * chunk_count


# -- Two-call mode ---------------------------------------------------------------

TOPIC_HEADING = "## Topic\n"


async def test_two_call_categorizes_once_then_extracts_every_chunk_under_that_topic() -> None:
    messages = large_conversation()
    chunks = chunk_messages(messages)
    steps: list[Step] = [category_output()] + [
        entries_output(entry(f"decision-{i}")) for i in range(len(chunks))
    ]
    provider = PipelineProvider(steps)

    outcome = await run(provider, messages, mode="two_call")

    assert provider.schemas == [CHUNK_CATEGORIZE_SCHEMA] + [CHUNK_ENTRIES_SCHEMA] * len(chunks)
    assert provider.transcript_of(0) == chunks[0].text
    assert [provider.transcript_of(i + 1) for i in range(len(chunks))] == [c.text for c in chunks]
    for call in range(1, len(chunks) + 1):
        assert f"{TOPIC_HEADING}work/launch\n" in provider.user_contents[call]
    assert "decision-0" in provider.prior_section_of(len(chunks))
    assert outcome.llm_calls == len(chunks) + 1
    assert outcome.chunks == len(chunks)
    assert outcome.usage == times(CALL_USAGE, len(chunks) + 1)


async def test_two_call_yields_the_same_outcome_as_single_for_the_same_model_answers() -> None:
    entries = [
        entry("picked vendor", "2026-01-20", 0),
        entry("launch in spring", None, 2),
        entry("future claim", "2099-01-01", 1),
    ]
    single = await run(
        PipelineProvider([output(entries=entries)]), small_conversation(), mode="single"
    )

    two_call = await run(
        PipelineProvider([category_output(), entries_output(*entries)]),
        small_conversation(),
        mode="two_call",
    )

    assert type(two_call) is type(single)
    assert two_call.candidates == single.candidates
    assert (two_call.topic_path, two_call.topic_title, two_call.summary) == (
        single.topic_path,
        single.topic_title,
        single.summary,
    )


async def test_two_call_truncated_entries_call_splits_without_recategorizing() -> None:
    messages = small_conversation()
    (chunk,) = chunk_messages(messages)
    first_half, second_half = split_in_two(chunk)
    provider = PipelineProvider(
        [category_output(), _truncated(), entries_output(entry("a")), entries_output(entry("b"))]
    )

    outcome = await run(provider, messages, mode="two_call")

    assert provider.schemas == [CHUNK_CATEGORIZE_SCHEMA] + [CHUNK_ENTRIES_SCHEMA] * 3
    assert provider.transcript_of(2) == first_half.text
    assert provider.transcript_of(3) == second_half.text
    assert [c.content for c in outcome.candidates] == ["a", "b"]
    assert outcome.chunks == 2
    assert outcome.usage == plus(times(CALL_USAGE, 3), FAILED_USAGE)


async def test_two_call_categorize_failure_makes_no_entries_call_and_counts_its_usage() -> None:
    provider = PipelineProvider([LLMRefusalError("refused", usage=FAILED_USAGE, attempts=2)])

    failure = await expect_failure(provider, small_conversation(), mode="two_call")

    assert isinstance(failure.error, LLMRefusalError)
    assert provider.schemas == [CHUNK_CATEGORIZE_SCHEMA]
    assert failure.llm_calls == 2
    assert failure.usage_so_far == FAILED_USAGE


# -- Triage ----------------------------------------------------------------------


def _verdict(worth_recording: bool) -> dict[str, Any]:
    return {"worth_recording": worth_recording}


@pytest.mark.parametrize("mode", ["single", "two_call"])
async def test_triage_negative_chunk_makes_no_extraction_call_and_counts_triage_usage(
    mode: str,
) -> None:
    provider = PipelineProvider([_verdict(False)], attempts=2)

    outcome = await run(provider, small_conversation(), mode=mode, triage_enabled=True)

    assert provider.schemas == [TRIAGE_SCHEMA]
    assert provider.transcript_of(0) == chunk_messages(small_conversation())[0].text
    assert outcome.candidates == ()
    assert (outcome.topic_path, outcome.topic_title, outcome.summary) == ("", "", "")
    assert outcome.usage == CALL_USAGE
    assert outcome.llm_calls == 2


async def test_triage_positive_chunk_is_extracted_and_both_calls_are_counted() -> None:
    provider = PipelineProvider([_verdict(True), output(entries=[entry("picked vendor")])])

    outcome = await run(provider, small_conversation(), triage_enabled=True)

    assert provider.schemas == [TRIAGE_SCHEMA, CHUNK_EXTRACTION_SCHEMA]
    assert [c.content for c in outcome.candidates] == ["picked vendor"]
    assert outcome.llm_calls == 2
    assert outcome.usage == times(CALL_USAGE, 2)


async def test_two_call_with_triage_categorizes_on_the_first_chunk_worth_recording() -> None:
    messages = large_conversation()
    chunks = chunk_messages(messages)
    rest: list[Step] = []
    for _ in chunks[2:]:
        rest += [_verdict(True), entries_output()]
    steps: list[Step] = [
        _verdict(False),
        _verdict(True),
        category_output(topic="work/second"),
        entries_output(),
        *rest,
    ]
    provider = PipelineProvider(steps)

    outcome = await run(provider, messages, mode="two_call", triage_enabled=True)

    assert provider.schemas[:4] == [
        TRIAGE_SCHEMA,
        TRIAGE_SCHEMA,
        CHUNK_CATEGORIZE_SCHEMA,
        CHUNK_ENTRIES_SCHEMA,
    ]
    assert provider.schemas.count(CHUNK_CATEGORIZE_SCHEMA) == 1
    assert provider.transcript_of(2) == chunks[1].text
    assert outcome.topic_path == "work/second"
    assert outcome.llm_calls == len(steps)


async def test_triage_failure_fails_the_run_and_counts_its_requests() -> None:
    provider = PipelineProvider([LLMTransientError("down", attempts=3)])

    failure = await expect_failure(provider, small_conversation(), triage_enabled=True)

    assert isinstance(failure.error, LLMTransientError)
    assert provider.schemas == [TRIAGE_SCHEMA]
    assert failure.llm_calls == 3


async def test_spans_name_the_stage_of_each_call(
    in_memory_tracer: tuple[Any, InMemoryExporter],
) -> None:
    _, exporter = in_memory_tracer
    provider = PipelineProvider([_verdict(True), category_output(), entries_output()])

    await run(provider, small_conversation(), mode="two_call", triage_enabled=True)

    assert [attrs["stage"] for attrs in _llm_spans(exporter)] == [
        "triage",
        "categorize",
        "extract",
    ]
