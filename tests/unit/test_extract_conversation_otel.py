"""OTel span tests for the extract_conversation Arq job.

Verifies that the job entry point emits a parent ``extraction.job`` span
that wraps the body, captures success/failure outcome and records
exceptions, and that an ``extraction.llm_call`` child span is fired around
each LLM provider call.

The tests use the in-memory tracer fixture (``tests/conftest.py``) so the
SDK records spans without dialling an OTLP collector.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from gubbi.extraction.jobs.extract_conversation import extract_conversation
from gubbi.extraction.llm.provider import LLMRateLimitError, LLMRefusalError
from gubbi.extraction.pipeline import PipelineFailure
from tests.fixtures.extraction_job import (
    CALL_USAGE,
    CONVERSATION_ID,
    USER_ID,
    ConnectionTracker,
    ScriptedProvider,
    chunk_output,
    make_ctx,
    patched_job,
)

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import ReadableSpan

    from tests.conftest import InMemoryExporter

pytestmark = pytest.mark.unit


def _spans(exporter: InMemoryExporter, name: str) -> list[ReadableSpan]:
    return [span for span in exporter.spans if span.name == name]


def _attrs(span: ReadableSpan) -> dict[str, Any]:
    return dict(span.attributes or {})


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        pytest.param(LLMRateLimitError("rate limited"), "llm_rate_limited", id="rate-limit"),
        pytest.param(LLMRefusalError("refused", usage=CALL_USAGE), "llm_refused", id="refusal"),
    ],
)
async def test_job_span_records_a_failed_llm_call(
    in_memory_tracer: tuple[Any, InMemoryExporter],
    error: Exception,
    reason: str,
) -> None:
    _tracer, exporter = in_memory_tracer
    tracker = ConnectionTracker()

    with patched_job(tracker), pytest.raises(PipelineFailure):
        await extract_conversation(make_ctx(ScriptedProvider([error])), CONVERSATION_ID, USER_ID)

    (job_span,) = _spans(exporter, "extraction.job")
    assert _attrs(job_span).get("success") is False
    assert _attrs(job_span).get("failure_reason") == reason
    assert "exception" in [event.name for event in job_span.events]
    (llm_span,) = _spans(exporter, "extraction.llm_call")
    assert "exception" in [event.name for event in llm_span.events]


async def test_job_span_records_success_with_llm_child_spans(
    in_memory_tracer: tuple[Any, InMemoryExporter],
) -> None:
    _tracer, exporter = in_memory_tracer
    tracker = ConnectionTracker()

    with patched_job(tracker):
        result = await extract_conversation(
            make_ctx(ScriptedProvider([chunk_output()])), CONVERSATION_ID, USER_ID
        )
    assert result["skipped"] is False

    (job_span,) = _spans(exporter, "extraction.job")
    job_attrs = _attrs(job_span)
    assert job_attrs.get("success") is True
    assert "latency_ms" in job_attrs
    assert "failure_reason" not in job_attrs

    (llm_span,) = _spans(exporter, "extraction.llm_call")
    llm_attrs = _attrs(llm_span)
    assert llm_attrs.get("provider_name") == "ScriptedProvider"
    assert llm_attrs.get("model_name") == "scripted-model"
    assert llm_attrs.get("prompt_size") == CALL_USAGE.input
    assert llm_attrs.get("completion_size") == CALL_USAGE.output
    assert "latency_ms" in llm_attrs


async def test_llm_call_spans_are_children_of_the_job_span(
    in_memory_tracer: tuple[Any, InMemoryExporter],
) -> None:
    """An llm_call span opened outside the job span could not be linked to the job."""
    _tracer, exporter = in_memory_tracer
    tracker = ConnectionTracker()

    with patched_job(tracker):
        await extract_conversation(
            make_ctx(ScriptedProvider([chunk_output()])), CONVERSATION_ID, USER_ID
        )

    (job_span,) = _spans(exporter, "extraction.job")
    llm_spans = _spans(exporter, "extraction.llm_call")
    assert llm_spans
    for child in llm_spans:
        assert child.parent is not None
        assert child.parent.span_id == job_span.context.span_id
