"""LLM phase of conversation extraction: chunk, extract per chunk, date entries.

The pipeline makes one structured call per transcript chunk, forwarding the
entries found so far to every later chunk so the model does not repeat them.
The topic comes from the first chunk; later chunks' topic fields are ignored.
A chunk whose output is truncated is split in two and each half retried once;
a second truncation fails the run. No new call starts past the deadline.

Every failure surfaces as :class:`PipelineFailure` carrying the token usage
of every call made, the failed call included, so the job can account for it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import structlog
from opentelemetry import trace

from gubbi.extraction.chunking import Chunk, chunk_messages, render_prior_entries, split_in_two
from gubbi.extraction.dating import resolve_event_date
from gubbi.extraction.llm.provider import (
    LLMProviderError,
    LLMRefusalError,
    LLMTruncatedError,
    LLMUsage,
)
from gubbi.telemetry.attrs import _NS_PER_MS, _TRACER_NAME, SpanNames, safe_set_attributes
from gubbi.telemetry.sanitized_errors import record_exception_sanitized

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from datetime import date

    from opentelemetry.trace import Span

    from gubbi.extraction.chunk_extraction import ChunkExtraction
    from gubbi.extraction.chunking import ChunkableMessage
    from gubbi.extraction.service import ExtractionService

__all__: list[str] = [
    "ExtractedCandidate",
    "PipelineDeadlineError",
    "PipelineFailure",
    "PipelineOutcome",
    "run_pipeline",
]

logger = structlog.get_logger(__name__)

_SUMMARY_SEPARATOR = " "


@dataclass(frozen=True)
class ExtractedCandidate:
    """One entry the model proposed, with its event date resolved."""

    content: str
    why: str
    tags: tuple[str, ...]
    entry_date: date
    source_message_index: int | None


@dataclass(frozen=True)
class PipelineOutcome:
    """Topic, summary and dated candidates for a conversation, plus call accounting."""

    topic_path: str
    topic_title: str
    summary: str
    candidates: tuple[ExtractedCandidate, ...]
    usage: LLMUsage
    llm_calls: int
    """Upstream requests made, provider retries included."""
    chunks: int
    """Chunks the transcript was cut into, a truncation split counting as two."""


class PipelineDeadlineError(Exception):
    """The job deadline passed before the next LLM call could start."""


class PipelineFailure(Exception):
    """The pipeline stopped; ``error`` is the cause, usage covers every call made."""

    def __init__(
        self,
        error: Exception,
        usage_so_far: LLMUsage,
        *,
        llm_calls: int,
        chunks: int,
    ) -> None:
        super().__init__(f"extraction pipeline failed: {type(error).__name__}")
        self.error = error
        self.usage_so_far = usage_so_far
        self.llm_calls = llm_calls
        self.chunks = chunks


@dataclass(frozen=True)
class _Run:
    service: ExtractionService
    messages: Sequence[ChunkableMessage]
    existing_topics: tuple[str, ...]
    conversation_created: date
    today: date
    deadline: float
    clock: Callable[[], float]


@dataclass(frozen=True)
class _Progress:
    usage: LLMUsage = field(default_factory=LLMUsage)
    llm_calls: int = 0
    chunks: int = 0
    extractions: tuple[ChunkExtraction, ...] = ()
    candidates: tuple[ExtractedCandidate, ...] = ()


class _CallFailedError(Exception):
    def __init__(self, error: Exception, progress: _Progress) -> None:
        super().__init__(type(error).__name__)
        self.error = error
        self.progress = progress


async def run_pipeline(
    service: ExtractionService,
    messages: Sequence[ChunkableMessage],
    *,
    existing_topics: Sequence[str],
    conversation_created: date,
    deadline: float,
    today: date,
    clock: Callable[[], float] = time.monotonic,
) -> PipelineOutcome:
    """Extract topic, summary and dated entry candidates from a role-filtered transcript.

    ``deadline`` is a reading of ``clock`` (``time.monotonic`` by default)
    after which no further LLM call starts. ``source_message_index`` on each
    candidate indexes ``messages``.

    Raises:
        PipelineFailure: any call failed, a chunk stayed truncated after one
            split, the deadline passed, or ``messages`` produced no chunk.
    """
    run = _Run(
        service=service,
        messages=messages,
        existing_topics=tuple(existing_topics),
        conversation_created=conversation_created,
        today=today,
        deadline=deadline,
        clock=clock,
    )
    chunks = chunk_messages(messages)
    progress = _Progress(chunks=len(chunks))
    try:
        if not chunks:
            raise _CallFailedError(ValueError("transcript produced no chunks"), progress)
        for chunk_index, chunk in enumerate(chunks):
            progress = await _extract_chunk_step(run, progress, chunk, chunk_index)
    except _CallFailedError as failed:
        raise PipelineFailure(
            failed.error,
            failed.progress.usage,
            llm_calls=failed.progress.llm_calls,
            chunks=failed.progress.chunks,
        ) from failed.error
    return _outcome(progress)


async def _extract_chunk_step(
    run: _Run, progress: _Progress, chunk: Chunk, chunk_index: int
) -> _Progress:
    """Extract one chunk, splitting it once if the model output is truncated."""
    try:
        return await _call(run, progress, chunk, chunk_index)
    except _CallFailedError as failed:
        if not isinstance(failed.error, LLMTruncatedError):
            raise
        progress = replace(failed.progress, chunks=failed.progress.chunks + 1)
    for half in split_in_two(chunk):
        progress = await _call(run, progress, half, chunk_index)
    return progress


async def _call(run: _Run, progress: _Progress, chunk: Chunk, chunk_index: int) -> _Progress:
    if run.clock() >= run.deadline:
        raise _CallFailedError(PipelineDeadlineError("job deadline passed"), progress)
    span_name = SpanNames.EXTRACTION_LLM_CALL
    start_ns = time.monotonic_ns()
    with trace.get_tracer(_TRACER_NAME).start_as_current_span(span_name) as span:
        try:
            extraction = await run.service.extract_chunk(
                chunk.text,
                list(run.existing_topics),
                render_prior_entries(progress.candidates),
            )
        except Exception as exc:
            usage = exc.usage if isinstance(exc, LLMProviderError) else None
            failed = _account(progress, LLMUsage() if usage is None else usage, _attempts_of(exc))
            await _report_failure(span, exc, usage, chunk_index, _latency_ms(start_ns))
            raise _CallFailedError(exc, failed) from exc
        attrs = {
            **_usage_attrs(extraction.usage),
            "finish_reason": extraction.finish_reason,
            "attempts": extraction.attempts,
            "chunk_index": chunk_index,
            "latency_ms": _latency_ms(start_ns),
        }
        safe_set_attributes(span_name, span, attrs)
        await logger.info("extraction_llm_call", outcome="ok", **attrs)
    return replace(
        _account(progress, extraction.usage, extraction.attempts),
        extractions=(*progress.extractions, extraction),
        candidates=(*progress.candidates, *_candidates(run, extraction)),
    )


async def _report_failure(
    span: Span,
    exc: Exception,
    usage: LLMUsage | None,
    chunk_index: int,
    latency_ms: float,
) -> None:
    attrs: dict[str, Any] = {"chunk_index": chunk_index, "latency_ms": latency_ms}
    if usage is not None:
        attrs.update(_usage_attrs(usage))
    finish_reason = _finish_reason_of(exc)
    if finish_reason is not None:
        attrs["finish_reason"] = finish_reason
    safe_set_attributes(SpanNames.EXTRACTION_LLM_CALL, span, attrs)
    record_exception_sanitized(span, exc)
    await logger.warning(
        "extraction_llm_call", outcome="failed", error_type=type(exc).__name__, **attrs
    )


def _finish_reason_of(exc: Exception) -> str | None:
    if isinstance(exc, LLMTruncatedError):
        return "length"
    if isinstance(exc, LLMRefusalError):
        return "content_filter"
    return None


def _usage_attrs(usage: LLMUsage) -> dict[str, int]:
    return {
        "prompt_size": usage.input,
        "completion_size": usage.output,
        "cached_prompt_size": usage.cached_input,
        "reasoning_size": usage.reasoning,
    }


def _latency_ms(start_ns: int) -> float:
    return round((time.monotonic_ns() - start_ns) / _NS_PER_MS, 2)


def _attempts_of(exc: Exception) -> int:
    """Upstream requests a failed call made; 1 when the error does not say."""
    if isinstance(exc, LLMProviderError) and exc.attempts is not None:
        return exc.attempts
    return 1


def _account(progress: _Progress, usage: LLMUsage, requests: int) -> _Progress:
    return replace(
        progress,
        usage=LLMUsage(
            input=progress.usage.input + usage.input,
            output=progress.usage.output + usage.output,
            cached_input=progress.usage.cached_input + usage.cached_input,
            reasoning=progress.usage.reasoning + usage.reasoning,
        ),
        llm_calls=progress.llm_calls + requests,
    )


def _candidates(run: _Run, extraction: ChunkExtraction) -> tuple[ExtractedCandidate, ...]:
    return tuple(
        ExtractedCandidate(
            content=entry.content,
            why=entry.why,
            tags=entry.tags,
            entry_date=resolve_event_date(
                entry.event_date,
                entry.source_message_index,
                run.messages,
                run.conversation_created,
                run.today,
            ),
            source_message_index=entry.source_message_index,
        )
        for entry in extraction.entries
    )


def _outcome(progress: _Progress) -> PipelineOutcome:
    first = progress.extractions[0]
    summaries = (extraction.summary.strip() for extraction in progress.extractions)
    return PipelineOutcome(
        topic_path=first.topic_path,
        topic_title=first.topic_title,
        summary=_SUMMARY_SEPARATOR.join(summary for summary in summaries if summary),
        candidates=progress.candidates,
        usage=progress.usage,
        llm_calls=progress.llm_calls,
        chunks=progress.chunks,
    )
