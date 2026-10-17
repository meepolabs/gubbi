"""LLM phase of conversation extraction: chunk, extract per chunk, date entries.

In ``single`` mode the pipeline makes one structured call per transcript
chunk; in ``two_call`` mode it first categorizes the conversation once, then
extracts entries per chunk under that topic. Either way the entries found so
far are forwarded to every later chunk so the model does not repeat them, and
the topic comes from the first chunk extracted; later chunks' topic fields
are ignored. With triage enabled, each chunk is first screened by a small
call and a chunk screened out gets no extraction call. A chunk whose
extraction output is truncated is split in two and each half retried once; a
second truncation fails the run. No new call starts past the deadline, and
each call gets the time left before it as its budget, retries included.

Every failure surfaces as :class:`PipelineFailure` carrying the token usage
of every call made, the failed call included, so the job can account for it.
The running spend is also reported after every call through ``on_spend``, so a
run cancelled mid-call still knows what its earlier calls cost.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from functools import partial
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol

import structlog
from opentelemetry import trace

from gubbi.constants import PRIOR_ENTRIES_MAX_TOKENS
from gubbi.extraction.chunking import (
    Chunk,
    chunk_messages,
    estimate_tokens,
    render_prior_entries,
    split_in_two,
)
from gubbi.extraction.dating import resolve_event_date
from gubbi.extraction.llm.provider import (
    LLMDeadlineError,
    LLMProviderError,
    LLMRefusalError,
    LLMTruncatedError,
    LLMUsage,
)
from gubbi.telemetry.attrs import _NS_PER_MS, _TRACER_NAME, SpanNames, safe_set_attributes
from gubbi.telemetry.sanitized_errors import record_exception_sanitized

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence
    from datetime import date

    from opentelemetry.trace import Span

    from gubbi.extraction.chunk_extraction import ChunkExtraction, TriageVerdict
    from gubbi.extraction.chunking import ChunkableMessage, PriorEntry
    from gubbi.extraction.service import ExtractionService

__all__: list[str] = [
    "ExtractedCandidate",
    "ExtractionMode",
    "PipelineDeadlineError",
    "PipelineFailure",
    "PipelineOutcome",
    "PipelineSpend",
    "run_pipeline",
]

logger = structlog.get_logger(__name__)

_SUMMARY_SEPARATOR = " "

ExtractionMode = Literal["single", "two_call"]

_STAGE_TRIAGE: Final = "triage"
_STAGE_CATEGORIZE: Final = "categorize"
_STAGE_EXTRACT: Final = "extract"

_BOUNDARY_NOTE: Final = (
    "Boundary: transcript messages [0] through [{last}] are earlier context whose entries "
    "were already recorded; read them for context only and record entries only from "
    "message [{first}] onward."
)


@dataclass(frozen=True)
class ExtractedCandidate:
    """One entry the model proposed, with its event date resolved."""

    content: str
    why: str
    tags: tuple[str, ...]
    entry_date: date
    source_message_index: int | None


@dataclass(frozen=True)
class PipelineSpend:
    """Running totals of every call the pipeline has made so far."""

    usage: LLMUsage
    llm_calls: int
    """Upstream requests made, provider retries included."""
    chunks: int
    unmetered_calls: int
    """Requests among ``llm_calls`` whose usage the upstream never reported
    (timed out), so ``usage`` undercounts what they cost."""


def _ignore_spend(_spend: PipelineSpend) -> None:
    return None


@dataclass(frozen=True)
class PipelineOutcome:
    """Topic, summary and dated candidates for a conversation, plus call accounting.

    When triage screened out every chunk, nothing was extracted: ``topic_path``,
    ``topic_title`` and ``summary`` are empty and ``candidates`` is empty.
    """

    topic_path: str
    topic_title: str
    summary: str
    candidates: tuple[ExtractedCandidate, ...]
    usage: LLMUsage
    llm_calls: int
    """Upstream requests made, provider retries included."""
    chunks: int
    """Chunks the transcript was cut into, a truncation split counting as two."""
    unmetered_calls: int = 0
    """Requests whose usage the upstream never reported; see :class:`PipelineSpend`."""


class PipelineDeadlineError(Exception):
    """The job deadline passed before an LLM call could start or finish."""


class PipelineFailure(Exception):
    """The pipeline stopped; ``error`` is the cause, usage covers every call made."""

    def __init__(
        self,
        error: Exception,
        usage_so_far: LLMUsage,
        *,
        llm_calls: int,
        chunks: int,
        unmetered_calls: int = 0,
    ) -> None:
        super().__init__(f"extraction pipeline failed: {type(error).__name__}")
        self.error = error
        self.usage_so_far = usage_so_far
        self.llm_calls = llm_calls
        self.chunks = chunks
        self.unmetered_calls = unmetered_calls


@dataclass(frozen=True)
class _Run:
    service: ExtractionService
    messages: Sequence[ChunkableMessage]
    existing_topics: tuple[str, ...]
    conversation_created: date
    today: date
    deadline: float
    clock: Callable[[], float]
    mode: ExtractionMode
    triage_enabled: bool
    on_spend: Callable[[PipelineSpend], None]
    context_count: int
    prior_entries: tuple[PriorEntry, ...]
    boundary_note: str


@dataclass(frozen=True)
class _Progress:
    usage: LLMUsage = field(default_factory=LLMUsage)
    llm_calls: int = 0
    chunks: int = 0
    unmetered_calls: int = 0
    extractions: tuple[ChunkExtraction, ...] = ()
    """Call results in order; in two-call mode the categorization comes first."""
    candidates: tuple[ExtractedCandidate, ...] = ()


class _CallResult(Protocol):
    @property
    def usage(self) -> LLMUsage: ...

    @property
    def finish_reason(self) -> str: ...

    @property
    def attempts(self) -> int: ...

    @property
    def unmetered_attempts(self) -> int: ...


class _Request[R](Protocol):
    def __call__(self, *, time_budget: float) -> Awaitable[R]: ...


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
    mode: ExtractionMode = "single",
    triage_enabled: bool = False,
    on_spend: Callable[[PipelineSpend], None] = _ignore_spend,
    context_count: int = 0,
    prior_entries: Sequence[PriorEntry] = (),
) -> PipelineOutcome:
    """Extract topic, summary and dated entry candidates from a role-filtered transcript.

    ``deadline`` is a reading of ``clock`` (``time.monotonic`` by default)
    after which no further LLM call starts; a call in flight gets the time
    left as its budget, so its retries end at the deadline too.
    ``source_message_index`` on each candidate indexes ``messages``, and
    dating reads the message at that index. ``mode`` picks one structured
    call per chunk (``single``) or a categorize call followed by an entries
    call per chunk (``two_call``); ``triage_enabled`` screens each chunk first. Triage
    calls count toward ``usage`` and ``llm_calls`` like any other call.
    ``on_spend`` receives the running totals once the transcript is chunked
    and again after every call, failed calls included.

    The first ``context_count`` messages are context already extracted: every
    extract call is told so, and a candidate whose ``source_message_index``
    falls among them is dropped, as is a candidate with no index whenever
    ``context_count`` is above zero.
    ``prior_entries`` are entries recorded earlier from the same conversation,
    oldest first; they are shown to every extract call ahead of the run's own
    candidates.

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
        mode=mode,
        triage_enabled=triage_enabled,
        on_spend=on_spend,
        context_count=context_count,
        prior_entries=tuple(prior_entries),
        boundary_note=_boundary_note(context_count),
    )
    chunks = chunk_messages(messages)
    progress = _Progress(chunks=len(chunks))
    on_spend(_spend_of(progress))
    try:
        if not chunks:
            raise _CallFailedError(ValueError("transcript produced no chunks"), progress)
        for chunk_index, chunk in enumerate(chunks):
            progress = await _chunk_step(run, progress, chunk, chunk_index)
    except _CallFailedError as failed:
        raise PipelineFailure(
            failed.error,
            failed.progress.usage,
            llm_calls=failed.progress.llm_calls,
            chunks=failed.progress.chunks,
            unmetered_calls=failed.progress.unmetered_calls,
        ) from failed.error
    return _outcome(progress)


async def _chunk_step(run: _Run, progress: _Progress, chunk: Chunk, chunk_index: int) -> _Progress:
    """Triage one chunk if enabled, categorize once in two-call mode, then extract it."""
    if run.triage_enabled:
        verdict: TriageVerdict
        verdict, progress = await _call(
            run, progress, chunk_index, _STAGE_TRIAGE, partial(run.service.triage, chunk.text)
        )
        if not verdict.worth_recording:
            return progress
    if run.mode == "two_call" and not progress.extractions:
        categorize = partial(run.service.categorize_chunk, chunk.text, list(run.existing_topics))
        categorized, progress = await _call(
            run, progress, chunk_index, _STAGE_CATEGORIZE, categorize
        )
        progress = replace(progress, extractions=(categorized,))
    return await _extract_chunk_step(run, progress, chunk, chunk_index)


async def _extract_chunk_step(
    run: _Run, progress: _Progress, chunk: Chunk, chunk_index: int
) -> _Progress:
    """Extract one chunk, splitting it once if the model output is truncated."""
    try:
        return await _extract(run, progress, chunk, chunk_index)
    except _CallFailedError as failed:
        if not isinstance(failed.error, LLMTruncatedError):
            raise
        progress = replace(failed.progress, chunks=failed.progress.chunks + 1)
        run.on_spend(_spend_of(progress))
    for half in split_in_two(chunk):
        progress = await _extract(run, progress, half, chunk_index)
    return progress


async def _extract(run: _Run, progress: _Progress, chunk: Chunk, chunk_index: int) -> _Progress:
    prior = _prior_text(run, progress)
    request: _Request[ChunkExtraction]
    if run.mode == "two_call":
        topic_path = progress.extractions[0].topic_path
        request = partial(run.service.extract_chunk_entries, chunk.text, topic_path, prior)
    else:
        request = partial(run.service.extract_chunk, chunk.text, list(run.existing_topics), prior)
    extraction, progress = await _call(run, progress, chunk_index, _STAGE_EXTRACT, request)
    return replace(
        progress,
        extractions=(*progress.extractions, extraction),
        candidates=(*progress.candidates, *_candidates(run, extraction)),
    )


async def _call[R: _CallResult](
    run: _Run,
    progress: _Progress,
    chunk_index: int,
    stage: str,
    request: _Request[R],
) -> tuple[R, _Progress]:
    """Make one LLM call under its span; account its usage and requests, failed or not."""
    time_budget = run.deadline - run.clock()
    if time_budget <= 0:
        raise _CallFailedError(PipelineDeadlineError("job deadline passed"), progress)
    span_name = SpanNames.EXTRACTION_LLM_CALL
    start_ns = time.monotonic_ns()
    with trace.get_tracer(_TRACER_NAME).start_as_current_span(span_name) as span:
        safe_set_attributes(
            span_name,
            span,
            {"provider_name": run.service.provider_name, "model_name": run.service.model_name},
        )
        try:
            result = await request(time_budget=time_budget)
        except Exception as exc:
            usage = exc.usage if isinstance(exc, LLMProviderError) else None
            failed = _account(
                progress,
                LLMUsage() if usage is None else usage,
                _attempts_of(exc),
                exc.unmetered_attempts if isinstance(exc, LLMProviderError) else 0,
            )
            run.on_spend(_spend_of(failed))
            await _report_failure(span, exc, usage, chunk_index, stage, _latency_ms(start_ns))
            raise _CallFailedError(_pipeline_error(exc), failed) from exc
        attrs = {
            **_usage_attrs(result.usage),
            "finish_reason": result.finish_reason,
            "attempts": result.attempts,
            "chunk_index": chunk_index,
            "stage": stage,
            "latency_ms": _latency_ms(start_ns),
        }
        safe_set_attributes(span_name, span, attrs)
        await logger.info("extraction_llm_call", outcome="ok", **attrs)
    progress = _account(progress, result.usage, result.attempts, result.unmetered_attempts)
    run.on_spend(_spend_of(progress))
    return result, progress


async def _report_failure(
    span: Span,
    exc: Exception,
    usage: LLMUsage | None,
    chunk_index: int,
    stage: str,
    latency_ms: float,
) -> None:
    attrs: dict[str, Any] = {"chunk_index": chunk_index, "stage": stage, "latency_ms": latency_ms}
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


def _pipeline_error(exc: Exception) -> Exception:
    if isinstance(exc, LLMDeadlineError):
        return PipelineDeadlineError("job deadline passed during an LLM call")
    return exc


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


def _account(progress: _Progress, usage: LLMUsage, requests: int, unmetered: int) -> _Progress:
    return replace(
        progress,
        usage=LLMUsage(
            input=progress.usage.input + usage.input,
            output=progress.usage.output + usage.output,
            cached_input=progress.usage.cached_input + usage.cached_input,
            reasoning=progress.usage.reasoning + usage.reasoning,
        ),
        llm_calls=progress.llm_calls + requests,
        unmetered_calls=progress.unmetered_calls + unmetered,
    )


def _spend_of(progress: _Progress) -> PipelineSpend:
    return PipelineSpend(
        usage=progress.usage,
        llm_calls=progress.llm_calls,
        chunks=progress.chunks,
        unmetered_calls=progress.unmetered_calls,
    )


def _boundary_note(context_count: int) -> str:
    if context_count <= 0:
        return ""
    return _BOUNDARY_NOTE.format(last=context_count - 1, first=context_count)


def _prior_text(run: _Run, progress: _Progress) -> str:
    entries = (*run.prior_entries, *progress.candidates)
    if not run.boundary_note:
        return render_prior_entries(entries)
    budget = PRIOR_ENTRIES_MAX_TOKENS - estimate_tokens(f"{run.boundary_note}\n")
    rendered = render_prior_entries(entries, budget)
    return run.boundary_note if not rendered else f"{run.boundary_note}\n{rendered}"


def _in_context_tail(run: _Run, source_message_index: int | None) -> bool:
    # An unattributed candidate cannot be shown to come from the new messages.
    if run.context_count <= 0:
        return False
    return source_message_index is None or source_message_index < run.context_count


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
        if not _in_context_tail(run, entry.source_message_index)
    )


def _outcome(progress: _Progress) -> PipelineOutcome:
    first = progress.extractions[0] if progress.extractions else None
    summaries = (extraction.summary.strip() for extraction in progress.extractions)
    return PipelineOutcome(
        topic_path="" if first is None else first.topic_path,
        topic_title="" if first is None else first.topic_title,
        summary=_SUMMARY_SEPARATOR.join(summary for summary in summaries if summary),
        candidates=progress.candidates,
        usage=progress.usage,
        llm_calls=progress.llm_calls,
        chunks=progress.chunks,
        unmetered_calls=progress.unmetered_calls,
    )
