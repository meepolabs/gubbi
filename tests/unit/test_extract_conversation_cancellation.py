"""Unit tests for extract_conversation when Arq cancels it at its job timeout.

Arq enforces the timeout by cancelling the task. The job must still record the
spend it knows of, mark the job ``deadline_exceeded``, and let the cancellation
propagate, within a bounded cleanup. Redis is never touched.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

import pytest

from gubbi.extraction.chunking import chunk_messages
from gubbi.extraction.jobs import extract_conversation as job_module
from gubbi.extraction.jobs.extract_conversation import extract_conversation
from gubbi.extraction.llm.provider import LLMMessage, LLMResponse
from gubbi.extraction.retry_policy import FailureOutcome
from gubbi.models.conversation import Message
from tests.fixtures.extraction_job import (
    CALL_CENTS,
    CALL_USAGE,
    CONVERSATION_ID,
    JOB_ID,
    USER_ID,
    ConnectionTracker,
    ScriptedProvider,
    chunk_output,
    make_ctx,
    patched_job,
)

pytestmark = pytest.mark.unit


class _BlockingProvider(ScriptedProvider):
    """Answers its scripted steps, then blocks; ``called`` is set once it blocks."""

    def __init__(self, steps: list[Any] | None = None) -> None:
        super().__init__(list(steps or []))
        self._answers_left = len(steps or [])
        self.called = asyncio.Event()

    async def complete(
        self,
        messages: list[LLMMessage],
        system_prompt: str,
        output_schema: Any = None,
        *,
        time_budget: float | None = None,
    ) -> LLMResponse:
        if self._answers_left:
            self._answers_left -= 1
            return await super().complete(
                messages, system_prompt, output_schema, time_budget=time_budget
            )
        self.called.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


class _RecordingRegistry(set[asyncio.Task[None]]):
    """A cleanup registry that remembers every task ever added to it."""

    def __init__(self) -> None:
        super().__init__()
        self.added: list[asyncio.Task[None]] = []

    def add(self, task: asyncio.Task[None]) -> None:
        self.added.append(task)
        super().add(task)


def _multi_chunk_transcript() -> list[Message]:
    # 40 x 10k chars: several chunks at the default chunk target.
    return [
        Message(
            role="user" if i % 2 == 0 else "assistant",
            content=f"m{i} " + "x" * 10_000,
            timestamp="2026-03-10T09:00:00",
        )
        for i in range(40)
    ]


async def _cancel_when(ctx: dict[str, Any], started: asyncio.Event) -> asyncio.Task[Any]:
    task = asyncio.create_task(extract_conversation(ctx, CONVERSATION_ID, USER_ID, JOB_ID))
    await asyncio.wait_for(started.wait(), 5)
    task.cancel()
    return task


async def test_cancellation_during_a_model_call_fails_the_job_and_propagates() -> None:
    provider = _BlockingProvider()
    tracker = ConnectionTracker()
    ctx = make_ctx(provider)

    with patched_job(tracker) as mocks:
        task = await _cancel_when(ctx, provider.called)
        with pytest.raises(asyncio.CancelledError):
            await task

    assert mocks.mark_failed.await_args.kwargs["error_code"] == "deadline_exceeded"
    assert mocks.record_audit.await_args.kwargs["action"] == "extraction_job.failed"
    assert ctx["redis"].mock_calls == []
    assert tracker.open_now == 0


async def test_cancellation_after_the_pipeline_records_its_cost() -> None:
    tracker = ConnectionTracker()
    persisting = asyncio.Event()

    async def _hang(*_args: Any) -> None:
        persisting.set()
        await asyncio.Event().wait()

    with patched_job(tracker) as mocks:
        mocks.persist.side_effect = _hang
        task = await _cancel_when(make_ctx(ScriptedProvider([chunk_output()])), persisting)
        with pytest.raises(asyncio.CancelledError):
            await task

    kwargs = mocks.mark_failed.await_args.kwargs
    assert (kwargs["error_code"], kwargs["cents_spent"]) == ("deadline_exceeded", CALL_CENTS)


async def test_cancellation_mid_pipeline_records_the_calls_already_made() -> None:
    messages = _multi_chunk_transcript()
    assert len(chunk_messages(messages)) >= 2
    provider = _BlockingProvider([chunk_output()])
    tracker = ConnectionTracker()

    with patched_job(tracker, messages=messages) as mocks:
        task = await _cancel_when(make_ctx(provider), provider.called)
        with pytest.raises(asyncio.CancelledError):
            await task

    kwargs = mocks.mark_failed.await_args.kwargs
    assert kwargs["cents_spent"] == CALL_CENTS
    assert (kwargs["input_tokens"], kwargs["output_tokens"]) == (
        CALL_USAGE.input,
        CALL_USAGE.output,
    )
    assert kwargs["llm_calls"] == 1


async def test_a_stuck_cancellation_cleanup_is_abandoned_after_its_bound() -> None:
    provider = _BlockingProvider()
    tracker = ConnectionTracker()
    registry = _RecordingRegistry()

    async def _stuck(*_args: Any, **_kwargs: Any) -> bool:
        await asyncio.Event().wait()
        return True

    with (
        patched_job(tracker) as mocks,
        patch.object(job_module, "_CANCEL_CLEANUP_TIMEOUT_SECS", 0.05),
    ):
        mocks.mark_failed.side_effect = _stuck
        ctx = make_ctx(provider)
        ctx["cancel_cleanups"] = registry
        task = await _cancel_when(ctx, provider.called)
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
        (cleanup,) = registry.added
        await asyncio.wait({cleanup}, timeout=1)

    mocks.mark_failed.assert_awaited_once()
    assert cleanup.cancelled()
    assert not registry


async def test_a_second_cancellation_does_not_abandon_the_cleanup() -> None:
    provider = _BlockingProvider()
    tracker = ConnectionTracker()
    marking = asyncio.Event()
    release = asyncio.Event()

    async def _slow_mark_failed(*_args: Any, **_kwargs: Any) -> FailureOutcome:
        marking.set()
        await release.wait()
        return FailureOutcome.RETRY

    with patched_job(tracker) as mocks:
        mocks.mark_failed.side_effect = _slow_mark_failed
        task = await _cancel_when(make_ctx(provider), provider.called)
        await asyncio.wait_for(marking.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        await asyncio.wait_for(_cleanups_drained(), 5)

    assert mocks.record_audit.await_args.kwargs["action"] == "extraction_job.failed"


async def _cleanups_drained() -> None:
    while job_module._DETACHED_CLEANUPS:
        await asyncio.sleep(0)
