"""Unit tests for extract_conversation when Arq cancels it at its job timeout.

Arq enforces the timeout by cancelling the task. The job must still settle the
pre-charge against the spend it knows of, mark the job ``deadline_exceeded``,
and let the cancellation propagate, within a bounded cleanup.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from gubbi_common.budget import PRE_CHARGE_CENTS

from gubbi.extraction.jobs import extract_conversation as job_module
from gubbi.extraction.jobs.extract_conversation import extract_conversation
from gubbi.extraction.llm.provider import LLMMessage, LLMResponse
from tests.fixtures.extraction_job import (
    CALL_CENTS,
    CONVERSATION_ID,
    JOB_ID,
    PERIOD,
    USER_ID,
    ConnectionTracker,
    ScriptedProvider,
    chunk_output,
    make_ctx,
    patched_job,
)

pytestmark = pytest.mark.unit


class _BlockingProvider(ScriptedProvider):
    """A provider whose call never returns; ``called`` is set once it is in flight."""

    def __init__(self) -> None:
        super().__init__([])
        self.called = asyncio.Event()

    async def complete(
        self,
        messages: list[LLMMessage],
        system_prompt: str,
        output_schema: Any = None,
        *,
        time_budget: float | None = None,
    ) -> LLMResponse:
        self.called.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


def _helper() -> MagicMock:
    helper = MagicMock()
    helper.record_actual_cost = AsyncMock()
    return helper


async def _cancel_when(ctx: dict[str, Any], started: asyncio.Event) -> asyncio.Task[Any]:
    task = asyncio.create_task(extract_conversation(ctx, CONVERSATION_ID, USER_ID, JOB_ID))
    await asyncio.wait_for(started.wait(), 5)
    task.cancel()
    return task


async def test_cancellation_during_a_model_call_fails_the_job_and_propagates() -> None:
    helper = _helper()
    provider = _BlockingProvider()
    tracker = ConnectionTracker()

    with patched_job(tracker) as mocks:
        task = await _cancel_when(make_ctx(provider, helper), provider.called)
        with pytest.raises(asyncio.CancelledError):
            await task

    assert mocks.mark_failed.await_args.kwargs["error_code"] == "deadline_exceeded"
    assert mocks.record_audit.await_args.kwargs["action"] == "extraction_job.failed"
    helper.record_actual_cost.assert_awaited_once()
    settled = helper.record_actual_cost.await_args.kwargs
    assert (settled["period_start"], settled["estimated_cents"]) == (PERIOD, PRE_CHARGE_CENTS)
    assert tracker.open_now == 0


async def test_cancellation_after_the_pipeline_charges_its_cost() -> None:
    helper = _helper()
    tracker = ConnectionTracker()
    persisting = asyncio.Event()

    async def _hang(*_args: Any) -> None:
        persisting.set()
        await asyncio.Event().wait()

    with patched_job(tracker) as mocks:
        mocks.persist.side_effect = _hang
        task = await _cancel_when(make_ctx(ScriptedProvider([chunk_output()]), helper), persisting)
        with pytest.raises(asyncio.CancelledError):
            await task

    kwargs = mocks.mark_failed.await_args.kwargs
    assert (kwargs["error_code"], kwargs["cents_spent"]) == ("deadline_exceeded", CALL_CENTS)
    assert helper.record_actual_cost.await_args.kwargs["actual_cents"] == CALL_CENTS


async def test_a_stuck_cancellation_cleanup_is_abandoned_after_its_bound() -> None:
    provider = _BlockingProvider()
    tracker = ConnectionTracker()

    async def _stuck(*_args: Any, **_kwargs: Any) -> bool:
        await asyncio.Event().wait()
        return True

    with (
        patched_job(tracker) as mocks,
        patch.object(job_module, "_CANCEL_CLEANUP_TIMEOUT_SECS", 0.05),
    ):
        mocks.mark_failed.side_effect = _stuck
        task = await _cancel_when(make_ctx(provider), provider.called)
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)

    mocks.mark_failed.assert_awaited_once()


async def test_a_second_cancellation_does_not_abandon_the_cleanup() -> None:
    provider = _BlockingProvider()
    tracker = ConnectionTracker()
    marking = asyncio.Event()
    release = asyncio.Event()

    async def _slow_mark_failed(*_args: Any, **_kwargs: Any) -> bool:
        marking.set()
        await release.wait()
        return True

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
