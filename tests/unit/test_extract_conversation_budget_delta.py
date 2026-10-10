"""Unit tests for the budget pre-charge reconciliation in extract_conversation.

Ingest pre-charges ``PRE_CHARGE_CENTS`` into the user's budget bucket. The
job reconciles that estimate against the estimated cost of every model call
it made, whatever the outcome: success, nothing to record, failure, or
cancellation. A rule skip makes no call and refunds in full. Every
reconciliation lands in the bucket read from the job row; when that bucket
is unknown the refund is skipped and counted instead.
"""

from __future__ import annotations

from datetime import date
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest
from gubbi_common.budget import PRE_CHARGE_CENTS

from gubbi.extraction.jobs import extract_conversation as job_module
from gubbi.extraction.jobs.extract_conversation import (
    _get_extraction_refund_skipped_counter,
    extract_conversation,
)
from gubbi.extraction.llm.provider import LLMPermanentError, LLMRefusalError
from gubbi.extraction.pipeline import PipelineFailure
from gubbi.models.conversation import Message
from tests.fixtures.extraction_job import (
    CALL_CENTS,
    CALL_USAGE,
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

_RUNTIME_PERIOD = date(2026, 6, 1)


def _helper(side_effect: Any = None) -> MagicMock:
    helper = MagicMock()
    helper.record_actual_cost = AsyncMock(side_effect=side_effect)
    return helper


async def _run(ctx: dict[str, Any]) -> dict[str, Any]:
    with patch.object(job_module, "current_period_start", return_value=_RUNTIME_PERIOD):
        return await extract_conversation(ctx, CONVERSATION_ID, USER_ID, JOB_ID)


def _settled(helper: MagicMock) -> dict[str, Any]:
    helper.record_actual_cost.assert_awaited_once()
    return dict(helper.record_actual_cost.await_args.kwargs)


# ---------------------------------------------------------------------------
# Success: actual cost against the pre-charge
# ---------------------------------------------------------------------------


async def test_success_records_the_actual_cost_against_the_pre_charge_bucket() -> None:
    helper = _helper()
    tracker = ConnectionTracker()

    with patched_job(tracker):
        await _run(make_ctx(ScriptedProvider([chunk_output()]), helper))

    assert _settled(helper) == {
        "user_id": UUID(USER_ID),
        "period_start": PERIOD,
        "actual_cents": CALL_CENTS,
        "estimated_cents": PRE_CHARGE_CENTS,
    }


async def test_budget_delta_failure_does_not_fail_a_persisted_run() -> None:
    helper = _helper(side_effect=ConnectionError("redis is down"))
    tracker = ConnectionTracker()

    with patched_job(tracker) as mocks:
        result = await _run(make_ctx(ScriptedProvider([chunk_output()]), helper))

    assert result["skipped"] is False
    mocks.persist.assert_awaited_once()
    mocks.mark_failed.assert_not_awaited()


async def test_without_a_budget_helper_the_job_runs_unchanged() -> None:
    tracker = ConnectionTracker()

    with patched_job(tracker):
        result = await _run(make_ctx(ScriptedProvider([chunk_output()])))

    assert result["skipped"] is False


# ---------------------------------------------------------------------------
# Failure: settle before the job is marked failed
# ---------------------------------------------------------------------------


async def test_failure_settles_the_pre_charge_before_marking_the_job_failed() -> None:
    call_log: list[str] = []
    helper = _helper(side_effect=lambda **_kw: call_log.append("settle"))
    tracker = ConnectionTracker()

    with patched_job(tracker) as mocks:
        mocks.mark_failed.side_effect = lambda *_a, **_kw: call_log.append("mark_failed")
        with pytest.raises(PipelineFailure):
            await _run(make_ctx(ScriptedProvider([LLMPermanentError("x")]), helper))

    settled = _settled(helper)
    assert settled["actual_cents"] == 0, "the failed call reported no usage"
    assert settled["estimated_cents"] == PRE_CHARGE_CENTS
    assert settled["period_start"] == PERIOD, "the job row's bucket, not the runtime period"
    assert call_log == ["settle", "mark_failed"]


async def test_a_failure_after_billed_calls_charges_what_they_cost() -> None:
    helper = _helper()
    tracker = ConnectionTracker()
    refused = LLMRefusalError("no", usage=CALL_USAGE)

    with patched_job(tracker) as mocks, pytest.raises(PipelineFailure):
        await _run(make_ctx(ScriptedProvider([refused]), helper))

    assert _settled(helper)["actual_cents"] == CALL_CENTS
    assert mocks.mark_failed.await_args.kwargs["cents_spent"] == CALL_CENTS


async def test_a_persistence_failure_charges_the_completed_pipelines_cost() -> None:
    helper = _helper()
    tracker = ConnectionTracker()

    with patched_job(tracker) as mocks:
        mocks.persist.side_effect = RuntimeError("disk full")
        with pytest.raises(RuntimeError, match="disk full"):
            await _run(make_ctx(ScriptedProvider([chunk_output()]), helper))

    assert _settled(helper)["actual_cents"] == CALL_CENTS


async def test_a_revived_failed_job_charges_its_full_cost() -> None:
    """The earlier failed run already settled its pre-charge, so none is left to offset."""
    helper = _helper()
    tracker = ConnectionTracker()

    with patched_job(tracker) as mocks:
        mocks.mark_running.return_value = "failed"
        await _run(make_ctx(ScriptedProvider([chunk_output()]), helper))

    settled = _settled(helper)
    assert (settled["actual_cents"], settled["estimated_cents"]) == (CALL_CENTS, 0)


async def test_a_job_another_execution_owns_leaves_the_budget_alone() -> None:
    helper = _helper()
    tracker = ConnectionTracker()

    with patched_job(tracker) as mocks:
        mocks.mark_running.return_value = None
        await _run(make_ctx(ScriptedProvider([chunk_output()]), helper))

    helper.record_actual_cost.assert_not_awaited()


async def test_losing_the_persist_race_still_charges_the_calls_made() -> None:
    helper = _helper()
    tracker = ConnectionTracker()

    with patched_job(tracker) as mocks:
        mocks.lock_processed_at.return_value = "2026-05-01T00:00:00Z"
        result = await _run(make_ctx(ScriptedProvider([chunk_output()]), helper))

    assert result["skipped"] is True
    assert _settled(helper)["actual_cents"] == CALL_CENTS


async def test_refund_failure_does_not_block_marking_the_job_failed() -> None:
    helper = _helper(side_effect=ConnectionError("redis is down"))
    tracker = ConnectionTracker()

    with patched_job(tracker) as mocks, pytest.raises(PipelineFailure):
        await _run(make_ctx(ScriptedProvider([LLMPermanentError("x")]), helper))

    helper.record_actual_cost.assert_awaited_once()
    mocks.mark_failed.assert_awaited_once()


async def test_failure_without_a_budget_helper_still_marks_the_job_failed() -> None:
    tracker = ConnectionTracker()

    with patched_job(tracker) as mocks, pytest.raises(PipelineFailure):
        await _run(make_ctx(ScriptedProvider([LLMPermanentError("x")])))

    mocks.mark_failed.assert_awaited_once()


# ---------------------------------------------------------------------------
# Unknown bucket: refund skipped and counted
# ---------------------------------------------------------------------------


def _capture_counter() -> tuple[Any, list[dict[str, str]]]:
    calls: list[dict[str, str]] = []

    def _add(_amount: int, attributes: dict[str, str] | None = None) -> None:
        calls.append(attributes or {})

    counter = _get_extraction_refund_skipped_counter()
    return patch.object(counter, "add", side_effect=_add), calls


async def test_refund_is_skipped_and_counted_when_failure_precedes_the_period_lookup() -> None:
    """A failure before the job row is read must not refund into the runtime period."""
    helper = _helper()
    tracker = ConnectionTracker()
    counter_patch, counted = _capture_counter()

    with patched_job(tracker) as mocks, counter_patch:
        mocks.get_processed_at.side_effect = ConnectionError("lost connection")
        with pytest.raises(ConnectionError, match="lost connection"):
            await _run(make_ctx(ScriptedProvider([]), helper))

    helper.record_actual_cost.assert_not_awaited()
    assert counted == [{"reason": "unknown_period"}]


async def test_refund_is_skipped_and_counted_when_the_job_row_has_no_period() -> None:
    helper = _helper()
    tracker = ConnectionTracker()
    counter_patch, counted = _capture_counter()

    with (
        patched_job(tracker, period_start=None),
        counter_patch,
        pytest.raises(PipelineFailure),
    ):
        await _run(make_ctx(ScriptedProvider([LLMPermanentError("x")]), helper))

    helper.record_actual_cost.assert_not_awaited()
    assert counted == [{"reason": "unknown_period"}]


# ---------------------------------------------------------------------------
# Nothing recorded: charged for the calls made, refunded when none were
# ---------------------------------------------------------------------------


_TRIVIAL = [Message(role="user", content="ok")]


@pytest.mark.parametrize(
    ("steps", "messages", "triage", "cents"),
    [
        pytest.param([chunk_output(entries=[])], None, False, CALL_CENTS, id="no-entries"),
        pytest.param([chunk_output(topic="../x")], None, False, CALL_CENTS, id="unusable-topic"),
        pytest.param([{"worth_recording": False}], None, True, CALL_CENTS, id="triaged-out"),
        pytest.param([], _TRIVIAL, False, 0, id="rule-skip"),
    ],
)
async def test_a_run_that_records_nothing_settles_the_cost_of_its_calls(
    monkeypatch: pytest.MonkeyPatch,
    steps: list[Any],
    messages: list[Message] | None,
    triage: bool,
    cents: int,
) -> None:
    monkeypatch.setenv("JOURNAL_EXTRACTION_TRIAGE", "true" if triage else "false")
    helper = _helper()
    tracker = ConnectionTracker()

    with patched_job(tracker, messages=messages):
        result = await _run(make_ctx(ScriptedProvider(steps), helper))

    assert result["skipped"] is True
    settled = _settled(helper)
    assert settled["actual_cents"] == cents
    assert settled["estimated_cents"] == PRE_CHARGE_CENTS
    assert settled["period_start"] == PERIOD


async def test_refund_failure_on_nothing_to_record_does_not_fail_the_job() -> None:
    helper = _helper(side_effect=ConnectionError("redis is down"))
    tracker = ConnectionTracker()

    with patched_job(tracker) as mocks:
        result = await _run(make_ctx(ScriptedProvider([chunk_output(entries=[])]), helper))

    assert result["skipped"] is True
    mocks.mark_completed.assert_awaited_once()
    mocks.mark_failed.assert_not_awaited()
