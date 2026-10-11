"""Behavior tests for the extract_conversation Arq job (connection-split, typed outcomes)."""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

import pytest

from gubbi.constants import EXTRACTION_VERSION
from gubbi.extraction.jobs.extract_conversation import _classify_error, extract_conversation
from gubbi.extraction.llm.provider import (
    LLMPermanentError,
    LLMProviderError,
    LLMRateLimitError,
    LLMRefusalError,
    LLMSchemaError,
    LLMTimeoutError,
    LLMTransientError,
    LLMTruncatedError,
    LLMUsage,
)
from gubbi.extraction.pipeline import PipelineDeadlineError, PipelineFailure
from gubbi.extraction.skips import content_hash
from gubbi.models.conversation import Message, keep_conversation_roles
from tests.fixtures.extraction_job import (
    CALL_CENTS,
    CALL_USAGE,
    CONVERSATION_ID,
    JOB_ID,
    PLATFORM,
    USER_ID,
    ConnectionTracker,
    ScriptedProvider,
    chunk_output,
    entry,
    extraction_state,
    make_ctx,
    patched_job,
    transcript,
)

pytestmark = pytest.mark.unit

_JOB_UUID = UUID(JOB_ID)


def _run(ctx: dict[str, Any], job_id: str = JOB_ID) -> Any:
    return extract_conversation(ctx, CONVERSATION_ID, USER_ID, job_id)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


async def test_happy_path_persists_the_pipeline_outcome_and_reports_it() -> None:
    tracker = ConnectionTracker()
    provider = ScriptedProvider([chunk_output()], tracker)
    ctx = make_ctx(provider)

    with patched_job(tracker) as mocks:
        result = await _run(ctx)

    assert result == {
        "topic_path": "health/fitness",
        "entries_created": 1,
        "input_tokens": CALL_USAGE.input,
        "output_tokens": CALL_USAGE.output,
        "cents_spent": CALL_CENTS,
        "skipped": False,
    }
    mocks.persist.assert_awaited_once()
    _conn, _cipher, embedder, run = mocks.persist.await_args.args
    assert embedder is ctx["embedding_service"]
    assert run.job_id == _JOB_UUID
    assert run.client == PLATFORM
    assert run.topic_path == "health/fitness"
    assert [c.content for c in run.candidates] == ["Started running"]
    assert len(run.embeddings) == 1
    assert run.usage == CALL_USAGE
    assert (run.llm_calls, run.chunks, run.cents_spent) == (1, 1, CALL_CENTS)
    assert run.content_hash == content_hash(keep_conversation_roles(transcript()))


async def test_no_connection_is_held_during_any_llm_call() -> None:
    tracker = ConnectionTracker()
    provider = ScriptedProvider([chunk_output()], tracker)

    with patched_job(tracker):
        await _run(make_ctx(provider))

    assert provider.open_connections_at_call == [0]
    assert len(tracker.conns) == 2, "one load connection, one persist connection"


async def test_persist_rechecks_processed_at_under_a_row_lock_in_its_transaction() -> None:
    tracker = ConnectionTracker()
    order: list[str] = []

    with patched_job(tracker) as mocks:
        persisted = mocks.persist.return_value
        mocks.lock_processed_at.side_effect = lambda *_a: order.append("lock")
        mocks.persist.side_effect = lambda *_a: order.append("persist") or persisted
        await _run(make_ctx(ScriptedProvider([chunk_output()])))

    conn1, conn2 = tracker.conns
    assert [c.args[0] for c in mocks.get_processed_at.await_args_list] == [conn1]
    mocks.lock_processed_at.assert_awaited_once_with(conn2, CONVERSATION_ID)
    conn2.transaction.assert_called_once()
    assert order == ["lock", "persist"]


async def test_system_turns_never_reach_the_provider() -> None:
    tracker = ConnectionTracker()
    provider = ScriptedProvider([chunk_output()])

    with patched_job(tracker):
        await _run(make_ctx(provider))

    (sent,) = provider.user_contents
    assert "I started running this week." in sent
    assert "SYSTEM-PROMPT-MARKER" not in sent


async def test_message_dates_reach_the_provider() -> None:
    tracker = ConnectionTracker()
    provider = ScriptedProvider([chunk_output()])

    with patched_job(tracker):
        await _run(make_ctx(provider))

    assert "2026-03-02" in provider.user_contents[0]


async def test_extraction_settings_choose_the_pipeline_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JOURNAL_EXTRACTION_MODE", "two_call")
    monkeypatch.setenv("JOURNAL_EXTRACTION_TRIAGE", "true")
    tracker = ConnectionTracker()
    category = {"topic_path": "health/fitness", "topic_title": "Fitness", "summary": "s"}
    provider = ScriptedProvider(
        [{"worth_recording": True}, category, {"entries": [entry("Started running")]}]
    )

    with patched_job(tracker):
        result = await _run(make_ctx(provider))

    assert result["entries_created"] == 1
    assert [sorted(schema["properties"]) for schema in provider.schemas if schema] == [
        ["worth_recording"],
        ["summary", "topic_path", "topic_title"],
        ["entries"],
    ]


async def test_success_publishes_progress_on_the_users_job_channel() -> None:
    tracker = ConnectionTracker()
    ctx = make_ctx(ScriptedProvider([chunk_output()]))

    with patched_job(tracker):
        await _run(ctx)

    channel, payload = ctx["redis"].publish.await_args.args
    assert channel == f"extraction:user:{USER_ID}:job:{JOB_ID}"
    assert json.loads(payload) == {
        "topic_path": "health/fitness",
        "entries_created": 1,
        "job_id": JOB_ID,
        "conversation_id": CONVERSATION_ID,
    }


async def test_progress_publish_failure_does_not_fail_the_job() -> None:
    tracker = ConnectionTracker()
    ctx = make_ctx(ScriptedProvider([chunk_output()]))
    ctx["redis"].publish.side_effect = ConnectionError("redis down")

    with patched_job(tracker):
        result = await _run(ctx)

    assert result["skipped"] is False


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


async def test_already_processed_conversation_completes_its_job_without_work() -> None:
    tracker = ConnectionTracker()
    provider = ScriptedProvider([])

    with patched_job(tracker) as mocks:
        mocks.get_processed_at.return_value = "2026-05-01T00:00:00Z"
        result = await _run(make_ctx(provider))

    assert result["skipped"] is True
    assert provider.user_contents == []
    mocks.read_conversation.assert_not_awaited()
    mocks.mark_running.assert_awaited_once_with(tracker.conns[0], _JOB_UUID)
    mocks.mark_completed.assert_awaited_once_with(
        tracker.conns[0],
        _JOB_UUID,
        topics_created=0,
        entries_created=0,
        cents_spent=0,
        filtered_tokens=0,
    )
    assert len(tracker.conns) == 1


async def test_already_processed_conversation_leaves_an_unclaimable_job_alone() -> None:
    tracker = ConnectionTracker()

    with patched_job(tracker) as mocks:
        mocks.get_processed_at.return_value = "2026-05-01T00:00:00Z"
        mocks.mark_running.return_value = False
        result = await _run(make_ctx(ScriptedProvider([])))

    assert result["skipped"] is True
    mocks.mark_completed.assert_not_awaited()
    mocks.record_audit.assert_not_awaited()


async def test_mark_running_follows_the_idempotency_check() -> None:
    tracker = ConnectionTracker()
    order: list[str] = []

    with patched_job(tracker) as mocks:
        mocks.get_processed_at.side_effect = lambda *_a: order.append("check")
        mocks.mark_running.side_effect = lambda *_a: order.append("running")
        await _run(make_ctx(ScriptedProvider([chunk_output()])))

    assert order[:2] == ["check", "running"]
    mocks.mark_running.assert_awaited_once_with(tracker.conns[0], _JOB_UUID)


async def test_a_concurrent_worker_that_committed_first_wins() -> None:
    tracker = ConnectionTracker()

    ctx = make_ctx(ScriptedProvider([chunk_output()]))

    with patched_job(tracker) as mocks:
        mocks.lock_processed_at.return_value = "2026-05-01T00:00:00Z"
        result = await _run(ctx)

    assert result["skipped"] is True
    mocks.persist.assert_not_awaited()
    mocks.record_audit.assert_not_awaited()
    ctx["redis"].publish.assert_not_awaited()


async def test_a_job_row_another_execution_owns_is_left_alone() -> None:
    tracker = ConnectionTracker()
    provider = ScriptedProvider([chunk_output()])

    with patched_job(tracker) as mocks:
        mocks.mark_running.return_value = False
        result = await _run(make_ctx(provider))

    assert result["skipped"] is True
    assert provider.user_contents == []
    mocks.read_conversation.assert_not_awaited()
    mocks.persist.assert_not_awaited()
    mocks.mark_completed.assert_not_awaited()
    mocks.mark_failed.assert_not_awaited()
    mocks.record_audit.assert_not_awaited()


async def test_without_a_job_row_no_lifecycle_update_runs() -> None:
    tracker = ConnectionTracker()

    with patched_job(tracker) as mocks:
        await _run(make_ctx(ScriptedProvider([chunk_output()])), job_id="unknown")

    mocks.mark_running.assert_not_awaited()
    assert mocks.persist.await_args.args[3].job_id is None


async def test_failure_without_a_job_row_marks_nothing_failed() -> None:
    tracker = ConnectionTracker()

    with patched_job(tracker) as mocks, pytest.raises(PipelineFailure):
        await _run(make_ctx(ScriptedProvider([LLMPermanentError("x")])), job_id="unknown")

    mocks.mark_failed.assert_not_awaited()


async def test_persistence_failure_marks_the_job_failed_on_a_fresh_connection() -> None:
    tracker = ConnectionTracker()

    with patched_job(tracker) as mocks:
        mocks.persist.side_effect = RuntimeError("disk full")
        with pytest.raises(RuntimeError, match="disk full"):
            await _run(make_ctx(ScriptedProvider([chunk_output()])))

    assert len(tracker.conns) == 3
    assert mocks.mark_failed.await_args.args[0] is tracker.conns[2]
    assert mocks.mark_failed.await_args.kwargs["error_code"] == "internal_error"
    assert mocks.mark_failed.await_args.kwargs["input_tokens"] == CALL_USAGE.input
    audit = mocks.record_audit.await_args.kwargs
    assert audit["action"] == "extraction_job.failed"
    assert audit["actor_type"] == "user"
    assert audit["metadata"] == {
        "error_code": "internal_error",
        "conversation_id": CONVERSATION_ID,
    }


async def test_original_error_propagates_when_marking_failed_also_fails() -> None:
    tracker = ConnectionTracker()

    with patched_job(tracker) as mocks:
        mocks.mark_failed.side_effect = ConnectionError("db gone")
        with pytest.raises(PipelineFailure):
            await _run(make_ctx(ScriptedProvider([LLMPermanentError("x")])))


async def test_missing_conversation_fails_the_job() -> None:
    tracker = ConnectionTracker()

    with patched_job(tracker) as mocks:
        mocks.get_extraction_state.return_value = None
        with pytest.raises(LookupError):
            await _run(make_ctx(ScriptedProvider([])))

    assert mocks.mark_failed.await_args.kwargs["error_code"] == "internal_error"


# ---------------------------------------------------------------------------
# Typed failure outcomes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("error", "code"),
    [
        pytest.param(LLMRefusalError("no", usage=CALL_USAGE), "llm_refused", id="refusal"),
        pytest.param(LLMSchemaError("bad", usage=CALL_USAGE), "llm_schema", id="schema"),
        pytest.param(LLMRateLimitError("slow"), "llm_rate_limited", id="rate-limit"),
        pytest.param(LLMTimeoutError("late"), "llm_provider_error", id="timeout"),
        pytest.param(LLMPermanentError("auth"), "llm_provider_error", id="permanent"),
        pytest.param(LLMProviderError("odd"), "llm_provider_error", id="provider"),
    ],
)
async def test_a_failed_call_marks_the_job_failed_with_its_error_code(
    error: Exception, code: str
) -> None:
    tracker = ConnectionTracker()

    with patched_job(tracker) as mocks, pytest.raises(PipelineFailure):
        await _run(make_ctx(ScriptedProvider([error])))

    assert mocks.mark_failed.await_args.kwargs["error_code"] == code


async def test_truncation_after_one_split_marks_the_job_llm_truncated() -> None:
    tracker = ConnectionTracker()
    truncated = LLMTruncatedError("cap", usage=CALL_USAGE)

    with patched_job(tracker) as mocks, pytest.raises(PipelineFailure):
        await _run(make_ctx(ScriptedProvider([truncated, truncated])))

    assert mocks.mark_failed.await_args.kwargs["error_code"] == "llm_truncated"


async def test_a_passed_deadline_marks_the_job_deadline_exceeded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "gubbi.extraction.jobs.extract_conversation.EXTRACTION_JOB_DEADLINE_SECS", -1
    )
    tracker = ConnectionTracker()
    provider = ScriptedProvider([chunk_output()])

    with patched_job(tracker) as mocks, pytest.raises(PipelineFailure):
        await _run(make_ctx(provider))

    assert provider.user_contents == []
    assert mocks.mark_failed.await_args.kwargs["error_code"] == "deadline_exceeded"


async def test_refusal_records_the_refused_calls_usage_on_the_failed_job() -> None:
    tracker = ConnectionTracker()
    provider = ScriptedProvider([LLMRefusalError("no", usage=CALL_USAGE)])

    with patched_job(tracker) as mocks, pytest.raises(PipelineFailure):
        await _run(make_ctx(provider))

    assert len(provider.user_contents) == 1
    kwargs = mocks.mark_failed.await_args.kwargs
    assert kwargs == {
        "error_code": "llm_refused",
        "cents_spent": CALL_CENTS,
        "input_tokens": CALL_USAGE.input,
        "output_tokens": CALL_USAGE.output,
        "cached_input_tokens": CALL_USAGE.cached_input,
        "reasoning_tokens": CALL_USAGE.reasoning,
        "llm_calls": 1,
        "chunks": 1,
    }


class TestClassifyError:
    """``_classify_error`` maps causes to the stored error_code."""

    @pytest.mark.parametrize(
        ("exc", "code"),
        [
            pytest.param(LLMRefusalError("x"), "llm_refused", id="refusal"),
            pytest.param(LLMTruncatedError("x"), "llm_truncated", id="truncated"),
            pytest.param(LLMSchemaError("x"), "llm_schema", id="schema"),
            pytest.param(LLMRateLimitError("x"), "llm_rate_limited", id="rate-limit"),
            pytest.param(LLMTimeoutError("x"), "llm_provider_error", id="timeout"),
            pytest.param(LLMTransientError("x"), "llm_provider_error", id="transient"),
            pytest.param(LLMPermanentError("x"), "llm_provider_error", id="permanent"),
            pytest.param(LLMProviderError("x"), "llm_provider_error", id="provider"),
            pytest.param(PipelineDeadlineError("x"), "deadline_exceeded", id="deadline"),
            pytest.param(RuntimeError("x"), "internal_error", id="runtime"),
            pytest.param(ValueError("x"), "internal_error", id="value"),
        ],
    )
    def test_maps_a_bare_cause(self, exc: Exception, code: str) -> None:
        assert _classify_error(exc) == code

    @pytest.mark.parametrize(
        ("cause", "code"),
        [
            pytest.param(LLMRefusalError("x"), "llm_refused", id="refusal"),
            pytest.param(PipelineDeadlineError("x"), "deadline_exceeded", id="deadline"),
            pytest.param(ValueError("x"), "internal_error", id="other"),
        ],
    )
    def test_maps_a_pipeline_failure_by_its_cause(self, cause: Exception, code: str) -> None:
        failure = PipelineFailure(cause, LLMUsage(), llm_calls=0, chunks=0)

        assert _classify_error(failure) == code


# ---------------------------------------------------------------------------
# Nothing to record
# ---------------------------------------------------------------------------


def _assert_completed_empty(
    mocks: Any, *, cents: int, llm_calls: int, filtered_tokens: int | None
) -> None:
    mocks.persist.assert_not_awaited()
    mocks.mark_processed.assert_awaited_once()
    kwargs = mocks.mark_completed.await_args.kwargs
    assert kwargs["topics_created"] == 0
    assert kwargs["entries_created"] == 0
    assert kwargs["cents_spent"] == cents
    assert kwargs["llm_calls"] == llm_calls
    assert kwargs["filtered_tokens"] == filtered_tokens
    audit = mocks.record_audit.await_args.kwargs
    assert audit["action"] == "extraction_job.completed"
    assert audit["metadata"] == {"conversation_id": CONVERSATION_ID, "entries_created": 0}


async def test_no_entries_completes_the_job_with_zero_entries_and_records_the_hash() -> None:
    tracker = ConnectionTracker()
    ctx = make_ctx(ScriptedProvider([chunk_output(entries=[])]))

    with patched_job(tracker) as mocks:
        result = await _run(ctx)

    assert result["skipped"] is True
    assert result["entries_created"] == 0
    _assert_completed_empty(mocks, cents=CALL_CENTS, llm_calls=1, filtered_tokens=None)
    expected_hash = content_hash(keep_conversation_roles(transcript()))
    mocks.set_extraction_state.assert_awaited_once_with(
        tracker.conns[1], CONVERSATION_ID, expected_hash, EXTRACTION_VERSION
    )
    ctx["redis"].publish.assert_not_awaited()


async def test_an_unusable_topic_completes_the_job_with_zero_entries() -> None:
    tracker = ConnectionTracker()

    with patched_job(tracker) as mocks:
        result = await _run(make_ctx(ScriptedProvider([chunk_output(topic="../../etc")])))

    assert result["topic_path"] is None
    _assert_completed_empty(mocks, cents=CALL_CENTS, llm_calls=1, filtered_tokens=None)


async def test_every_chunk_triaged_out_completes_with_zero_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JOURNAL_EXTRACTION_TRIAGE", "true")
    tracker = ConnectionTracker()

    with patched_job(tracker) as mocks:
        await _run(make_ctx(ScriptedProvider([{"worth_recording": False}])))

    _assert_completed_empty(mocks, cents=CALL_CENTS, llm_calls=1, filtered_tokens=None)


# ---------------------------------------------------------------------------
# Rule-based skips
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "messages",
    [
        pytest.param([Message(role="system", content="only a system prompt")], id="no-text"),
        pytest.param(
            [Message(role="user", content="ok"), Message(role="assistant", content="Sure!")],
            id="trivial-user-text",
        ),
    ],
)
async def test_a_rule_skip_completes_the_job_without_a_model_call(
    messages: list[Message],
) -> None:
    tracker = ConnectionTracker()
    provider = ScriptedProvider([])

    with patched_job(tracker, messages=messages) as mocks:
        result = await _run(make_ctx(provider))

    assert result["skipped"] is True
    assert provider.user_contents == []
    _assert_completed_empty(mocks, cents=0, llm_calls=0, filtered_tokens=0)


async def test_unchanged_content_at_this_version_is_not_extracted_again() -> None:
    tracker = ConnectionTracker()
    provider = ScriptedProvider([])
    stored = content_hash(keep_conversation_roles(transcript()))

    with patched_job(
        tracker, state=extraction_state(stored_hash=stored, stored_version=EXTRACTION_VERSION)
    ) as mocks:
        await _run(make_ctx(provider))

    assert provider.user_contents == []
    _assert_completed_empty(mocks, cents=0, llm_calls=0, filtered_tokens=0)


async def test_unchanged_content_at_an_older_version_is_extracted_again() -> None:
    tracker = ConnectionTracker()
    provider = ScriptedProvider([chunk_output()])
    stored = content_hash(keep_conversation_roles(transcript()))

    with patched_job(
        tracker, state=extraction_state(stored_hash=stored, stored_version=EXTRACTION_VERSION - 1)
    ) as mocks:
        await _run(make_ctx(provider))

    assert len(provider.user_contents) == 1
    mocks.persist.assert_awaited_once()


async def test_load_failure_is_logged_and_re_raised() -> None:
    tracker = ConnectionTracker()

    with patched_job(tracker) as mocks:
        mocks.read_conversation.side_effect = RuntimeError("decrypt")
        with pytest.raises(RuntimeError, match="decrypt"):
            await _run(make_ctx(ScriptedProvider([])))

    assert len(tracker.conns) == 2, "load connection, then the failure connection"
