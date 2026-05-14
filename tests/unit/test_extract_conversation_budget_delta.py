"""Unit tests for the budget delta integration in extract_conversation."""

from __future__ import annotations

from datetime import date
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from gubbi.extraction.jobs.extract_conversation import extract_conversation
from gubbi.extraction.service import (
    CategorizationResult,
    ExtractedEntry,
    ExtractionEntriesResult,
)

_USER_UUID = UUID("aaaaaaaa-1111-2222-3333-444444444444")
_USER_ID_STR = str(_USER_UUID)
_JOB_ID = str(uuid4())

_FAKE_CATEGORIZATION = CategorizationResult(
    topic_path="health/fitness",
    topic_title="Fitness",
    summary="s",
    confidence=0.9,
)
_FAKE_EXTRACTION = ExtractionEntriesResult(
    entries=[ExtractedEntry(content="ran 5k", reasoning="daily log", tags=[], entry_date=None)],
    input_tokens=10,
    output_tokens=10,
)


def _make_minimal_ctx(redis: Any = None, budget_helper: Any = None) -> dict[str, Any]:
    """Build a minimal ExtractionContext-like dict."""
    # Set _llm=None so the cost-estimation getattr chain in
    # extract_conversation returns 0 cents cleanly without spawning
    # un-awaited coroutines from AsyncMock auto-magic on _llm.estimate_cost_cents.
    extraction_service = AsyncMock()
    extraction_service._llm = None
    return {
        "pool": AsyncMock(),
        "cipher": MagicMock(),
        "extraction_service": extraction_service,
        "redis": redis,
        "budget_helper": budget_helper,
    }


def _make_patch_stack(
    *,
    budget_enabled: bool,
    persist_side_effect: Any = None,
) -> list[Any]:
    """Return list of context managers to patch all private helpers.

    budget_enabled is kept as a parameter for API compatibility with existing
    test call sites.  The extract_conversation worker no longer checks
    get_settings().llm.journal_llm_budget_enabled -- the helper presence
    alone gates the delta write (B3-L2 simplification).  The parameter is
    therefore unused inside this function but retained so callers need no
    changes.
    """

    async def _persist(*args: Any, **kwargs: Any) -> int:
        if persist_side_effect is not None:
            return await persist_side_effect(*args, **kwargs)
        return 1

    return [
        patch("gubbi.extraction.jobs.extract_conversation.user_scoped_connection"),
        patch("gubbi.extraction.jobs.extract_conversation._check_idempotent", return_value=False),
        patch(
            "gubbi.extraction.jobs.extract_conversation._load_conversation_for_extraction",
            return_value=(MagicMock(), [], []),
        ),
        patch(
            "gubbi.extraction.jobs.extract_conversation._categorize_and_resolve_topic",
            return_value=(_FAKE_CATEGORIZATION, "health/fitness"),
        ),
        patch(
            "gubbi.extraction.jobs.extract_conversation._persist_extraction", side_effect=_persist
        ),
        patch("gubbi.extraction.jobs.extract_conversation._publish_progress", new=AsyncMock()),
        patch(
            "gubbi.extraction.jobs.extract_conversation.current_period_start",
            return_value=date(2026, 5, 1),
        ),
        patch(
            "gubbi.extraction.jobs.extract_conversation.extraction_jobs.get_period_start",
            new=AsyncMock(return_value=date(2026, 5, 1)),
        ),
    ]


def _setup_conn_mock(mock_conn_cm: Any) -> None:
    """Configure user_scoped_connection mock to return a usable async context manager."""
    mock_conn = AsyncMock()

    # transaction() must return a proper async context manager (not a coroutine)
    txn_ctx = MagicMock()
    txn_ctx.__aenter__ = AsyncMock(return_value=None)
    txn_ctx.__aexit__ = AsyncMock(return_value=False)
    # Override transaction to return the context manager directly (not a coroutine)
    mock_conn.transaction = MagicMock(return_value=txn_ctx)

    # user_scoped_connection itself is an async context manager
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=mock_conn)
    cm.__aexit__ = AsyncMock(return_value=False)
    mock_conn_cm.return_value = cm


@pytest.mark.asyncio
async def test_budget_delta_called_when_flag_enabled() -> None:
    """When helper is set, record_actual_cost is called with all required kwargs."""
    helper = MagicMock()
    helper.record_actual_cost = AsyncMock()
    redis = AsyncMock()
    ctx = _make_minimal_ctx(redis=redis, budget_helper=helper)

    patches = _make_patch_stack(budget_enabled=True)
    with (
        patches[0] as mock_conn_cm,
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
        patches[6],
        patches[7],
    ):
        _setup_conn_mock(mock_conn_cm)
        await extract_conversation(ctx, conversation_id=1, user_id=_USER_ID_STR, job_id=_JOB_ID)

    helper.record_actual_cost.assert_called_once()
    kwargs = helper.record_actual_cost.call_args.kwargs
    assert "user_id" in kwargs
    assert "period_start" in kwargs
    assert "actual_cents" in kwargs
    assert "estimated_cents" in kwargs


@pytest.mark.asyncio
async def test_budget_delta_called_when_helper_is_set() -> None:
    """Delta is written whenever helper is not None, regardless of budget flag.

    After B3-L2: the worker checks only `helper is not None`.
    The budget flag controls whether BudgetHelper is constructed at startup;
    at runtime the helper presence is the sole gate.
    """
    helper = MagicMock()
    helper.record_actual_cost = AsyncMock()
    redis = AsyncMock()
    ctx = _make_minimal_ctx(redis=redis, budget_helper=helper)

    # budget_enabled param is kept for API compat but no longer drives the gate.
    patches = _make_patch_stack(budget_enabled=False)
    with (
        patches[0] as mock_conn_cm,
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
        patches[6],
        patches[7],
    ):
        _setup_conn_mock(mock_conn_cm)
        await extract_conversation(ctx, conversation_id=1, user_id=_USER_ID_STR, job_id=_JOB_ID)

    # helper is set -> delta is written (flag no longer suppresses).
    helper.record_actual_cost.assert_called_once()


@pytest.mark.asyncio
async def test_budget_delta_redis_failure_does_not_rollback() -> None:
    """Redis failure in budget delta must NOT affect the persistence transaction."""
    redis = AsyncMock()

    persist_called = False

    async def _persist(*args: Any, **kwargs: Any) -> int:
        nonlocal persist_called
        persist_called = True
        return 2

    async def _raise(*args: Any, **kwargs: Any) -> None:
        raise ConnectionError("redis is down")

    helper = MagicMock()
    helper.record_actual_cost = AsyncMock(side_effect=_raise)
    ctx = _make_minimal_ctx(redis=redis, budget_helper=helper)

    patches = _make_patch_stack(
        budget_enabled=True,
        persist_side_effect=_persist,
    )

    with (
        patches[0] as mock_conn_cm,
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
        patches[6],
        patches[7],
    ):
        _setup_conn_mock(mock_conn_cm)
        # Must NOT raise -- Redis failure is swallowed
        result = await extract_conversation(
            ctx, conversation_id=1, user_id=_USER_ID_STR, job_id=_JOB_ID
        )

    # Persistence happened (SAVEPOINT was NOT rolled back)
    assert persist_called
    # Function returned a normal result (skipped=False)
    assert result["skipped"] is False


@pytest.mark.asyncio
async def test_budget_delta_skipped_when_helper_is_none() -> None:
    """Worker without a budget_helper in ctx (self-host) skips delta cleanly."""
    redis = AsyncMock()
    ctx = _make_minimal_ctx(redis=redis, budget_helper=None)

    patches = _make_patch_stack(budget_enabled=True)
    with (
        patches[0] as mock_conn_cm,
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
        patches[6],
        patches[7],
    ):
        _setup_conn_mock(mock_conn_cm)
        result = await extract_conversation(
            ctx, conversation_id=1, user_id=_USER_ID_STR, job_id=_JOB_ID
        )

    # No exception, skipped=False, no helper to assert against.
    assert result["skipped"] is False


# ---------------------------------------------------------------------------
# Pre-charge refund on worker failure (B2 / Q2)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pre_charge_refund_called_before_mark_failed_on_worker_exception() -> None:
    """When the worker raises in Phase 2/3, the pre-charge is refunded before mark_failed.

    Sequencing matters: a refund-failure must NOT block the row update, and
    the row update is what gives orphan_cleanup something to reason about.
    Both ops run independently (each in their own try/except in the worker).
    """
    from gubbi_common.budget import PRE_CHARGE_CENTS

    redis = AsyncMock()
    helper = MagicMock()
    helper.record_actual_cost = AsyncMock()
    ctx = _make_minimal_ctx(redis=redis, budget_helper=helper)

    call_log: list[str] = []

    async def _record_cost(**_kw: Any) -> None:
        call_log.append("refund")

    async def _mark_failed(*_a: Any, **_kw: Any) -> None:
        call_log.append("mark_failed")

    helper.record_actual_cost.side_effect = _record_cost

    async def _categorize_raises(*_a: Any, **_kw: Any) -> Any:
        # Force a worker failure inside the try block.
        raise RuntimeError("LLM hard failure")

    patches = [
        patch("gubbi.extraction.jobs.extract_conversation.user_scoped_connection"),
        patch("gubbi.extraction.jobs.extract_conversation._check_idempotent", return_value=False),
        patch(
            "gubbi.extraction.jobs.extract_conversation._load_conversation_for_extraction",
            return_value=(MagicMock(), [], []),
        ),
        patch(
            "gubbi.extraction.jobs.extract_conversation._categorize_and_resolve_topic",
            side_effect=_categorize_raises,
        ),
        patch(
            "gubbi.extraction.jobs.extract_conversation._mark_job_failed",
            new=AsyncMock(side_effect=_mark_failed),
        ),
        patch(
            "gubbi.extraction.jobs.extract_conversation.current_period_start",
            return_value=date(2026, 5, 1),
        ),
        patch(
            "gubbi.extraction.jobs.extract_conversation.extraction_jobs.get_period_start",
            new=AsyncMock(return_value=date(2026, 5, 1)),
        ),
    ]

    with (
        patches[0] as mock_conn_cm,
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
        patches[6],
    ):
        _setup_conn_mock(mock_conn_cm)
        with pytest.raises(RuntimeError, match="LLM hard failure"):
            await extract_conversation(ctx, conversation_id=1, user_id=_USER_ID_STR, job_id=_JOB_ID)

    helper.record_actual_cost.assert_called_once()
    kwargs = helper.record_actual_cost.call_args.kwargs
    assert kwargs["actual_cents"] == 0
    assert kwargs["estimated_cents"] == PRE_CHARGE_CENTS

    # Refund happened BEFORE mark_failed (Q2 sequencing).
    assert call_log == ["refund", "mark_failed"]


@pytest.mark.asyncio
async def test_pre_charge_refund_failure_does_not_block_mark_failed() -> None:
    """A Redis failure during refund must NOT prevent the extraction_jobs row from being marked failed."""
    redis = AsyncMock()
    helper = MagicMock()

    async def _refund_redis_down(**_kw: Any) -> None:
        raise ConnectionError("redis is down")

    helper.record_actual_cost = AsyncMock(side_effect=_refund_redis_down)
    ctx = _make_minimal_ctx(redis=redis, budget_helper=helper)

    mark_failed_called = False

    async def _mark_failed(*_a: Any, **_kw: Any) -> None:
        nonlocal mark_failed_called
        mark_failed_called = True

    async def _categorize_raises(*_a: Any, **_kw: Any) -> Any:
        raise RuntimeError("LLM hard failure")

    patches = [
        patch("gubbi.extraction.jobs.extract_conversation.user_scoped_connection"),
        patch("gubbi.extraction.jobs.extract_conversation._check_idempotent", return_value=False),
        patch(
            "gubbi.extraction.jobs.extract_conversation._load_conversation_for_extraction",
            return_value=(MagicMock(), [], []),
        ),
        patch(
            "gubbi.extraction.jobs.extract_conversation._categorize_and_resolve_topic",
            side_effect=_categorize_raises,
        ),
        patch(
            "gubbi.extraction.jobs.extract_conversation._mark_job_failed",
            new=AsyncMock(side_effect=_mark_failed),
        ),
        patch(
            "gubbi.extraction.jobs.extract_conversation.current_period_start",
            return_value=date(2026, 5, 1),
        ),
        patch(
            "gubbi.extraction.jobs.extract_conversation.extraction_jobs.get_period_start",
            new=AsyncMock(return_value=date(2026, 5, 1)),
        ),
    ]

    with (
        patches[0] as mock_conn_cm,
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
        patches[6],
    ):
        _setup_conn_mock(mock_conn_cm)
        with pytest.raises(RuntimeError, match="LLM hard failure"):
            await extract_conversation(ctx, conversation_id=1, user_id=_USER_ID_STR, job_id=_JOB_ID)

    helper.record_actual_cost.assert_called_once()
    assert mark_failed_called, "mark_failed must run even when refund fails"


@pytest.mark.asyncio
async def test_no_refund_when_helper_is_none() -> None:
    """Self-host (helper=None) takes the same exception path with no refund call."""
    redis = AsyncMock()
    ctx = _make_minimal_ctx(redis=redis, budget_helper=None)

    async def _categorize_raises(*_a: Any, **_kw: Any) -> Any:
        raise RuntimeError("LLM hard failure")

    patches = [
        patch("gubbi.extraction.jobs.extract_conversation.user_scoped_connection"),
        patch("gubbi.extraction.jobs.extract_conversation._check_idempotent", return_value=False),
        patch(
            "gubbi.extraction.jobs.extract_conversation._load_conversation_for_extraction",
            return_value=(MagicMock(), [], []),
        ),
        patch(
            "gubbi.extraction.jobs.extract_conversation._categorize_and_resolve_topic",
            side_effect=_categorize_raises,
        ),
        patch(
            "gubbi.extraction.jobs.extract_conversation._mark_job_failed",
            new=AsyncMock(),
        ),
        patch(
            "gubbi.extraction.jobs.extract_conversation.current_period_start",
            return_value=date(2026, 5, 1),
        ),
        patch(
            "gubbi.extraction.jobs.extract_conversation.extraction_jobs.get_period_start",
            new=AsyncMock(return_value=date(2026, 5, 1)),
        ),
    ]

    with (
        patches[0] as mock_conn_cm,
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
        patches[6],
    ):
        _setup_conn_mock(mock_conn_cm)
        with pytest.raises(RuntimeError, match="LLM hard failure"):
            await extract_conversation(ctx, conversation_id=1, user_id=_USER_ID_STR, job_id=_JOB_ID)
