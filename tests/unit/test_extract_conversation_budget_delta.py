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
    get_settings().llm.llm_budget_enabled -- the helper presence
    alone gates the delta write.  The parameter is
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

    The worker checks only `helper is not None`.
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
# Pre-charge refund on worker failure
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

    # Refund happened BEFORE mark_failed.
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


# ---------------------------------------------------------------------------
# Refund period-bucket safety
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refund_skipped_with_metric_when_period_unknown() -> None:
    """When the worker fails BEFORE Phase 1's period_start lookup, refund is skipped.

    Simulates the dangerous window:
      * pre-charge has already been debited at ingest (period X = 2026-05-01)
      * period boundary is crossed BEFORE the worker runs
        (current_period_start() now returns 2026-06-01)
      * worker fails inside Phase 1 BEFORE extraction_jobs.get_period_start
        completes (e.g. asyncpg connection failure on _check_idempotent)

    Without the gate, the refund would land in bucket Y (2026-06-01) instead
    of bucket X -- giving the user a credit they shouldn't have AND leaving a
    phantom debit in the original period. The fix skips the refund and emits
    `extraction.refund_skipped_total{reason=unknown_period}` so an operator
    can manually reconcile.
    """
    from gubbi.extraction.jobs.extract_conversation import _get_extraction_refund_skipped_counter

    redis = AsyncMock()
    helper = MagicMock()
    helper.record_actual_cost = AsyncMock()
    ctx = _make_minimal_ctx(redis=redis, budget_helper=helper)

    async def _idempotent_raises(*_a: Any, **_kw: Any) -> bool:
        # Failure BEFORE mark_running / get_period_start runs.
        raise ConnectionError("asyncpg lost connection")

    add_calls: list[tuple[int, dict[str, str]]] = []

    def _capture_add(amount: int, attributes: dict[str, str] | None = None) -> None:
        add_calls.append((amount, attributes or {}))

    # Bind the counter to a local before patch.object so the lru_cache
    # is sealed deliberately rather than as an implicit side effect of
    # constructing the patches list.
    refund_counter = _get_extraction_refund_skipped_counter()

    patches = [
        patch("gubbi.extraction.jobs.extract_conversation.user_scoped_connection"),
        patch(
            "gubbi.extraction.jobs.extract_conversation._check_idempotent",
            side_effect=_idempotent_raises,
        ),
        patch(
            "gubbi.extraction.jobs.extract_conversation._mark_job_failed",
            new=AsyncMock(),
        ),
        # current_period_start NOW (worker time) is the wrong bucket -- the
        # ingest pre-charge debited 2026-05-01, but a boundary has been
        # crossed since.
        patch(
            "gubbi.extraction.jobs.extract_conversation.current_period_start",
            return_value=date(2026, 6, 1),
        ),
        # get_period_start patched but should not be reached (idempotent_raises
        # fires first); patched defensively so a regression that calls it does
        # not silently hit the real DB-shaped function on the AsyncMock conn.
        patch(
            "gubbi.extraction.jobs.extract_conversation.extraction_jobs.get_period_start",
            new=AsyncMock(return_value=date(2026, 5, 1)),
        ),
        patch.object(refund_counter, "add", side_effect=_capture_add),
    ]

    with (
        patches[0] as mock_conn_cm,
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
    ):
        _setup_conn_mock(mock_conn_cm)
        with pytest.raises(ConnectionError, match="asyncpg lost connection"):
            await extract_conversation(ctx, conversation_id=1, user_id=_USER_ID_STR, job_id=_JOB_ID)

    # Refund MUST NOT be issued: the period was unknown.
    helper.record_actual_cost.assert_not_called()

    # Metric MUST be emitted with reason=unknown_period.
    unknown_period_calls = [c for c in add_calls if c[1].get("reason") == "unknown_period"]
    assert len(unknown_period_calls) == 1, (
        f"expected one extraction.refund_skipped_total{{reason=unknown_period}} "
        f"emission, got {add_calls!r}"
    )


@pytest.mark.asyncio
async def test_refund_uses_pre_charge_period() -> None:
    """When Phase 1's period_start lookup succeeds, refund uses THAT period (not runtime).

    Companion to the unknown_period skip case: confirms the happy path still
    routes the refund into the bucket that ingest pre-charged, even when the
    runtime current_period_start() has advanced past a boundary.
    """
    redis = AsyncMock()
    helper = MagicMock()
    helper.record_actual_cost = AsyncMock()
    ctx = _make_minimal_ctx(redis=redis, budget_helper=helper)

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
        # Runtime period (worker time) has advanced past the original bucket.
        patch(
            "gubbi.extraction.jobs.extract_conversation.current_period_start",
            return_value=date(2026, 6, 1),
        ),
        # Job row records the original pre-charge bucket.
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
    assert kwargs["period_start"] == date(2026, 5, 1), (
        "Refund must use the pre-charge bucket (2026-05-01) loaded from the "
        "job row, not the runtime period (2026-06-01)."
    )


@pytest.mark.asyncio
async def test_refund_skipped_with_metric_when_job_period_lookup_returns_none() -> None:
    """Missing job-row period_start must stay on the skip path.

    A successful call to get_period_start() is not enough if it returns None:
    the worker still does not know which bucket ingest pre-charged. Refunding
    against the runtime period would mis-bucket across a month boundary.
    """
    from gubbi.extraction.jobs.extract_conversation import _get_extraction_refund_skipped_counter

    redis = AsyncMock()
    helper = MagicMock()
    helper.record_actual_cost = AsyncMock()
    ctx = _make_minimal_ctx(redis=redis, budget_helper=helper)

    async def _categorize_raises(*_a: Any, **_kw: Any) -> Any:
        raise RuntimeError("LLM hard failure")

    add_calls: list[tuple[int, dict[str, str]]] = []

    def _capture_add(amount: int, attributes: dict[str, str] | None = None) -> None:
        add_calls.append((amount, attributes or {}))

    # Bind the counter to a local before patch.object (see helper-binding
    # rationale at the first patch.object site in this module).
    refund_counter = _get_extraction_refund_skipped_counter()

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
            return_value=date(2026, 6, 1),
        ),
        patch(
            "gubbi.extraction.jobs.extract_conversation.extraction_jobs.get_period_start",
            new=AsyncMock(return_value=None),
        ),
        patch.object(refund_counter, "add", side_effect=_capture_add),
    ]

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
        with pytest.raises(RuntimeError, match="LLM hard failure"):
            await extract_conversation(ctx, conversation_id=1, user_id=_USER_ID_STR, job_id=_JOB_ID)

    helper.record_actual_cost.assert_not_called()
    unknown_period_calls = [c for c in add_calls if c[1].get("reason") == "unknown_period"]
    assert len(unknown_period_calls) == 1


# ---------------------------------------------------------------------------
# Pre-charge refund on no-topic skip path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_topic_path_refunds_pre_charge() -> None:
    """When categorization yields no usable topic_path, the pre-charge is refunded.

    The worker reaches Phase 2, the LLM returns a topic_path that
    ``harden_llm_topic_path`` rejects (or returns None), and the worker exits
    via ``_mark_skipped_no_topic`` without spending any cents on the user's
    bucket. The pre-charge debited at ingest must therefore be refunded with
    ``actual_cents=0, estimated_cents=PRE_CHARGE_CENTS`` against the
    pre-charge period (loaded from the job row in Phase 1).

    Catches: a regression that drops the refund call from the no-topic exit,
    leaving PRE_CHARGE_CENTS as a phantom debit on the user's bucket -- the
    operator over-bills the user for a categorization that produced nothing.
    """
    from gubbi_common.budget import PRE_CHARGE_CENTS

    redis = AsyncMock()
    helper = MagicMock()
    helper.record_actual_cost = AsyncMock()
    ctx = _make_minimal_ctx(redis=redis, budget_helper=helper)

    patches = [
        patch("gubbi.extraction.jobs.extract_conversation.user_scoped_connection"),
        patch("gubbi.extraction.jobs.extract_conversation._check_idempotent", return_value=False),
        patch(
            "gubbi.extraction.jobs.extract_conversation._load_conversation_for_extraction",
            return_value=(MagicMock(), [], []),
        ),
        # Phase 2 returns topic_path=None -- the no-topic exit fires.
        patch(
            "gubbi.extraction.jobs.extract_conversation._categorize_and_resolve_topic",
            return_value=(_FAKE_CATEGORIZATION, None),
        ),
        patch(
            "gubbi.extraction.jobs.extract_conversation._mark_skipped_no_topic",
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
        result = await extract_conversation(
            ctx, conversation_id=1, user_id=_USER_ID_STR, job_id=_JOB_ID
        )

    assert result["skipped"] is True
    assert result["topic_path"] is None
    helper.record_actual_cost.assert_called_once()
    kwargs = helper.record_actual_cost.call_args.kwargs
    assert kwargs["actual_cents"] == 0
    assert kwargs["estimated_cents"] == PRE_CHARGE_CENTS
    # Refund routes against the pre-charge bucket loaded from the job row.
    assert kwargs["period_start"] == date(2026, 5, 1)


@pytest.mark.asyncio
async def test_no_topic_path_refund_failure_logged_and_swallowed() -> None:
    """A Redis failure during the no-topic refund must not break the skip exit.

    Mirrors the success-path delta tolerance: extraction-side persistence
    decisions are durable in Postgres; the budget refund is best-effort.
    A Redis blip cannot turn the no-topic skip into a hard failure that
    triggers the outer-except path (which would then mark the job failed
    rather than completed).
    """
    redis = AsyncMock()
    helper = MagicMock()
    helper.record_actual_cost = AsyncMock(side_effect=ConnectionError("redis is down"))
    ctx = _make_minimal_ctx(redis=redis, budget_helper=helper)

    patches = [
        patch("gubbi.extraction.jobs.extract_conversation.user_scoped_connection"),
        patch("gubbi.extraction.jobs.extract_conversation._check_idempotent", return_value=False),
        patch(
            "gubbi.extraction.jobs.extract_conversation._load_conversation_for_extraction",
            return_value=(MagicMock(), [], []),
        ),
        patch(
            "gubbi.extraction.jobs.extract_conversation._categorize_and_resolve_topic",
            return_value=(_FAKE_CATEGORIZATION, None),
        ),
        patch(
            "gubbi.extraction.jobs.extract_conversation._mark_skipped_no_topic",
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
        # Must NOT raise -- Redis failure is swallowed and the no-topic
        # skip envelope returns cleanly.
        result = await extract_conversation(
            ctx, conversation_id=1, user_id=_USER_ID_STR, job_id=_JOB_ID
        )

    assert result["skipped"] is True
    helper.record_actual_cost.assert_called_once()


@pytest.mark.asyncio
async def test_no_topic_path_skips_refund_when_helper_missing() -> None:
    """Self-host (helper=None) takes the no-topic exit cleanly with no refund call."""
    redis = AsyncMock()
    ctx = _make_minimal_ctx(redis=redis, budget_helper=None)

    patches = [
        patch("gubbi.extraction.jobs.extract_conversation.user_scoped_connection"),
        patch("gubbi.extraction.jobs.extract_conversation._check_idempotent", return_value=False),
        patch(
            "gubbi.extraction.jobs.extract_conversation._load_conversation_for_extraction",
            return_value=(MagicMock(), [], []),
        ),
        patch(
            "gubbi.extraction.jobs.extract_conversation._categorize_and_resolve_topic",
            return_value=(_FAKE_CATEGORIZATION, None),
        ),
        patch(
            "gubbi.extraction.jobs.extract_conversation._mark_skipped_no_topic",
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
        result = await extract_conversation(
            ctx, conversation_id=1, user_id=_USER_ID_STR, job_id=_JOB_ID
        )

    assert result["skipped"] is True
    assert result["topic_path"] is None
