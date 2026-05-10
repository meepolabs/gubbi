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


def _make_minimal_ctx(redis: Any = None) -> dict[str, Any]:
    """Build a minimal ExtractionContext-like dict."""
    return {
        "pool": AsyncMock(),
        "cipher": MagicMock(),
        "extraction_service": AsyncMock(),
        "redis": redis,
    }


def _make_patch_stack(
    *,
    budget_enabled: bool,
    record_side_effect: Any = None,
    persist_side_effect: Any = None,
) -> list[Any]:
    """Return list of context managers to patch all private helpers."""
    mock_settings = MagicMock()
    mock_settings.llm.journal_llm_budget_enabled = budget_enabled

    record_mock = AsyncMock(side_effect=record_side_effect) if record_side_effect else AsyncMock()

    async def _persist(*args: Any, **kwargs: Any) -> int:
        if persist_side_effect is not None:
            return await persist_side_effect(*args, **kwargs)
        return 1

    return [
        patch(
            "gubbi.extraction.jobs.extract_conversation.get_settings", return_value=mock_settings
        ),
        patch("gubbi.extraction.jobs.extract_conversation.record_extraction_cost", new=record_mock),
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
    """When budget flag is enabled and redis is set, record_extraction_cost is called."""
    redis = AsyncMock()
    ctx = _make_minimal_ctx(redis=redis)

    patches = _make_patch_stack(budget_enabled=True)

    with (
        patches[0] as _,
        patches[1] as mock_record,
        patches[2] as mock_conn_cm,
        patches[3] as _,
        patches[4] as _,
        patches[5] as _,
        patches[6] as _,
        patches[7] as _,
        patches[8] as _,
    ):
        _setup_conn_mock(mock_conn_cm)
        await extract_conversation(ctx, conversation_id=1, user_id=_USER_ID_STR, job_id=_JOB_ID)

    assert mock_record.called
    call_kwargs = mock_record.call_args[1]
    assert "actual_cents" in call_kwargs
    assert "estimated_cents" in call_kwargs
    assert "redis" in call_kwargs


@pytest.mark.asyncio
async def test_budget_delta_not_called_when_flag_disabled() -> None:
    """When budget flag is False, record_extraction_cost must NOT be called."""
    redis = AsyncMock()
    ctx = _make_minimal_ctx(redis=redis)

    patches = _make_patch_stack(budget_enabled=False)

    with (
        patches[0] as _,
        patches[1] as mock_record,
        patches[2] as mock_conn_cm,
        patches[3] as _,
        patches[4] as _,
        patches[5] as _,
        patches[6] as _,
        patches[7] as _,
        patches[8] as _,
    ):
        _setup_conn_mock(mock_conn_cm)
        await extract_conversation(ctx, conversation_id=1, user_id=_USER_ID_STR, job_id=_JOB_ID)

    assert not mock_record.called


@pytest.mark.asyncio
async def test_budget_delta_redis_failure_does_not_rollback() -> None:
    """Redis failure in budget delta must NOT affect the persistence transaction."""
    redis = AsyncMock()
    ctx = _make_minimal_ctx(redis=redis)

    persist_called = False

    async def _persist(*args: Any, **kwargs: Any) -> int:
        nonlocal persist_called
        persist_called = True
        return 2

    async def _raise(*args: Any, **kwargs: Any) -> None:
        raise ConnectionError("redis is down")

    patches = _make_patch_stack(
        budget_enabled=True,
        record_side_effect=_raise,
        persist_side_effect=_persist,
    )

    with (
        patches[0] as _,
        patches[1] as _,
        patches[2] as mock_conn_cm,
        patches[3] as _,
        patches[4] as _,
        patches[5] as _,
        patches[6] as _,
        patches[7] as _,
        patches[8] as _,
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
