"""Unit tests for the @audited decorator (M3 requirement).

Tests cover:
    - Successful handler triggers an audit write
    - Failed handler (returns dict with "error" key) does NOT trigger an audit write
    - Audit write failure is swallowed (handler result returned, warning logged)
    - target_id extraction from result dict
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

from journalctl.core.audit_decorator import _extract_target_id, audited
from journalctl.core.context import AppContext

pytestmark = pytest.mark.unit

_USER_ID = UUID("11111111-2222-3333-4444-555555555555")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app_ctx() -> AppContext:
    """Return an AppContext with a mock pool."""
    ctx = MagicMock(spec=AppContext)
    ctx.pool = MagicMock()
    return ctx


async def _ok_handler(**kwargs: Any) -> dict[str, Any]:
    """A mock successful tool handler."""
    return {"status": "ok", "entry_id": 42}


async def _error_handler(**kwargs: Any) -> dict[str, Any]:
    """A mock failed tool handler (returns error)."""
    return {"error": "something went wrong"}


async def _ok_handler_no_target(**kwargs: Any) -> dict[str, Any]:
    """A mock successful tool handler that returns no target ID."""
    return {"status": "ok"}


async def _raising_handler(**kwargs: Any) -> dict[str, Any]:
    """A mock handler that raises an exception."""
    raise RuntimeError("handler failed")


# ---------------------------------------------------------------------------
# Successful handler triggers audit
# ---------------------------------------------------------------------------


@patch("journalctl.core.audit_decorator.current_user_id")
@patch("journalctl.core.audit_decorator.user_scoped_connection")
@patch("journalctl.core.audit_decorator.record_audit_persistence_failure")
async def test_successful_handler_triggers_audit(
    mock_persistence_failure: MagicMock,
    mock_user_scoped_conn: MagicMock,
    mock_current_user_id: MagicMock,
) -> None:
    """A successful handler should call record_audit with correct args."""
    # Arrange
    mock_current_user_id.get.return_value = _USER_ID
    mock_conn = AsyncMock()
    # user_scoped_connection is an async context manager
    mock_user_scoped_conn.return_value.__aenter__.return_value = mock_conn

    app_ctx = _make_app_ctx()

    decorated = audited("entry.created", target_type="entry", app_ctx=app_ctx)(_ok_handler)

    # Act
    with patch(
        "journalctl.core.audit_decorator.record_audit", new=AsyncMock()
    ) as mock_record_audit:
        result = await decorated(topic="test", content="hello")

    # Assert
    assert result == {"status": "ok", "entry_id": 42}
    mock_record_audit.assert_awaited_once()
    call_kwargs = mock_record_audit.call_args.kwargs
    assert call_kwargs["actor_type"] == "user"
    assert call_kwargs["actor_id"] == str(_USER_ID)
    assert call_kwargs["action"] == "entry.created"
    assert call_kwargs["target_type"] == "entry"
    assert call_kwargs["target_id"] == "42"
    mock_persistence_failure.assert_not_called()


# ---------------------------------------------------------------------------
# Failed handler does NOT trigger audit
# ---------------------------------------------------------------------------


@patch("journalctl.core.audit_decorator.current_user_id")
@patch("journalctl.core.audit_decorator.user_scoped_connection")
async def test_failed_handler_does_not_audit(
    mock_user_scoped_conn: MagicMock,
    mock_current_user_id: MagicMock,
) -> None:
    """A handler that returns an error dict should skip the audit write."""
    # Arrange
    mock_current_user_id.get.return_value = _USER_ID
    mock_conn = AsyncMock()
    mock_user_scoped_conn.return_value.__aenter__.return_value = mock_conn

    app_ctx = _make_app_ctx()

    decorated = audited("entry.created", target_type="entry", app_ctx=app_ctx)(_error_handler)

    # Act
    with patch(
        "journalctl.core.audit_decorator.record_audit", new=AsyncMock()
    ) as mock_record_audit:
        result = await decorated()

    # Assert
    assert result == {"error": "something went wrong"}
    mock_record_audit.assert_not_awaited()
    mock_record_audit.assert_not_called()


# ---------------------------------------------------------------------------
# Audit write failure is swallowed
# ---------------------------------------------------------------------------


@patch("journalctl.core.audit_decorator.current_user_id")
@patch("journalctl.core.audit_decorator.user_scoped_connection")
@patch("journalctl.core.audit_decorator.logger")
@patch("journalctl.core.audit_decorator.record_audit_persistence_failure")
async def test_audit_failure_swallowed(
    mock_persistence_failure: MagicMock,
    mock_logger: MagicMock,
    mock_user_scoped_conn: MagicMock,
    mock_current_user_id: MagicMock,
) -> None:
    """An audit write failure should log a warning and increment the counter, not raise."""
    # Arrange
    mock_current_user_id.get.return_value = _USER_ID
    mock_conn = AsyncMock()
    mock_user_scoped_conn.return_value.__aenter__.return_value = mock_conn

    app_ctx = _make_app_ctx()

    decorated = audited("entry.created", target_type="entry", app_ctx=app_ctx)(_ok_handler)

    # Act
    with patch("journalctl.core.audit_decorator.record_audit") as mock_record_audit:
        mock_record_audit.side_effect = RuntimeError("DB connection lost")
        result = await decorated(topic="test", content="hello")

    # Assert - handler result returned despite audit failure
    assert result == {"status": "ok", "entry_id": 42}
    mock_record_audit.assert_awaited_once()
    mock_logger.warning.assert_called_once()
    mock_persistence_failure.assert_called_once_with("entry.created")


# ---------------------------------------------------------------------------
# No user ID skips audit without warning
# ---------------------------------------------------------------------------


@patch("journalctl.core.audit_decorator.current_user_id")
@patch("journalctl.core.audit_decorator.logger")
@patch("journalctl.core.audit_decorator.record_audit_persistence_failure")
async def test_no_user_id_skips_audit(
    mock_persistence_failure: MagicMock,
    mock_logger: MagicMock,
    mock_current_user_id: MagicMock,
) -> None:
    """No authenticated user should skip audit write silently."""
    mock_current_user_id.get.return_value = None
    app_ctx = _make_app_ctx()
    decorated = audited("entry.created", target_type="entry", app_ctx=app_ctx)(_ok_handler)

    with patch(
        "journalctl.core.audit_decorator.record_audit", new=AsyncMock()
    ) as mock_record_audit:
        result = await decorated(topic="test", content="hello")

    assert result == {"status": "ok", "entry_id": 42}
    mock_record_audit.assert_not_called()
    mock_logger.warning.assert_not_called()
    mock_persistence_failure.assert_not_called()


# ---------------------------------------------------------------------------
# Handler exceptions propagate unchanged
# ---------------------------------------------------------------------------


@patch("journalctl.core.audit_decorator.current_user_id")
async def test_handler_exception_propagates(
    mock_current_user_id: MagicMock,
) -> None:
    """Handler exceptions must be re-raised and skip audit write."""
    mock_current_user_id.get.return_value = _USER_ID
    app_ctx = _make_app_ctx()
    decorated = audited("entry.created", target_type="entry", app_ctx=app_ctx)(_raising_handler)

    with (
        patch("journalctl.core.audit_decorator.record_audit", new=AsyncMock()) as mock_record_audit,
        pytest.raises(RuntimeError, match="handler failed"),
    ):
        await decorated(topic="test", content="hello")

    mock_record_audit.assert_not_called()


# ---------------------------------------------------------------------------
# target_id extraction
# ---------------------------------------------------------------------------


class TestExtractTargetId:
    """Verify target ID extraction from various result shapes."""

    def test_entry_id(self) -> None:
        assert _extract_target_id({"entry_id": 42}) == "42"

    def test_entry_id_str(self) -> None:
        assert _extract_target_id({"entry_id": "abc-123"}) == "abc-123"

    def test_conversation_id(self) -> None:
        assert _extract_target_id({"conversation_id": 99}) == "99"

    def test_topic(self) -> None:
        assert _extract_target_id({"topic": "work/test"}) == "work/test"

    def test_prefers_entry_id_over_topic(self) -> None:
        assert _extract_target_id({"entry_id": 1, "topic": "work"}) == "1"

    def test_prefers_conversation_id_over_topic(self) -> None:
        assert _extract_target_id({"conversation_id": 5, "topic": "work"}) == "5"

    def test_none_when_no_match(self) -> None:
        assert _extract_target_id({"status": "ok"}) is None

    def test_none_for_empty_dict(self) -> None:
        assert _extract_target_id({}) is None
