"""Unit tests for the @audited decorator (M3 requirement).

Tests cover:
    - Successful handler triggers an audit write (with target_kind)
    - Failed handler (returns dict with "error" key converted to "success": False) does NOT trigger
    - CallToolResult with isError=True triggers failure path
    - CallToolResult with isError=False triggers success path
    - Audit write failure is swallowed (handler result returned, warning logged)
    - No user ID skips audit silently
    - Handler exceptions propagate unchanged
    - target_id extraction returns (target_id, target_kind) tuple
    - caller-supplied target_kind takes precedence over derived kind
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

from gubbi.core.audit_decorator import _extract_target_id, _result_is_success, audited
from gubbi.core.context import AppContext

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
    """A mock failed tool handler (returns with success=False)."""
    return {"status": "error", "success": False, "error": "something went wrong"}


async def _ok_handler_no_target(**kwargs: Any) -> dict[str, Any]:
    """A mock successful tool handler that returns no target ID."""
    return {"status": "ok"}


async def _raising_handler(**kwargs: Any) -> dict[str, Any]:
    """A mock handler that raises an exception."""
    raise RuntimeError("handler failed")


# ---------------------------------------------------------------------------
# Successful handler triggers audit
# ---------------------------------------------------------------------------


@patch("gubbi.core.audit_decorator.current_user_id")
@patch("gubbi.core.audit_decorator.user_scoped_connection")
@patch("gubbi.core.audit_decorator.record_audit_persistence_failure")
async def test_successful_handler_triggers_audit(
    mock_persistence_failure: MagicMock,
    mock_user_scoped_conn: MagicMock,
    mock_current_user_id: MagicMock,
) -> None:
    """A successful handler should call record_audit with correct args, including target_kind."""
    # Arrange
    mock_current_user_id.get.return_value = _USER_ID
    mock_conn = AsyncMock()
    mock_user_scoped_conn.return_value.__aenter__.return_value = mock_conn

    app_ctx = _make_app_ctx()

    decorated = audited("entry.created", target_type="entry", target_kind="entry", app_ctx=app_ctx)(
        _ok_handler
    )

    # Act
    with patch("gubbi.core.audit_decorator.record_audit", new=AsyncMock()) as mock_record_audit:
        result = await decorated(topic="test", content="hello")

    # Assert
    assert result == {"status": "ok", "entry_id": 42}
    mock_record_audit.assert_awaited_once()
    call_kwargs = mock_record_audit.call_args.kwargs
    assert call_kwargs["actor_type"] == "user"
    assert call_kwargs["actor_id"] == str(_USER_ID)
    assert call_kwargs["action"] == "entry.created"
    assert call_kwargs["target_type"] == "entry"
    assert call_kwargs["target_kind"] == "entry"
    assert call_kwargs["target_id"] == "42"
    mock_persistence_failure.assert_not_called()


# ---------------------------------------------------------------------------
# Failed handler (dict with success=False) does NOT trigger audit
# ---------------------------------------------------------------------------


@patch("gubbi.core.audit_decorator.current_user_id")
@patch("gubbi.core.audit_decorator.user_scoped_connection")
async def test_failed_handler_does_not_audit(
    mock_user_scoped_conn: MagicMock,
    mock_current_user_id: MagicMock,
) -> None:
    """A handler that returns success=False should skip the audit write."""
    # Arrange
    mock_current_user_id.get.return_value = _USER_ID
    mock_conn = AsyncMock()
    mock_user_scoped_conn.return_value.__aenter__.return_value = mock_conn

    app_ctx = _make_app_ctx()

    decorated = audited("entry.created", target_type="entry", target_kind="entry", app_ctx=app_ctx)(
        _error_handler
    )

    # Act
    with patch("gubbi.core.audit_decorator.record_audit", new=AsyncMock()) as mock_record_audit:
        result = await decorated()

    # Assert
    assert result == {"status": "error", "success": False, "error": "something went wrong"}
    mock_record_audit.assert_not_awaited()
    mock_record_audit.assert_not_called()


# ---------------------------------------------------------------------------
# Audit write failure is swallowed
# ---------------------------------------------------------------------------


@patch("gubbi.core.audit_decorator.current_user_id")
@patch("gubbi.core.audit_decorator.user_scoped_connection")
@patch("gubbi.core.audit_decorator.logger")
@patch("gubbi.core.audit_decorator.record_audit_persistence_failure")
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

    decorated = audited("entry.created", target_type="entry", target_kind="entry", app_ctx=app_ctx)(
        _ok_handler
    )

    # Act
    with patch("gubbi.core.audit_decorator.record_audit") as mock_record_audit:
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


@patch("gubbi.core.audit_decorator.current_user_id")
@patch("gubbi.core.audit_decorator.logger")
@patch("gubbi.core.audit_decorator.record_audit_persistence_failure")
async def test_no_user_id_skips_audit(
    mock_persistence_failure: MagicMock,
    mock_logger: MagicMock,
    mock_current_user_id: MagicMock,
) -> None:
    """No authenticated user should skip audit write silently."""
    mock_current_user_id.get.return_value = None
    app_ctx = _make_app_ctx()
    decorated = audited("entry.created", target_type="entry", target_kind="entry", app_ctx=app_ctx)(
        _ok_handler
    )

    with patch("gubbi.core.audit_decorator.record_audit", new=AsyncMock()) as mock_record_audit:
        result = await decorated(topic="test", content="hello")

    assert result == {"status": "ok", "entry_id": 42}
    mock_record_audit.assert_not_called()
    mock_logger.warning.assert_not_called()
    mock_persistence_failure.assert_not_called()


# ---------------------------------------------------------------------------
# Handler exceptions propagate unchanged
# ---------------------------------------------------------------------------


@patch("gubbi.core.audit_decorator.current_user_id")
async def test_handler_exception_propagates(
    mock_current_user_id: MagicMock,
) -> None:
    """Handler exceptions must be re-raised and skip audit write."""
    mock_current_user_id.get.return_value = _USER_ID
    app_ctx = _make_app_ctx()
    decorated = audited("entry.created", target_type="entry", target_kind="entry", app_ctx=app_ctx)(
        _raising_handler
    )

    with (
        patch("gubbi.core.audit_decorator.record_audit", new=AsyncMock()) as mock_record_audit,
        pytest.raises(RuntimeError, match="handler failed"),
    ):
        await decorated(topic="test", content="hello")

    mock_record_audit.assert_not_called()


# ---------------------------------------------------------------------------
# _result_is_success — CallToolResult vs. dict
# ---------------------------------------------------------------------------


class MockCallToolResult:
    """Minimal stand-in for mcp.types.CallToolResult so we don't need the real import."""

    def __init__(self, isError: bool = False) -> None:  # noqa: N803
        self.isError = isError


class TestResultIsSuccess:
    """Verify success heuristic for both CallToolResult and dict."""

    def test_call_tool_result_is_error_true(self) -> None:
        assert _result_is_success(MockCallToolResult(isError=True)) is False

    def test_call_tool_result_is_error_false(self) -> None:
        assert _result_is_success(MockCallToolResult(isError=False)) is True

    def test_dict_success_true(self) -> None:
        assert _result_is_success({"success": True}) is True

    def test_dict_success_false(self) -> None:
        assert _result_is_success({"success": False}) is False

    def test_dict_no_success_key(self) -> None:
        """A dict without a 'success' key is assumed successful."""
        assert _result_is_success({"status": "ok"}) is True

    def test_dict_empty(self) -> None:
        assert _result_is_success({}) is True

    def test_dict_success_none(self) -> None:
        """Edge case: success=None should be treated as falsy."""
        assert _result_is_success({"success": None}) is False


# ---------------------------------------------------------------------------
# target_id extraction — now returns (target_id, target_kind) tuple
# ---------------------------------------------------------------------------


class TestExtractTargetId:
    """Verify target ID extraction from various result shapes."""

    def test_entry_id(self) -> None:
        assert _extract_target_id({"entry_id": 42}, "entry") == ("42", "entry")

    def test_entry_id_str(self) -> None:
        assert _extract_target_id({"entry_id": "abc-123"}, "entry") == ("abc-123", "entry")

    def test_conversation_id(self) -> None:
        assert _extract_target_id({"conversation_id": 99}, "conversation") == ("99", "conversation")

    def test_topic(self) -> None:
        assert _extract_target_id({"topic": "work/test"}, "topic") == ("work/test", "topic")

    def test_entry_type_picks_entry_id_when_both_present(self) -> None:
        """target_type=entry picks entry_id even when topic is also in result."""
        assert _extract_target_id({"entry_id": 1, "topic": "work"}, "entry") == ("1", "entry")

    def test_topic_type_picks_topic_when_both_present(self) -> None:
        """target_type=topic picks topic even when entry_id is also in result."""
        assert _extract_target_id({"entry_id": 1, "topic": "work"}, "topic") == ("work", "topic")

    def test_conversation_type_picks_conversation_id(self) -> None:
        """target_type=conversation picks conversation_id."""
        assert _extract_target_id({"conversation_id": 5}, "conversation") == ("5", "conversation")

    def test_unknown_type_returns_none_none(self) -> None:
        """target_type not in _TARGET_KEYS returns (None, None)."""
        assert _extract_target_id({"entry_id": 1, "topic": "work"}, "unknown_type") == (None, None)

    def test_none_when_no_match(self) -> None:
        assert _extract_target_id({"status": "ok"}, "entry") == (None, None)

    def test_none_for_empty_dict(self) -> None:
        assert _extract_target_id({}, "entry") == (None, None)


class TestAuditedTargetTypeWarning:
    """Verify @audited warns at decoration time for unmapped target_type."""

    def test_unmapped_target_type_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Decorating with target_type not in _TARGET_KEYS logs a warning."""
        ctx = _make_app_ctx()
        with caplog.at_level("WARNING", logger="gubbi.core.audit_decorator"):

            @audited("user.flagged", target_type="user", app_ctx=ctx)
            async def _handler() -> dict[str, Any]:
                return {"success": True}

        assert any(
            "target_type='user'" in r.message and "no _TARGET_KEYS entry" in r.message
            for r in caplog.records
        ), f"Expected warning for unmapped target_type; got: {[r.message for r in caplog.records]}"

    def test_mapped_target_type_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Decorating with a mapped target_type does NOT warn."""
        ctx = _make_app_ctx()
        with caplog.at_level("WARNING", logger="gubbi.core.audit_decorator"):

            @audited("entry.created", target_type="entry", app_ctx=ctx)
            async def _handler() -> dict[str, Any]:
                return {"success": True}

        assert not any(
            "no _TARGET_KEYS entry" in r.message for r in caplog.records
        ), "Mapped target_type should not log unmapped-warning"
