"""Unit tests for response-size safety net (TASK-03.23)."""

import pytest

from journalctl.tools._response_size import _assert_response_ok

pytestmark = pytest.mark.unit


def test_ok_payload_returns_none() -> None:
    result = _assert_response_ok({"entries": [{"id": "1"}]})
    assert result is None


def test_large_payload_returns_error_dict() -> None:
    huge = {"entries": [{"content": "x" * 100_000}]}
    result = _assert_response_ok(huge)
    assert result is not None
    assert result["error"] == "response_too_large"
    assert "current_size_chars" in result
    assert "suggested_max_limit" in result


def test_custom_threshold() -> None:
    payload = {"x": "a" * 200}
    assert _assert_response_ok(payload, error_threshold_chars=100) is not None
    assert _assert_response_ok(payload, error_threshold_chars=10_000) is None


def test_unserializable_passes_through() -> None:
    class NotSerializable:
        pass

    # json.dumps raises TypeError for non-serializable objects
    result = _assert_response_ok({"obj": NotSerializable()})
    assert result is None


def test_simple_payload_fits_threshold() -> None:
    small = {"msg": "hello", "count": 42}
    result = _assert_response_ok(small)
    assert result is None


# ---------------------------------------------------------------------------
# Conversations-shaped oversized payload
# ---------------------------------------------------------------------------


def test_conversations_too_large_returns_error() -> None:
    """Verify guard fires for an oversized journal_list_conversations response."""
    huge = {
        "conversations": [
            {
                "id": 1,
                "title": "x",
                "summary": "y" * 100_000,
                "summary_truncated": False,
            }
        ],
        "total": 1,
        "limit": 20,
        "offset": 0,
    }
    result = _assert_response_ok(huge, tool_name="journal_list_conversations")
    assert result is not None
    assert result["error"] == "response_too_large"
    assert "current_size_chars" in result
    assert "suggested_max_limit" in result


def test_conversations_normal_fits() -> None:
    small = {
        "conversations": [{"id": 1, "title": "x", "summary": "y", "summary_truncated": False}],
        "total": 1,
        "limit": 20,
        "offset": 0,
    }
    result = _assert_response_ok(small, tool_name="journal_list_conversations")
    assert result is None


# ---------------------------------------------------------------------------
# Topics-shaped oversized payload
# ---------------------------------------------------------------------------


def test_topics_too_large_returns_error() -> None:
    """Verify guard fires for an oversized journal_list_topics response."""
    huge = {
        "topics": [
            {
                "topic": "x/y",
                "title": "X" * 100_000,
                "description": "desc",
            }
        ],
        "total": 1,
        "offset": 0,
        "limit": 20,
    }
    result = _assert_response_ok(huge, tool_name="journal_list_topics")
    assert result is not None
    assert result["error"] == "response_too_large"
    assert "current_size_chars" in result
    assert "suggested_max_limit" in result


def test_topics_normal_fits() -> None:
    small = {
        "topics": [{"topic": "x/y", "title": "X", "description": "desc"}],
        "total": 1,
        "offset": 0,
        "limit": 20,
    }
    result = _assert_response_ok(small, tool_name="journal_list_topics")
    assert result is None
