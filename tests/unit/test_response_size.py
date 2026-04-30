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
