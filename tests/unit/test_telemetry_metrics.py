"""Unit tests for ``gubbi.telemetry.metrics._validate_metric_attrs``.

After S8 H2 the filter delegates to
``gubbi_common.telemetry.allowlist.is_banned_key`` so it honours
``DERIVATIVE_MODIFIERS`` (safe suffixes like ``_hash``, ``_size``,
``_len``) and ``NEVER_EXEMPT_BASES`` (credential-shaped roots like
``password``, ``api_key``).
"""

from __future__ import annotations

import pytest

from gubbi.telemetry.metrics import _validate_metric_attrs

pytestmark = pytest.mark.unit


def test_validate_metric_attrs_keeps_derivative_modifiers() -> None:
    """Derivative-suffixed keys (``_hash``, ``_size``, ``_len``) must survive.

    The pre-S8 H2 substring loop dropped these because it matched ``agent``,
    ``text`` and ``query`` as substrings of the safe suffix forms. The
    canonical ``is_banned_key`` consults ``DERIVATIVE_MODIFIERS`` so these
    keys pass through.
    """
    # Arrange
    attrs = {
        "user_agent_hash": "abc123",
        "text_hash": "def456",
        "query_size": "42",
        "text_len": "128",
        "tool.name": "journal_append_entry",
    }

    # Act
    cleaned = _validate_metric_attrs(attrs)

    # Assert
    assert cleaned == attrs


def test_validate_metric_attrs_drops_banned_keys() -> None:
    """Credential-shaped keys must be dropped even with a derivative suffix.

    ``NEVER_EXEMPT_BASES`` overrides ``DERIVATIVE_MODIFIERS`` so
    ``password_hash`` and ``api_key_hash`` are still banned even though they
    end in ``_hash``.
    """
    # Arrange
    attrs = {
        "password": "secret",
        "password_hash": "should-still-drop",
        "api_key": "sk-xxx",
        "tool.name": "safe",
    }

    # Act
    cleaned = _validate_metric_attrs(attrs)

    # Assert
    assert "password" not in cleaned
    assert "password_hash" not in cleaned
    assert "api_key" not in cleaned
    assert cleaned.get("tool.name") == "safe"


def test_validate_metric_attrs_drops_substring_content() -> None:
    """Keys containing banned substrings without derivative exemption drop.

    The banned substrings here are ``content`` (inside ``request_content``)
    and ``email`` (inside ``user_email``); the leading ``request_`` /
    ``user_`` prefixes are not what triggers the drop.
    """
    # Arrange
    attrs = {
        # "content" is the banned substring inside "request_content"
        "request_content": "raw body",
        # "email" is the banned substring inside "user_email"
        "user_email": "alice@example.com",
        "tool.name": "ok",
    }

    # Act
    cleaned = _validate_metric_attrs(attrs)

    # Assert
    assert "request_content" not in cleaned
    assert "user_email" not in cleaned
    assert cleaned.get("tool.name") == "ok"


def test_validate_metric_attrs_logs_warning_on_drop(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Operational signal: dropped keys must produce a WARNING log line."""
    # Arrange
    import logging

    caplog.set_level(logging.WARNING, logger="gubbi.telemetry.metrics")

    # Act
    _validate_metric_attrs({"password": "leak"})

    # Assert
    assert any(
        "password" in rec.getMessage() and rec.levelno == logging.WARNING for rec in caplog.records
    )


def test_validate_metric_attrs_empty_input() -> None:
    """Empty input maps to empty output without raising."""
    assert _validate_metric_attrs({}) == {}
