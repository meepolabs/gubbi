"""Tests for journalctl.core.cipher_guard (TASK-02.13)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from journalctl.core.cipher_guard import require_cipher
from journalctl.core.crypto import ContentCipher


def _make_cipher() -> ContentCipher:
    return ContentCipher({1: bytes([1]) * 32})


def test_require_cipher_returns_instance_when_present() -> None:
    cipher = _make_cipher()
    ctx = SimpleNamespace(cipher=cipher)
    assert require_cipher(ctx) is cipher


def test_require_cipher_raises_runtime_error_when_none() -> None:
    ctx = SimpleNamespace(cipher=None)
    with pytest.raises(RuntimeError) as exc_info:
        require_cipher(ctx)
    assert "JOURNAL_ENCRYPTION_MASTER_KEY_V1" in str(exc_info.value)
    assert "encryption cipher required" in str(exc_info.value)


def test_require_cipher_does_not_catch_attribute_error() -> None:
    class NoCipherAttr:
        pass

    with pytest.raises(AttributeError):
        require_cipher(NoCipherAttr())  # type: ignore[arg-type]


def test_require_cipher_message_is_operator_oriented() -> None:
    ctx = SimpleNamespace(cipher=None)
    with pytest.raises(RuntimeError) as exc_info:
        require_cipher(ctx)
    msg = str(exc_info.value)
    assert "master key" not in msg.lower() or "V1" in msg
