"""Conversation reads honour the soft-fail decrypt contract on the web surface.

The repository's ``_row_to_meta`` and ``_message_from_row`` raise loudly on a
corrupt row when ``soft_fail=False`` (the MCP-tool default) but surface the
``[decryption failed]`` sentinel + ``decryption_failed=True`` when
``soft_fail=True`` (the web read path). A half-NULL ciphertext/nonce column pair
is the corruption signal exercised here -- no key management needed.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from gubbi.crypto.cipher import ContentCipher, DecryptionError
from gubbi.models.conversation import Message
from gubbi.storage.repositories import conversations as conv_repo

_CIPHER = ContentCipher({1: bytes([7]) * 32})
_NOW = datetime(2026, 6, 1, tzinfo=UTC)


def _meta_row(*, corrupt: bool) -> dict[str, object]:
    """A conversations row. ``corrupt`` half-NULLs the title column pair."""
    title_ct, title_nonce = _CIPHER.encrypt("My Conversation")
    summary_ct, summary_nonce = _CIPHER.encrypt("A summary")
    return {
        "id": 7,
        "source": "claude",
        "topic": "work/notes",
        "tags": [],
        "created_at": _NOW,
        "updated_at": _NOW,
        "participants": [],
        "message_count": 1,
        "title_encrypted": title_ct,
        # Half-NULL nonce is the corruption signal.
        "title_nonce": None if corrupt else title_nonce,
        "summary_encrypted": summary_ct,
        "summary_nonce": summary_nonce,
    }


def _message_row(*, corrupt: bool) -> dict[str, object]:
    """A messages row. ``corrupt`` half-NULLs the content column pair."""
    content_ct, content_nonce = _CIPHER.encrypt("hello there")
    return {
        "role": "user",
        "content_encrypted": content_ct,
        "content_nonce": None if corrupt else content_nonce,
        "timestamp": None,
    }


def test_meta_soft_fail_surfaces_sentinel_not_raise() -> None:
    """soft_fail=True: a corrupt title yields the sentinel + decryption_failed."""
    # Arrange
    row = _meta_row(corrupt=True)

    # Act
    meta = conv_repo._row_to_meta(_CIPHER, row, soft_fail=True)  # type: ignore[arg-type]

    # Assert
    assert meta.decryption_failed is True
    assert meta.title == conv_repo.DECRYPTION_FAILED_SENTINEL
    # Summary decrypted fine, so it is plaintext.
    assert meta.summary == "A summary"


def test_meta_loud_path_raises_on_corruption() -> None:
    """soft_fail=False (default, MCP path): a corrupt title raises."""
    # Arrange
    row = _meta_row(corrupt=True)

    # Act / Assert
    with pytest.raises(DecryptionError):
        conv_repo._row_to_meta(_CIPHER, row)  # type: ignore[arg-type]


def test_meta_happy_path_decrypts() -> None:
    """A clean row decrypts with decryption_failed False on both paths."""
    # Arrange
    row = _meta_row(corrupt=False)

    # Act
    soft = conv_repo._row_to_meta(_CIPHER, row, soft_fail=True)  # type: ignore[arg-type]
    loud = conv_repo._row_to_meta(_CIPHER, row)  # type: ignore[arg-type]

    # Assert
    assert soft.title == loud.title == "My Conversation"
    assert soft.decryption_failed is False


def test_message_soft_fail_surfaces_sentinel() -> None:
    """soft_fail=True: a corrupt message yields the sentinel + flag."""
    # Arrange
    row = _message_row(corrupt=True)

    # Act
    message: Message = conv_repo._message_from_row(_CIPHER, row, soft_fail=True)

    # Assert
    assert message.decryption_failed is True
    assert message.content == conv_repo.DECRYPTION_FAILED_SENTINEL


def test_message_loud_path_raises() -> None:
    """soft_fail=False: a corrupt message raises."""
    # Arrange
    row = _message_row(corrupt=True)

    # Act / Assert
    with pytest.raises(DecryptionError):
        conv_repo._message_from_row(_CIPHER, row, soft_fail=False)


def test_message_happy_path_decrypts() -> None:
    """A clean message decrypts with decryption_failed False."""
    # Arrange
    row = _message_row(corrupt=False)

    # Act
    message = conv_repo._message_from_row(_CIPHER, row, soft_fail=True)

    # Assert
    assert message.content == "hello there"
    assert message.decryption_failed is False
