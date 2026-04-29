"""Unit tests for ``journalctl.scripts.rotate_encryption_key``.

Pure-function coverage:
- ``_verify_sample_size`` bounds for the verify path
- Cryptographic round-trip through a two-version cipher (V1 -> V2)
- Module-level constants are well-formed

Behavioural coverage of the rotation loop (skip-already-at-target, idempotent
re-runs, --verify against gappy IDENTITY sequences) lives in
``tests/integration/test_rotate_encryption_key.py`` because it requires a
real database to be meaningful.
"""

from __future__ import annotations

from journalctl.core.crypto import ContentCipher
from journalctl.scripts.rotate_encryption_key import (
    _AUDIT_ACTION,
    _ROTATION_SCREENS,
    _VERIFY_SAMPLE_MAX,
    _VERIFY_SAMPLE_MIN,
    _verify_sample_size,
)


def _key(v: int) -> bytes:
    """Deterministic V<N> key for tests (not real crypto)."""
    return bytes([v]) * 32


# ---------------------------------------------------------------------------
# 1. Decrypt -> re-encrypt round trip via dual-version cipher
# ---------------------------------------------------------------------------


def test_decrypt_re_encrypt_v1_to_v2_keyed_cipher() -> None:
    """Plaintext encrypted with V1 can be decrypted and re-encrypted under V2."""
    cipher = ContentCipher({1: _key(1), 2: _key(2)})

    plaintext = "this is a test entry content"

    # Encrypt as if it were originally at V1.
    ct_v1, nonce_v1 = ContentCipher({1: _key(1)}).encrypt_with_version(plaintext, version=1)
    assert nonce_v1[0] == 1

    # Decrypt under dual-key cipher and re-encrypt as V2.
    decrypted = cipher.decrypt(ct_v1, nonce_v1)
    assert decrypted == plaintext

    ct_v2, nonce_v2 = cipher.encrypt_with_version(decrypted, version=2)
    assert nonce_v2[0] == 2

    # Round-trip under V2 nonce.
    final = cipher.decrypt(ct_v2, nonce_v2)
    assert final == plaintext


# ---------------------------------------------------------------------------
# 2. _verify_sample_size bounds
# ---------------------------------------------------------------------------


def test_verify_sample_size_zero_total() -> None:
    assert _verify_sample_size(0) == 0


def test_verify_sample_size_below_min_floor() -> None:
    # 5 rows -> 1% rounds to 1, floor to MIN.
    assert _verify_sample_size(5) == _VERIFY_SAMPLE_MIN


def test_verify_sample_size_at_min_returns_min() -> None:
    # 1000 rows -> 1% = 10 = MIN exactly.
    assert _verify_sample_size(1000) == _VERIFY_SAMPLE_MIN


def test_verify_sample_size_at_one_percent() -> None:
    # 50_000 rows -> 1% = 500. Within [MIN, MAX] -> returned as-is.
    assert _verify_sample_size(50_000) == 500


def test_verify_sample_size_above_max_caps() -> None:
    # 1M rows -> 1% = 10_000 -> caps to MAX.
    assert _verify_sample_size(1_000_000) == _VERIFY_SAMPLE_MAX


# ---------------------------------------------------------------------------
# 3. Module constants
# ---------------------------------------------------------------------------


def test_audit_action_is_encryption_key_rotated() -> None:
    """The audit action constant must be the canonical 'encryption.key_rotated' string."""
    assert _AUDIT_ACTION == "encryption.key_rotated"


def test_rotation_screens_is_the_documented_five_pairs() -> None:
    """Exactly five (table, col_encrypted, col_nonce) tuples in the documented order."""
    assert _ROTATION_SCREENS == [
        ("entries", "content_encrypted", "content_nonce"),
        ("entries", "reasoning_encrypted", "reasoning_nonce"),
        ("messages", "content_encrypted", "content_nonce"),
        ("conversations", "title_encrypted", "title_nonce"),
        ("conversations", "summary_encrypted", "summary_nonce"),
    ]
