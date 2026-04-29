"""Tests for journalctl.core.crypto (TASK-02.11).

Covers ContentCipher round-trips, nonce uniqueness, version rotation,
construction guards, error handling, load_master_keys_from_env behaviour,
and a slow performance test.
"""

from __future__ import annotations

import base64
import random
import time

import pytest
from cryptography.exceptions import InvalidTag
from hypothesis import given, settings
from hypothesis import strategies as st

from journalctl.core.crypto import (
    ContentCipher,
    DecryptionError,
    decrypt_or_raise,
    load_master_keys_from_env,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _key(n: int) -> bytes:
    return bytes([n]) * 32


def _b64_key(n: int) -> str:
    return base64.b64encode(_key(n)).decode("ascii")


# ── 1. Round-trip decrypt == plaintext ───────────────────────────────────────


def _round_trip(plaintext: str) -> None:
    cipher = ContentCipher({1: _key(1)})
    ct, nonce = cipher.encrypt(plaintext)
    assert cipher.decrypt(ct, nonce) == plaintext


def test_round_trip_empty() -> None:
    _round_trip("")


def test_round_trip_single_char() -> None:
    _round_trip("a")


def test_round_trip_1kb_random_ascii() -> None:
    seeded = random.Random(42)  # noqa: S311 deterministic test data, not crypto use
    plaintext = "".join(chr(32 + seeded.randint(0, 94)) for _ in range(1024))
    _round_trip(plaintext)


def test_round_trip_100kb_repeated_char() -> None:
    _round_trip("x" * 100_000)


def test_round_trip_utf8_emoji() -> None:
    _round_trip("hello \U0001f44b world \u65e5\u672c\u8a9e")


def test_round_trip_null_byte() -> None:
    _round_trip("a\x00b")


# ── 2. Nonce uniqueness (50 distinct nonces AND 50 distinct ciphertexts) ────


def test_nonce_uniqueness() -> None:
    cipher = ContentCipher({1: _key(1)})
    nonces: set[bytes] = set()
    ciphertexts: set[bytes] = set()
    for _ in range(50):
        ct, nonce = cipher.encrypt("same")
        nonces.add(nonce)
        ciphertexts.add(ct)
    assert len(nonces) == 50
    assert len(ciphertexts) == 50


# ── 3. active_version == max(keys) ──────────────────────────────────────────


def test_active_version_is_max_key() -> None:
    cipher = ContentCipher({1: _key(1), 2: _key(2), 5: _key(5)})
    assert cipher.active_version == 5


def test_encrypt_nonce_active_version_byte() -> None:
    cipher = ContentCipher({1: _key(1), 2: _key(2), 5: _key(5)})
    _, nonce = cipher.encrypt("version-check")
    assert nonce[0] == 5


# ── 4. known_versions ───────────────────────────────────────────────────────


def test_known_versions() -> None:
    cipher = ContentCipher({1: _key(1), 2: _key(2), 5: _key(5)})
    assert cipher.known_versions == frozenset({1, 2, 5})


# ── 5. Tampered ciphertext / nonce → InvalidTag ─────────────────────────────


def test_decrypt_raises_on_tampered_ciphertext() -> None:
    cipher = ContentCipher({1: _key(1)})
    ct, nonce = cipher.encrypt("taint-me")
    tampered = bytearray(ct)
    tampered[0] ^= 0xFF
    with pytest.raises(InvalidTag):
        cipher.decrypt(bytes(tampered), nonce)


def test_decrypt_raises_on_tampered_nonce() -> None:
    cipher = ContentCipher({1: _key(1)})
    ct, nonce = cipher.encrypt("taint-nonce")
    tampered_nonce = bytearray(nonce)
    tampered_nonce[1] ^= 0xFF  # mutate byte 1, NOT byte 0
    with pytest.raises(InvalidTag):
        cipher.decrypt(ct, bytes(tampered_nonce))


# ── 6. Wrong key → InvalidTag ──────────────────────────────────────────────


def test_decrypt_raises_on_wrong_key() -> None:
    k_a = _key(10)
    k_b = _key(20)
    cipher_a = ContentCipher({1: k_a})
    cipher_b = ContentCipher({1: k_b})
    ct, nonce = cipher_a.encrypt("wrong-key")
    with pytest.raises(InvalidTag):
        cipher_b.decrypt(ct, nonce)


# ── 7. Unknown key version → ValueError ─────────────────────────────────────


def test_decrypt_raises_on_unknown_key_version() -> None:
    cipher = ContentCipher({1: _key(1)})
    ct, nonce = cipher.encrypt("version-check")
    bad_nonce = bytearray(nonce)
    bad_nonce[0] = 99
    with pytest.raises(ValueError, match="unknown key version"):
        cipher.decrypt(ct, bytes(bad_nonce))


# ── 8. Malformed nonce lengths → ValueError ────────────────────────────────


def test_decrypt_raises_on_empty_nonce() -> None:
    cipher = ContentCipher({1: _key(1)})
    with pytest.raises(ValueError, match="12 bytes"):
        cipher.decrypt(b"fake", b"")


def test_decrypt_raises_on_short_nonce() -> None:
    cipher = ContentCipher({1: _key(1)})
    with pytest.raises(ValueError, match="12 bytes"):
        cipher.decrypt(b"fake", b"short")


def test_decrypt_raises_on_13_byte_nonce() -> None:
    cipher = ContentCipher({1: _key(1)})
    with pytest.raises(ValueError, match="12 bytes"):
        cipher.decrypt(b"fake", b"x" * 13)


# ── 9. Non-bytes nonce → ValueError ────────────────────────────────────────


def test_decrypt_raises_on_string_nonce() -> None:
    cipher = ContentCipher({1: _key(1)})
    with pytest.raises(ValueError, match="12 bytes"):
        cipher.decrypt(b"fake", "not-bytes")  # type: ignore[arg-type]


# ── 10. Version rotation ───────────────────────────────────────────────────


def test_version_rotation_old_encrypts_decrypt_with_v2() -> None:
    k1 = _key(1)
    k2 = _key(2)
    cipher_v1 = ContentCipher({1: k1})
    cipher_v1_v2 = ContentCipher({1: k1, 2: k2})

    ct, nonce = cipher_v1.encrypt("old")
    assert nonce[0] == 1

    result = cipher_v1_v2.decrypt(ct, nonce)
    assert result == "old"


def test_version_rotation_new_encrypt_uses_highest_version() -> None:
    k1 = _key(1)
    k2 = _key(2)
    cipher_v1_v2 = ContentCipher({1: k1, 2: k2})

    ct2, nonce2 = cipher_v1_v2.encrypt("new-data")
    assert nonce2[0] == 2
    result = cipher_v1_v2.decrypt(ct2, nonce2)
    assert result == "new-data"


# ── 11. Construction guards ────────────────────────────────────────────────


def test_construct_rejects_empty_dict() -> None:
    with pytest.raises(ValueError, match="master_keys must contain at least one version"):
        ContentCipher({})


def test_construct_rejects_short_key() -> None:
    with pytest.raises(ValueError, match="32 bytes"):
        ContentCipher({1: _key(1)[:31]})


def test_construct_rejects_str_version() -> None:
    with pytest.raises(TypeError):
        ContentCipher({"1": _key(1)})  # type: ignore[dict-item]


def test_construct_rejects_bool_version() -> None:
    with pytest.raises(TypeError):
        ContentCipher({True: _key(1)})  # type: ignore[dict-item]


def test_construct_rejects_version_zero() -> None:
    with pytest.raises(ValueError, match="out of range"):
        ContentCipher({0: _key(1)})


def test_construct_rejects_version_256() -> None:
    with pytest.raises(ValueError, match="out of range"):
        ContentCipher({256: _key(1)})


def test_construct_rejects_str_key() -> None:
    with pytest.raises(TypeError):
        ContentCipher({1: "not-bytes"})  # type: ignore[dict-item]


def test_construct_accepts_bytearray_key() -> None:
    ba = bytearray(32)
    ba[0] = 0x42
    cipher = ContentCipher({1: ba})
    assert cipher.decrypt(*cipher.encrypt("bytearray-ok")) == "bytearray-ok"


# ── 12. Key isolation: mutate bytearray after construction ──────────────────


def test_key_isolation_bytearray_mutation() -> None:
    key = bytearray(b"A" * 32)
    cipher = ContentCipher({1: key})
    key[:] = b"B" * 32  # mutate after construction
    assert cipher.decrypt(*cipher.encrypt("isolated")) == "isolated"


# ── 13-18. load_master_keys_from_env ────────────────────────────────────────


def test_env_single_v1_key() -> None:
    env = {"JOURNAL_ENCRYPTION_MASTER_KEY_V1": _b64_key(1)}
    result = load_master_keys_from_env(env)
    assert result == {1: _key(1)}


def test_env_v1_and_v3_present() -> None:
    env = {
        "JOURNAL_ENCRYPTION_MASTER_KEY_V1": _b64_key(1),
        "JOURNAL_ENCRYPTION_MASTER_KEY_V3": _b64_key(3),
    }
    result = load_master_keys_from_env(env)
    assert result == {1: _key(1), 3: _key(3)}


def test_env_empty_dict() -> None:
    assert load_master_keys_from_env({}) == {}


def test_env_non_matching_keys() -> None:
    env = {"FOO": "bar", "JOURNAL_OTHER": "baz", "JOURNAL_ENCRYPTION_OTHER_V1": _b64_key(1)}
    assert load_master_keys_from_env(env) == {}


def test_env_invalid_base64() -> None:
    env = {"JOURNAL_ENCRYPTION_MASTER_KEY_V1": "not-valid-base64!"}
    with pytest.raises(ValueError, match="invalid base64"):
        load_master_keys_from_env(env)


def test_env_decoded_wrong_length_24() -> None:
    short = base64.b64encode(b"x" * 24).decode("ascii")
    env = {"JOURNAL_ENCRYPTION_MASTER_KEY_V1": short}
    with pytest.raises(ValueError, match="32 bytes"):
        load_master_keys_from_env(env)


def test_env_decoded_wrong_length_48() -> None:
    long_key = base64.b64encode(b"x" * 48).decode("ascii")
    env = {"JOURNAL_ENCRYPTION_MASTER_KEY_V1": long_key}
    with pytest.raises(ValueError, match="32 bytes"):
        load_master_keys_from_env(env)


def test_env_version_0_out_of_range() -> None:
    env = {"JOURNAL_ENCRYPTION_MASTER_KEY_V0": _b64_key(1)}
    with pytest.raises(ValueError, match="out of range"):
        load_master_keys_from_env(env)


def test_env_version_256_out_of_range() -> None:
    env = {"JOURNAL_ENCRYPTION_MASTER_KEY_V256": _b64_key(1)}
    with pytest.raises(ValueError, match="out of range"):
        load_master_keys_from_env(env)


def test_env_rejects_leading_zero_version() -> None:
    env = {"JOURNAL_ENCRYPTION_MASTER_KEY_V01": _b64_key(1)}
    with pytest.raises(ValueError, match="leading zeros"):
        load_master_keys_from_env(env)


def test_env_pattern_rejects_similar_names() -> None:
    env = {
        "JOURNAL_ENCRYPTION_MASTER_KEY_V1A": _b64_key(1),
        "JOURNAL_ENCRYPTION_MASTER_KEY_V": _b64_key(1),
        "JOURNAL_ENCRYPTION_MASTER_KEY_Vfoo": _b64_key(1),
    }
    assert load_master_keys_from_env(env) == {}


# ── 19. Performance (1 MB round-trip < 100 ms) ─────────────────────────────


@pytest.mark.slow
def test_encrypt_decrypt_1mb_under_100ms() -> None:
    cipher = ContentCipher({1: _key(1)})
    plaintext = "x" * (1024 * 1024)
    t0 = time.perf_counter()
    ct, nonce = cipher.encrypt(plaintext)
    result = cipher.decrypt(ct, nonce)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.1, f"encrypt+decrypt took {elapsed:.3f}s"
    assert result == plaintext


# -- DecryptionError + decrypt_or_raise wrap (TASK-02.13) --


def test_decrypt_or_raise_happy_path() -> None:
    cipher = ContentCipher({1: _key(1)})
    ct, nonce = cipher.encrypt("hello")
    assert decrypt_or_raise(cipher, ct, nonce) == "hello"


def test_decrypt_or_raise_flattens_invalidtag() -> None:
    cipher = ContentCipher({1: _key(1)})
    ct, nonce = cipher.encrypt("x")
    tampered = bytearray(ct)
    tampered[0] ^= 0xFF
    with pytest.raises(DecryptionError) as exc_info:
        decrypt_or_raise(cipher, bytes(tampered), nonce)
    assert isinstance(exc_info.value.__cause__, InvalidTag)


def test_decrypt_or_raise_flattens_valueerror_unknown_version() -> None:
    cipher = ContentCipher({1: _key(1)})
    ct, nonce = cipher.encrypt("x")
    mutation = bytearray(nonce)
    mutation[0] = 99
    with pytest.raises(DecryptionError) as exc_info:
        decrypt_or_raise(cipher, ct, bytes(mutation))
    assert isinstance(exc_info.value.__cause__, ValueError)


def test_decrypt_or_raise_flattens_valueerror_malformed_nonce() -> None:
    cipher = ContentCipher({1: _key(1)})
    ct, nonce = cipher.encrypt("x")
    truncated = nonce[:11]
    with pytest.raises(DecryptionError) as exc_info:
        decrypt_or_raise(cipher, ct, truncated)
    assert isinstance(exc_info.value.__cause__, ValueError)


def test_decrypt_or_raise_empty_string_round_trip() -> None:
    cipher = ContentCipher({1: _key(1)})
    ct, nonce = cipher.encrypt("")
    assert decrypt_or_raise(cipher, ct, nonce) == ""


def test_decrypt_or_raise_does_not_swallow_type_error(monkeypatch: pytest.MonkeyPatch) -> None:
    cipher = ContentCipher({1: _key(1)})
    monkeypatch.setattr(cipher, "decrypt", lambda *a, **k: (_ for _ in ()).throw(TypeError("oops")))
    with pytest.raises(TypeError, match="oops"):
        decrypt_or_raise(cipher, b"ct", b"nonce")


# -- hypothesis property tests (TASK-02.16) --


@settings(max_examples=100, deadline=None)
@given(plaintext=st.text())
def test_hypothesis_round_trip_any_text(plaintext: str) -> None:
    """decrypt(encrypt(x)) == x for any valid str."""
    cipher = ContentCipher({1: _key(1)})
    ct, nonce = cipher.encrypt(plaintext)
    assert cipher.decrypt(ct, nonce) == plaintext


@settings(max_examples=100, deadline=None)
@given(plaintext=st.text(min_size=1), nonce_byte=st.integers(min_value=1, max_value=11))
def test_hypothesis_nonce_byte_tamper_raises_invalid_tag(plaintext: str, nonce_byte: int) -> None:
    """Flipping any non-version byte of the nonce fails the GCM auth tag.

    Indices 1..11 are the 11 CSPRNG bytes (nonce is 12 bytes, byte 0 is
    the key version). Mutating byte 0 would trigger an "unknown version"
    ValueError path, which is covered by hand-written tests; this
    property focuses on the auth-tag behaviour.
    """
    cipher = ContentCipher({1: _key(1)})
    ct, nonce = cipher.encrypt(plaintext)
    tampered = bytearray(nonce)
    tampered[nonce_byte] ^= 0xFF
    with pytest.raises(InvalidTag):
        cipher.decrypt(ct, bytes(tampered))


@settings(max_examples=100, deadline=None)
@given(plaintext=st.text(min_size=1), byte_offset=st.integers(min_value=0))
def test_hypothesis_ciphertext_byte_tamper_raises_invalid_tag(
    plaintext: str, byte_offset: int
) -> None:
    """Flipping any ciphertext byte (including the GCM auth tag) raises InvalidTag.

    ``byte_offset`` is modulo-reduced to a valid index so hypothesis can
    explore the full range without shrinking constraints every iteration.
    The final 16 bytes of the ciphertext are the GCM auth tag; flipping
    anywhere in the ciphertext (data OR tag) must fail decryption.
    """
    cipher = ContentCipher({1: _key(1)})
    ct, nonce = cipher.encrypt(plaintext)
    idx = byte_offset % len(ct)
    tampered = ct[:idx] + bytes([ct[idx] ^ 0xFF]) + ct[idx + 1 :]
    with pytest.raises(InvalidTag):
        cipher.decrypt(tampered, nonce)


# ── A. encrypt_with_version tests (TASK-03.13 Approach A) ────────────────────


def test_encrypt_with_version_valid_version_round_trip() -> None:
    """Encrypting at v1 with a {1,2}-key cipher round-trips under v1 nonce."""
    keys = {1: _key(1), 2: _key(2)}
    ct, nonce = ContentCipher(keys).encrypt_with_version("vx-test", version=1)
    assert nonce[0] == 1
    assert ContentCipher(keys).decrypt(ct, nonce) == "vx-test"


def test_encrypt_with_version_non_default_version_nonce_byte() -> None:
    """encrypt_with_version(V2) produces nonce byte-0 == 2 even if active is different."""
    keys = {1: _key(1), 2: _key(2), 5: _key(5)}
    ct_v2, nonce_v2 = ContentCipher(keys).encrypt_with_version("v2-only", version=2)
    assert nonce_v2[0] == 2

    ct_v1, nonce_v1 = ContentCipher(keys).encrypt_with_version("v1-specific", version=1)
    assert nonce_v1[0] == 1


def test_encrypt_with_version_non_default_is_independent_of_active() -> None:
    """The active_version property does not influence encrypt_with_version output."""
    keys = {1: _key(1), 2: _key(2)}
    cipher = ContentCipher(keys)
    assert cipher.active_version == 2

    ct, nonce = cipher.encrypt("active-writes-v2")
    assert nonce[0] == 2

    # But encrypt_with_version(1) still targets V1.
    ct_v1, nonce_v1 = cipher.encrypt_with_version("explicit-v1", version=1)
    assert nonce_v1[0] == 1


def test_encrypt_with_version_unknown_version_raises_value_error() -> None:
    """Passing an unknown version raises ValueError."""
    keys = {1: _key(1), 2: _key(2)}
    with pytest.raises(ValueError, match="not known"):
        ContentCipher(keys).encrypt_with_version("bad", version=99)


def test_encrypt_with_version_missing_from_known_versions_raises() -> None:
    """Specifying a key that is not in the cipher's known set must fail."""
    keys = {1: _key(1), 2: _key(2)}
    with pytest.raises(ValueError, match="not known"):
        ContentCipher(keys).encrypt_with_version("no-such-key", version=3)


def test_encrypt_with_version_zero_raises_value_error() -> None:
    """Version 0 is below the allowed range."""
    keys = {1: _key(1)}
    with pytest.raises(ValueError, match="not known"):
        ContentCipher(keys).encrypt_with_version("zero", version=0)


def test_encrypt_with_version_256_raises_value_error() -> None:
    """Version 256 is above the allowed range."""
    keys = {1: _key(1)}
    with pytest.raises(ValueError, match="not known"):
        ContentCipher(keys).encrypt_with_version("too-high", version=256)
