"""Single shared decryption mapper for the web REST endpoints.

Every endpoint that returns a decrypted column routes through
:func:`decrypt_field` so decryption robustness lives in exactly one place: a
single corrupted row yields the ``[decryption failed]`` sentinel and a
``decryption_failed=True`` flag rather than raising and failing the whole
response. Sibling routers must NOT re-implement this -- reuse this helper so
the failure semantics stay identical across resources.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from gubbi.crypto.cipher import DecryptionError, decrypt_content_field

if TYPE_CHECKING:
    from collections.abc import Mapping

    import asyncpg

    from gubbi.crypto.cipher import ContentCipher

__all__: list[str] = [
    "DECRYPTION_FAILED_SENTINEL",
    "decrypt_field",
]

# Surfaced in place of plaintext when a row cannot be decrypted. Paired with
# the ``decryption_failed`` flag so clients can render a placeholder without
# inspecting the string.
DECRYPTION_FAILED_SENTINEL: str = "[decryption failed]"


def decrypt_field(
    cipher: ContentCipher,
    row: asyncpg.Record | Mapping[str, Any],
    encrypted_key: str,
    nonce_key: str,
) -> tuple[str | None, bool]:
    """Decrypt one ciphertext/nonce column pair, never raising on bad data.

    Wraps :func:`gubbi.storage.repositories.entries._decrypt_content_field`,
    which returns ``None`` when both columns are NULL (a legitimate "no value",
    e.g. an entry with no reasoning) and raises :class:`DecryptionError` when a
    stored value cannot be decrypted or the column/nonce pair is half-present.

    Returns:
        ``(value, decryption_failed)`` where:

        * ``(plaintext, False)`` -- decrypted successfully.
        * ``(None, False)`` -- column legitimately has no value (both NULL).
        * ``(DECRYPTION_FAILED_SENTINEL, True)`` -- decryption failed; the
          caller surfaces the sentinel and sets its item's ``decryption_failed``
          flag. Never a 500 for one bad row.
    """
    try:
        return decrypt_content_field(cipher, row, encrypted_key, nonce_key), False
    except DecryptionError:
        return DECRYPTION_FAILED_SENTINEL, True
