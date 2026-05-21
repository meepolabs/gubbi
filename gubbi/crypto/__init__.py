"""Crypto package -- application-layer AES-256-GCM content encryption."""

from __future__ import annotations

# Re-export from cipher module
from gubbi.crypto.cipher import (
    ContentCipher,
    DecryptionError,
    decrypt_or_raise,
    load_master_keys_from_env,
)
from gubbi.crypto.guard import require_cipher

__all__: list[str] = [
    "ContentCipher",
    "DecryptionError",
    "decrypt_or_raise",
    "load_master_keys_from_env",
    "require_cipher",
]
