"""Compatibility shim -- crypto package."""

from __future__ import annotations

import warnings
from typing import Any


def __getattr__(name: str) -> Any:
    if name == "ContentCipher":
        warnings.warn(
            "gubbi.core.crypto.ContentCipher is deprecated; "
            "import from gubbi.crypto.ContentCipher instead",
            DeprecationWarning,
            stacklevel=2,
        )
        from gubbi.crypto.cipher import ContentCipher

        return ContentCipher
    if name == "load_master_keys_from_env":
        warnings.warn(
            "gubbi.core.crypto.load_master_keys_from_env is deprecated; "
            "import from gubbi.crypto.load_master_keys_from_env instead",
            DeprecationWarning,
            stacklevel=2,
        )
        from gubbi.crypto.cipher import load_master_keys_from_env

        return load_master_keys_from_env
    if name == "DecryptionError":
        warnings.warn(
            "gubbi.core.crypto.DecryptionError is deprecated; "
            "import from gubbi.crypto.DecryptionError instead",
            DeprecationWarning,
            stacklevel=2,
        )
        from gubbi.crypto.cipher import DecryptionError

        return DecryptionError
    if name == "decrypt_or_raise":
        warnings.warn(
            "gubbi.core.crypto.decrypt_or_raise is deprecated; "
            "import from gubbi.crypto.decrypt_or_raise instead",
            DeprecationWarning,
            stacklevel=2,
        )
        from gubbi.crypto.cipher import decrypt_or_raise

        return decrypt_or_raise
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
