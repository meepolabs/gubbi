"""Compatibility shim -- cipher_guard moved to gubbi.crypto.require_cipher."""

from __future__ import annotations

import warnings
from typing import Any


def __getattr__(name: str) -> Any:
    if name == "require_cipher":
        warnings.warn(
            "gubbi.core.cipher_guard is deprecated; "
            "import from gubbi.crypto.require_cipher instead",
            DeprecationWarning,
            stacklevel=2,
        )
        from gubbi.crypto.guard import require_cipher  # noqa: PLC0415

        return require_cipher
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
