"""Compatibility shim -- context moved to gubbi.app_context."""

from __future__ import annotations

import warnings
from typing import Any


def __getattr__(name: str) -> Any:
    if name == "AppContext":
        warnings.warn(
            "gubbi.core.context is deprecated; import from gubbi.app_context instead",
            DeprecationWarning,
            stacklevel=2,
        )
        from gubbi.app_context import AppContext  # noqa: PLC0415

        return AppContext
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
