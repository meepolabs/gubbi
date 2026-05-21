"""Compatibility shim -- logger moved to gubbi.telemetry.logger."""

from __future__ import annotations

import warnings
from typing import Any


def __getattr__(name: str) -> Any:
    if name == "initialize_logger":
        warnings.warn(
            "gubbi.core.logger is deprecated; " "import from gubbi.telemetry.logger instead",
            DeprecationWarning,
            stacklevel=2,
        )
        from gubbi.telemetry.logger import initialize_logger

        return initialize_logger
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
