"""Compatibility shim -- scope moved to gubbi.auth.scope."""

from __future__ import annotations

import importlib
import warnings
from typing import Any

_MAP = {
    "check_scope": ("gubbi.auth.scope", "check_scope"),
    "require_scope": ("gubbi.auth.scope", "require_scope"),
    "insufficient_scope_response": ("gubbi.auth.scope", "insufficient_scope_response"),
    "SCOPE_DESCRIPTIONS": ("gubbi.auth.scope", "SCOPE_DESCRIPTIONS"),
    "SCOPE_GRANTS": ("gubbi.auth.scope", "SCOPE_GRANTS"),
}


def __getattr__(name: str) -> Any:
    target = _MAP.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_path, attr = target
    warnings.warn(
        f"gubbi.core.scope.{name} is deprecated; import from {module_path}.{attr} instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return getattr(importlib.import_module(module_path), attr)
