"""Compatibility shim -- validation moved to gubbi.validation."""

from __future__ import annotations

import importlib
import warnings
from typing import Any

_MAP = {
    "sanitize_label": ("gubbi.validation", "sanitize_label"),
    "sanitize_freetext": ("gubbi.validation", "sanitize_freetext"),
    "reject_tool_call_syntax": ("gubbi.validation", "reject_tool_call_syntax"),
    "validate_topic": ("gubbi.validation", "validate_topic"),
    "harden_llm_topic_path": ("gubbi.validation", "harden_llm_topic_path"),
    "validate_title": ("gubbi.validation", "validate_title"),
    "validate_date": ("gubbi.validation", "validate_date"),
    "local_today": ("gubbi.validation", "local_today"),
    "is_future_date": ("gubbi.validation", "is_future_date"),
    "slugify": ("gubbi.validation", "slugify"),
}


def __getattr__(name: str) -> Any:
    target = _MAP.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_path, attr = target
    warnings.warn(
        f"gubbi.core.validation.{name} is deprecated; import from {module_path}.{attr} instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return getattr(importlib.import_module(module_path), attr)
