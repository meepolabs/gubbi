"""Shared query-building helpers for repository modules."""

from __future__ import annotations

from typing import Any


def _escape_like(s: str) -> str:
    """Escape SQL LIKE metacharacters using ! as the escape character."""
    return s.replace("!", "!!").replace("%", "!%").replace("_", "!_")


def _add_param(params: list[Any], value: Any) -> str:
    """Append value to params and return the next $N placeholder."""
    params.append(value)
    return f"${len(params)}"
