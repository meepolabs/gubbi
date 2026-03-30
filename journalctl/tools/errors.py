"""Structured error helpers for MCP tool responses.

Tools return these dicts instead of raising so the LLM can read the
error_code and suggestions and self-correct without a round trip.
"""

from __future__ import annotations

import re
from typing import Any, NotRequired, TypedDict, cast


class ErrorResult(TypedDict):
    """Expected shape of every error dict returned by MCP tools."""

    error: str
    error_code: str
    suggestions: list[str]
    input: NotRequired[str]


def _topic_suggestions(raw: str) -> list[str]:
    """Generate a sanitized topic path candidate from an invalid input."""
    cleaned = re.sub(r"[^a-z0-9/]+", "-", raw.lower()).strip("-/")
    cleaned = re.sub(r"-{2,}", "-", cleaned)
    parts = [p.strip("-") for p in cleaned.split("/") if p.strip("-")][:2]
    if parts:
        return ["/".join(parts)]
    return []


def invalid_topic(raw: str, detail: str = "") -> dict[str, Any]:
    result: ErrorResult = {
        "error": (
            detail
            or f"Invalid topic path: '{raw}'. "
            "Use lowercase alphanumeric with hyphens, "
            "max 2 levels (e.g. 'health', 'projects/my-app')."
        ),
        "error_code": "INVALID_TOPIC",
        "input": raw,
        "suggestions": _topic_suggestions(raw),
    }
    return cast(dict[str, Any], result)


def invalid_date(raw: str) -> dict[str, Any]:
    result: ErrorResult = {
        "error": f"Invalid date: '{raw}'. Expected format: YYYY-MM-DD (e.g. 2026-03-29).",
        "error_code": "INVALID_DATE",
        "input": raw,
        "suggestions": [],
    }
    return cast(dict[str, Any], result)


def not_found(resource: str, identifier: str | int) -> dict[str, Any]:
    result: ErrorResult = {
        "error": f"{resource} not found: {identifier}",
        "error_code": "NOT_FOUND",
        "input": str(identifier),
        "suggestions": [],
    }
    return cast(dict[str, Any], result)


def already_exists(topic: str) -> dict[str, Any]:
    result: ErrorResult = {
        "error": f"Topic already exists: '{topic}'",
        "error_code": "ALREADY_EXISTS",
        "input": topic,
        "suggestions": [],
    }
    return cast(dict[str, Any], result)


def validation_error(detail: str) -> dict[str, Any]:
    result: ErrorResult = {
        "error": detail,
        "error_code": "VALIDATION_ERROR",
        "suggestions": [],
    }
    return cast(dict[str, Any], result)
