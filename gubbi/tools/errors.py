"""Structured error helpers for MCP tool responses.

Tools return these dicts instead of raising so the LLM can read the
error_code and suggestions and self-correct without a round trip.

Type notes:
    The public helper functions return ``dict[str, Any]`` (not ``ErrorResult``)
    because the registered MCP tool callbacks declare ``-> dict[str, Any]``
    and ``mcp.server.fastmcp`` introspects that annotation to decide on
    structured vs unstructured output.  Widening the registered callback
    return type to a union (``ErrorResult | dict[str, Any]``) would cascade
    through 40+ callsites AND change what FastMCP reports as the structured
    schema.

    For callers that genuinely need to narrow on ``Literal[False]`` (e.g.
    a future code path that does ``if not result["success"]: return result``
    and wants type-checker support), use the ``ToolResult`` alias below and
    cast the helper output at the boundary.  Until such a caller appears,
    the cast inside each helper is the correct + minimal trade-off and
    matches the rest of the codebase's narrowing patterns.
"""

from __future__ import annotations

import re
from typing import Any, Literal, NotRequired, TypedDict, cast


class ErrorResult(TypedDict):
    """Expected shape of every error dict returned by MCP tools."""

    error: str
    error_code: str
    success: Literal[False]
    suggestions: list[str]
    input: NotRequired[str]


# Union alias for any code path that wants narrowing semantics.  Not used
# as the helper return type for the reason explained in the module
# docstring above; provided so callers can opt in:
#
#     from gubbi.tools.errors import ErrorResult, ToolResult, validation_error
#     result: ToolResult = cast(ErrorResult, validation_error("..."))
#     if not result["success"]:        # narrowed to ErrorResult
#         return result
#
# Adding it here keeps the type vocabulary discoverable without forcing
# every caller to relearn the envelope shape.
ToolResult = ErrorResult | dict[str, Any]


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
        "success": False,
        "input": raw,
        "suggestions": _topic_suggestions(raw),
    }
    return cast("dict[str, Any]", result)


def invalid_date(raw: str) -> dict[str, Any]:
    result: ErrorResult = {
        "error": f"Invalid date: '{raw}'. Expected format: YYYY-MM-DD (e.g. 2026-03-29).",
        "error_code": "INVALID_DATE",
        "success": False,
        "input": raw,
        "suggestions": [],
    }
    return cast("dict[str, Any]", result)


def not_found(resource: str, identifier: str | int) -> dict[str, Any]:
    result: ErrorResult = {
        "error": f"{resource} not found: {identifier}",
        "error_code": "NOT_FOUND",
        "success": False,
        "input": str(identifier),
        "suggestions": [],
    }
    return cast("dict[str, Any]", result)


def already_exists(topic: str) -> dict[str, Any]:
    result: ErrorResult = {
        "error": f"Topic already exists: '{topic}'",
        "error_code": "ALREADY_EXISTS",
        "success": False,
        "input": topic,
        "suggestions": [],
    }
    return cast("dict[str, Any]", result)


def validation_error(detail: str) -> dict[str, Any]:
    result: ErrorResult = {
        "error": detail,
        "error_code": "VALIDATION_ERROR",
        "success": False,
        "suggestions": [],
    }
    return cast("dict[str, Any]", result)
