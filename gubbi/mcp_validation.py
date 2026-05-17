"""Map pre-body Pydantic ``ValidationError`` to the canonical envelope.

The MCP SDK's tool dispatch validates arguments against the tool's
Pydantic-derived schema BEFORE the tool body's try/except can intercept.
On a mismatch, ``Tool.run`` catches the ``ValidationError`` and re-raises
as ``ToolError`` (with the original ValidationError as ``__cause__``),
which the lowlevel server reports back to the client as ``isError=true``
with the raw pydantic message. That bypasses
``gubbi.tools.errors.validation_error`` -- the universal envelope used
by every other tool failure mode (NOT_FOUND, INVALID_TOPIC, INVALID_DATE,
ALREADY_EXISTS, VALIDATION_ERROR for in-body checks).

Surfaced 2026-05-15 by gubbi-testbench probing ``journal_update_entry``
with ``mode="prepend"`` (not in ``Literal["replace", "append"]``). The
error IS surfaced to clients (no silent acceptance of bad enums) but it
is not in the canonical shape, so client code that branches on
``error_code`` to drive auto-correction sees a different shape for
schema-validation failures than for in-body failures.

The fix: subclass ``FastMCP`` and override ``call_tool``. When a
``ToolError`` arrives whose ``__cause__`` is a ``ValidationError``,
return a canonical envelope dict instead of re-raising. The strict
``Literal[...]`` typing on tool signatures is preserved -- the schema
clients see (and use for argument completion) still reflects the
canonical type set; only the failure path changes.
"""

from __future__ import annotations

import traceback
from collections.abc import Sequence
from typing import Any, cast

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import ContentBlock
from pydantic import ValidationError

from gubbi.tools.errors import validation_error

__all__ = ["JournalFastMCP", "format_validation_error"]


def _is_argument_validation_error(exc: ValidationError) -> bool:
    """Return whether the ValidationError came from argument validation."""
    return any(
        frame.name == "call_fn_with_arg_validation"
        for frame in traceback.extract_tb(exc.__traceback__)
    )


def format_validation_error(exc: ValidationError) -> str:
    """Render a pydantic ``ValidationError`` as a single readable string.

    Multiple field-level errors are joined by ``"; "``; each entry shows
    the dotted path of the bad argument, the message, and the offending
    value when it is a non-collection scalar (collection inputs are
    omitted from the string to avoid leaking large payloads into the
    error envelope).
    """
    parts: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err.get("loc", ())) or "(input)"
        msg = err.get("msg", "invalid value")
        bad = err.get("input")
        if bad is not None and not isinstance(bad, dict | list):
            parts.append(f"{loc}: {msg} (got {bad!r})")
        else:
            parts.append(f"{loc}: {msg}")
    return "; ".join(parts) or "argument validation failed"


class JournalFastMCP(FastMCP):
    """FastMCP that returns the canonical envelope on argument failure.

    Pre-body pydantic ``ValidationError`` is mapped to
    ``gubbi.tools.errors.validation_error`` so clients see a normal tool
    result with ``success=False, error_code=VALIDATION_ERROR`` instead of
    ``isError=true`` with a raw pydantic message. Other tool errors
    propagate unchanged so real bugs surface loudly.
    """

    async def call_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> Sequence[ContentBlock] | dict[str, Any]:
        """Dispatch a tool, mapping ValidationError to canonical envelope."""
        try:
            result = await super().call_tool(name, arguments)
            return cast(Sequence[ContentBlock] | dict[str, Any], result)
        except ToolError as exc:
            cause = exc.__cause__
            if isinstance(cause, ValidationError) and _is_argument_validation_error(cause):
                return validation_error(format_validation_error(cause))
            raise
