"""Tests for the FastMCP subclass that maps pre-body pydantic
``ValidationError`` to the canonical ``VALIDATION_ERROR`` envelope.

Covers three contracts:

1. A bad ``Literal[...]`` argument flows back as the canonical
   ``{success: False, error_code: "VALIDATION_ERROR", ...}`` dict --
   not as ``isError=true`` with a raw pydantic message. Mirrors the
   exact failure mode the gubbi-testbench probe surfaced
   (``journal_update_entry`` with ``mode="prepend"``).
2. A correct argument still routes through to the tool body. The
   wrapper only intercepts the failure path.
3. A tool body raising a non-validation exception still surfaces as
   ``ToolError``. We do not silently swallow real bugs by widening the
   except clause.
"""

from __future__ import annotations

from typing import Any, Literal, cast

import pytest
from mcp.server.fastmcp.exceptions import ToolError
from pydantic import BaseModel, ValidationError

from gubbi.mcp_validation import JournalFastMCP, format_validation_error

pytestmark = pytest.mark.unit


def _build_server() -> JournalFastMCP:
    """Build a JournalFastMCP with a single tool that has a Literal arg."""
    mcp = JournalFastMCP("test-journal-fastmcp")

    @mcp.tool()
    async def tool_with_literal(mode: Literal["replace", "append"]) -> dict[str, Any]:
        return {"success": True, "echoed_mode": mode}

    return mcp


async def test_invalid_literal_arg_returns_canonical_envelope() -> None:
    """A bad Literal value must flow back as the canonical envelope dict.

    Reproduces the gubbi-testbench finding: probing
    ``journal_update_entry`` with ``mode="prepend"`` previously surfaced
    as ``isError=true`` with a raw pydantic message, instead of
    ``{success: False, error_code: "VALIDATION_ERROR"}`` like every
    other failure mode in gubbi.
    """
    mcp = _build_server()

    result = await mcp.call_tool("tool_with_literal", {"mode": "prepend"})

    assert isinstance(result, dict), f"expected dict, got {type(result).__name__}"
    assert result.get("success") is False
    assert result.get("error_code") == "VALIDATION_ERROR"
    assert "suggestions" in result, "canonical envelope must include suggestions"
    # The error string should mention the field and the bad value so the
    # caller can self-correct without a round trip.
    err = result.get("error", "")
    assert "mode" in err, f"expected 'mode' in error message; got {err!r}"
    assert "prepend" in err, f"expected bad value 'prepend' in error message; got {err!r}"


async def test_valid_literal_arg_routes_through_to_tool_body() -> None:
    """A correct Literal value still hits the tool body untouched."""
    mcp = _build_server()

    result = await mcp.call_tool("tool_with_literal", {"mode": "replace"})

    # On the success path the lowlevel server's ``convert_result`` step
    # may wrap the dict into ContentBlocks. Either shape is acceptable;
    # we just assert the tool body ran and produced the expected mode.
    if isinstance(result, dict):
        assert result.get("success") is True
        assert result.get("echoed_mode") == "replace"
    else:
        # ContentBlock sequence -- rare in tests but defensible against
        # an SDK upgrade that changes the convert_result default.
        assert result, "expected non-empty result on success path"


async def test_tool_body_exceptions_still_propagate_as_tool_error() -> None:
    """A non-validation exception inside the tool body must still raise.

    Guards against a future widening of the except clause that would
    silently swallow real bugs.
    """
    mcp = JournalFastMCP("test-tool-body-error")

    @mcp.tool()
    async def failing_tool() -> dict[str, Any]:
        raise RuntimeError("intentional test failure")

    with pytest.raises(ToolError):
        await mcp.call_tool("failing_tool", {})


async def test_output_validation_errors_still_propagate_as_tool_error() -> None:
    """Structured-output validation failures must not look like bad args."""
    mcp = JournalFastMCP("test-output-validation-error")

    @mcp.tool()
    async def bad_output_tool() -> Literal["replace", "append"]:
        return cast(Literal["replace", "append"], "prepend")

    with pytest.raises(ToolError):
        await mcp.call_tool("bad_output_tool", {})


def test_format_validation_error_combines_multiple_field_errors() -> None:
    """Multiple field errors combine into one ``"; "``-joined string."""

    class MultiFieldModel(BaseModel):
        mode: Literal["replace", "append"]
        date: str

    with pytest.raises(ValidationError) as ctx:
        MultiFieldModel.model_validate({"mode": "prepend"})

    formatted = format_validation_error(ctx.value)

    assert "mode" in formatted
    assert "prepend" in formatted
    assert "date" in formatted
    assert "; " in formatted, "multiple errors should be joined with '; '"


def test_format_validation_error_handles_collection_input() -> None:
    """A collection-typed bad input is described without leaking the payload.

    The error string must not embed the raw list/dict (which could be
    arbitrarily large and pull payload data into operator log lines).
    """

    class CollectionModel(BaseModel):
        choice: Literal["a", "b"]

    with pytest.raises(ValidationError) as ctx:
        CollectionModel.model_validate({"choice": ["nested", "list"]})

    formatted = format_validation_error(ctx.value)

    assert "choice" in formatted
    # The raw list payload must not be embedded in the formatted string.
    assert "nested" not in formatted
    assert "list" not in formatted or "nested" not in formatted
