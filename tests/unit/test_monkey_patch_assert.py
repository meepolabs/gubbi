"""Test monkey-patch assert at startup (M-9.10)."""

from __future__ import annotations

import inspect

from mcp.server.fastmcp.tools import ToolManager


def test_monkey_patch_assert_at_startup() -> None:
    """Patching ToolManager.call_tool sets __wrapped__ and passes check."""
    from journalctl.tools.registry import _patch_tool_manager

    tm = ToolManager()
    # Patch should install without side-effects.
    _patch_tool_manager(tm)

    # Check the wrapper chain: __wrapped__ must exist on the bound method.
    wrapped = getattr(tm.call_tool, "__wrapped__", None)
    assert wrapped is not None  # noqa: PT018
    assert hasattr(wrapped, "__func__")  # noqa: PT018

    # The patched call must still be callable.
    assert callable(tm.call_tool)


def test_patch_removed_wasted_attrs() -> None:
    """The patched call_tool no longer emits empty user_id / scope_required."""
    from journalctl.tools.registry import _patch_tool_manager

    tm = ToolManager()
    _patch_tool_manager(tm)

    # Inspect the source of the wrapper function to confirm fields were
    # removed rather than left as stale placeholders.
    # tm.call_tool is a BoundMethod; .func gives us the underlying unbound
    # function that functools.wraps decorated.
    wrapped = getattr(tm.call_tool, "__wrapped__", None)
    func = wrapped.__func__  # type: ignore[union-attr]
    src = inspect.getsource(func)
    assert '"user_id"' not in src, "wasted user_id field should be removed"
    assert '"tool.scope_required"' not in src, "wasted tool.scope_required field should be removed"
