"""Invariant: every read/list tool must call _assert_response_ok.

AST-walks each tool source file and asserts the function body contains
a call to ``_assert_response_ok``.  This catches the next read/list tool
added without the response-size guard at PR review time.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from gubbi.tools.registry import READ_TOOLS

pytestmark = pytest.mark.unit

# All read/list tool names that must include the response-size guard.
# This set mirrors the tools in ``gubbi.tools.registry.READ_TOOLS``.
READ_AND_LIST_TOOLS: frozenset[str] = READ_TOOLS

# Maps tool name to the source file it lives in.
# Most tools map to ``<stem>.py`` (e.g. journal_search -> search.py),
# but ``journal_briefing`` and ``journal_timeline`` are in ``context.py``.
_TOOL_SOURCE: dict[str, str] = {
    "journal_search": "search.py",
    "journal_list_topics": "topics.py",
    "journal_list_conversations": "conversations.py",
    "journal_read_topic": "entries.py",
    "journal_read_conversation": "conversations.py",
    "journal_briefing": "context.py",
    "journal_timeline": "context.py",
}


def _function_contains_call_assert_response_ok(tree: ast.AST, func_name: str) -> bool:
    """Return True if ``func_name`` (async def or def) contains a call to _assert_response_ok."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == func_name:
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    fn = child.func
                    # Direct call: _assert_response_ok(...)
                    if isinstance(fn, ast.Name) and fn.id == "_assert_response_ok":
                        return True
                    # Attribute call: mod._assert_response_ok(...)
                    if isinstance(fn, ast.Attribute) and fn.attr == "_assert_response_ok":
                        return True
            return False
    return False


class TestResponseSizeGuardInvariant:
    """Every read/list tool must call _assert_response_ok before returning."""

    tools_dir = Path(__file__).resolve().parents[2] / "gubbi" / "tools"

    def test_all_read_tools_have_guard(self) -> None:
        missing: list[str] = []
        for tool_name in sorted(READ_AND_LIST_TOOLS):
            filename = _TOOL_SOURCE.get(tool_name)
            assert filename is not None, f"No source mapping for tool {tool_name}"
            source_path = self.tools_dir / filename
            assert source_path.exists(), f"Source file not found: {source_path}"
            tree = ast.parse(source_path.read_text(encoding="utf-8"))
            if not _function_contains_call_assert_response_ok(tree, tool_name):
                missing.append(tool_name)
        assert (
            not missing
        ), f"The following tools are missing a call to _assert_response_ok: {missing}"
