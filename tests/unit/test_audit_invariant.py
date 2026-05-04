"""Invariant: every write tool decorated with @require_scope("journal:write")
must also be decorated with @audited(...).

AST-walks each tool source file and asserts that every function decorated
with ``@require_scope("journal:write")`` also carries ``@audited(...)``.
This catches future write tools added without an audit trail at PR review
time.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from gubbi.tools.registry import WRITE_TOOLS

pytestmark = pytest.mark.unit

# Maps tool name to the source file it lives in.
# Every write tool is in one of three files.
_TOOL_SOURCE: dict[str, str] = {
    "journal_append_entry": "entries.py",
    "journal_update_entry": "entries.py",
    "journal_delete_entry": "entries.py",
    "journal_create_topic": "topics.py",
    "journal_save_conversation": "conversations.py",
}


def _function_has_require_scope_write(decorators: list[ast.expr]) -> bool:
    """Return True if the decorator list includes @require_scope("journal:write")."""
    for d in decorators:
        if (
            isinstance(d, ast.Call)
            and isinstance(d.func, ast.Name)
            and d.func.id == "require_scope"
            and d.args
            and isinstance(d.args[0], ast.Constant)
            and d.args[0].value == "journal:write"
        ):
            return True
    return False


def _function_has_audited(decorators: list[ast.expr]) -> bool:
    """Return True if the decorator list includes @audited(...)."""
    for d in decorators:
        if isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id == "audited":
            return True
    return False


class TestAuditInvariant:
    """Every write tool must have @audited(...) alongside @require_scope("journal:write")."""

    tools_dir = Path(__file__).resolve().parents[2] / "gubbi" / "tools"

    def test_all_write_tools_have_audited(self) -> None:
        missing: list[str] = []
        for tool_name in sorted(WRITE_TOOLS):
            filename = _TOOL_SOURCE.get(tool_name)
            assert filename is not None, f"No source mapping for tool {tool_name}"
            source_path = self.tools_dir / filename
            assert source_path.exists(), f"Source file not found: {source_path}"
            tree = ast.parse(source_path.read_text(encoding="utf-8"))

            found_func = False
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
                    and node.name == tool_name
                ):
                    found_func = True
                    has_scope = _function_has_require_scope_write(node.decorator_list)
                    has_audit = _function_has_audited(node.decorator_list)
                    if has_scope and not has_audit:
                        missing.append(tool_name)
                    break

            assert found_func, f"Function {tool_name} not found in {filename}"

        assert not missing, f'The following write tools have @require_scope("journal:write") but are missing @audited: {missing}'
