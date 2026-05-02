"""AST scan: every ``Action.NAME`` referenced in journalctl source must exist
in the pinned ``gubbi_common.audit.actions.Action``.

This is the consumer-side drift guard for the Action contract. It catches
the failure mode where journalctl HEAD references an Action constant that
isn't in the gubbi-common version pinned in pyproject.toml -- a clean
install at HEAD would raise AttributeError on first use.

If this test fails, either:
  * Bump the gubbi-common pin to a release that includes the missing
    constant, or
  * Add the constant to gubbi-common and cut a new release.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from gubbi_common.audit.actions import Action

_SOURCE_ROOT = Path(__file__).resolve().parent.parent.parent / "journalctl"


def _action_names_referenced() -> dict[str, list[tuple[Path, int]]]:
    """Walk every .py under journalctl/ and collect ``Action.X`` attribute references."""
    refs: dict[str, list[tuple[Path, int]]] = {}
    for py in _SOURCE_ROOT.rglob("*.py"):
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "Action"
            ):
                refs.setdefault(node.attr, []).append((py, node.lineno))
    return refs


@pytest.mark.unit
def test_every_action_reference_resolves() -> None:
    refs = _action_names_referenced()
    valid_names = {name for name in dir(Action) if not name.startswith("_")}

    missing: list[str] = []
    for name, locations in sorted(refs.items()):
        if name not in valid_names:
            for path, lineno in locations:
                rel = path.relative_to(_SOURCE_ROOT.parent)
                missing.append(f"  {rel}:{lineno} -- Action.{name}")

    assert not missing, (
        "journalctl references Action constants missing from the pinned "
        "gubbi-common.audit.actions.Action.\n"
        "Either bump the gubbi-common pin in pyproject.toml or add the "
        "constant to gubbi-common and release.\n\n"
        "Missing references:\n" + "\n".join(missing)
    )
