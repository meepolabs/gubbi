"""Guard test: MCP tool handlers must NOT call ``pool.acquire()`` directly.

Every tenant-scoped tool has to go through ``user_scoped_connection`` so that
``app.current_user_id`` is set before RLS policies evaluate. A stray
``pool.acquire()`` inside a tool handler would silently return zero rows
under RLS (default-deny) — the kind of regression that passes code review
by accident because the failure mode looks like "no results" rather than
"crash".

This test greps the tool source files for the forbidden pattern. Admin
module (``admin.py``) is exempt — it is no longer an MCP tool; its
``pool.acquire()`` calls are library internals for the future admin API.
"""

from __future__ import annotations

import re
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[2] / "gubbi" / "tools"
ADMIN_ALLOWLIST = {"admin.py"}
PATTERN = re.compile(r"\bpool\.acquire\s*\(")


def test_tool_files_do_not_call_pool_acquire_directly() -> None:
    """Any occurrence of ``pool.acquire(`` outside admin.py is a regression."""
    offenders: list[str] = []
    for py_file in sorted(TOOLS_DIR.glob("*.py")):
        if py_file.name in ADMIN_ALLOWLIST or py_file.name == "__init__.py":
            continue
        text = py_file.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if PATTERN.search(line):
                offenders.append(
                    f"{py_file.relative_to(TOOLS_DIR.parent.parent)}:{lineno}: {line.strip()}"
                )
    assert not offenders, (
        "Tool handlers must use user_scoped_connection, not pool.acquire() directly.\n"
        "Offending lines:\n  " + "\n  ".join(offenders)
    )
