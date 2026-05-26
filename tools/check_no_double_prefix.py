#!/usr/bin/env python3
"""Ban accidental ``JOURNAL_JOURNAL_`` doubled-prefix env-var names.

Catches the failure mode where a pydantic Settings field whose Python name
starts with ``journal_`` is declared under
``model_config = SettingsConfigDict(env_prefix="JOURNAL_", ...)`` and silently
produces an env var named ``JOURNAL_JOURNAL_<NAME>``.  The fix is to set
``validation_alias=`` on the field to declare its env name explicitly.

Allowlist: ``JOURNAL_JOURNAL_TARGET_URL`` is documented as a known exception
(rename deferred until it can ride a deploy).

Exit 0 on success, 1 on any non-allowlisted match.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

# Match a doubled JOURNAL_ prefix followed by an UPPER_SNAKE_CASE name.
PATTERN = re.compile(r"\bJOURNAL_JOURNAL_[A-Z][A-Z0-9_]*\b")

# Allowlisted names: documented exceptions whose rename is out of scope.
ALLOWLIST = frozenset({"JOURNAL_JOURNAL_TARGET_URL"})


def _out(msg: str, file: Any = sys.stderr) -> None:
    file.write(msg + "\n")


def find_violations(paths: list[Path]) -> list[tuple[Path, int, str]]:
    """Return (file, line, matched-name) for every non-allowlisted match."""
    violations: list[tuple[Path, int, str]] = []
    for path in paths:
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            # Binary file or unreadable -- regex would not match anyway.
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            for match in PATTERN.finditer(line):
                name = match.group(0)
                if name not in ALLOWLIST:
                    violations.append((path, line_no, name))
    return violations


def main(argv: list[str]) -> int:
    """CLI entry point: scan paths from argv for doubled-prefix violations."""
    paths = [Path(p) for p in argv[1:]]
    violations = find_violations(paths)
    if violations:
        for path, line, name in violations:
            _out(
                f"{path}:{line}: doubled JOURNAL_ prefix forbidden "
                f"({name}); use validation_alias to set explicit env name",
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
