"""Static verification that migration 0004 contains no user-row mutations.

This test never requires a DB -- it reads the migration source file directly.
Run on every ``pytest tests/unit`` invocation to catch regressions early.
"""

from __future__ import annotations

import pathlib
import re

_MIGRATION_FILE = (
    pathlib.Path(__file__).resolve().parents[2]
    / "journalctl"
    / "alembic"
    / "versions"
    / "20260419_0004_add_user_id_to_tenants.py"
)


def _strip_comments(src: str) -> str:
    """Remove triple-quoted docstrings and single-line comments from source."""
    # Remove triple-quoted strings (module docstrings, inline docs)
    src = re.sub(r'"""[\s\S]*?"""', "", src)
    src = re.sub(r"'''[\s\S]*?'''", "", src)
    # Remove single-line comments (# to end of line), but not inside f-strings
    lines: list[str] = []
    for line in src.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue  # full-line comment
        hash_pos = _find_outside_string(line, "#")
        if hash_pos is not None:
            line = line[:hash_pos].rstrip()
        lines.append(line)
    return "\n".join(lines)


def _find_outside_string(s: str, char: str) -> int | None:
    """Find first occurrence of *char* that is not inside a string literal."""
    in_single = False
    in_double = False
    for i, c in enumerate(s):
        if c == "'" and not in_double:
            in_single = not in_single
        elif c == '"' and not in_single:
            in_double = not in_double
        elif c == char and not in_single and not in_double:
            return i
    return None


def _check_no_pattern(pattern: str, src: str) -> None:
    """Assert that *src* contains no match for *pattern*.

    *src* should already be comment-stripped via _strip_comments().
    """
    match = re.search(pattern, src, re.IGNORECASE)
    assert match is None, (
        f"Source must not contain '{match.group()}' -- "
        "user-row mutations belong in journalctl/users/bootstrap.py, not migrations."
    )


def test_migration_0004_has_no_insert_into_users() -> None:
    """Migration file must not contain INSERT INTO users."""
    src = _strip_comments(_MIGRATION_FILE.read_text(encoding="utf-8"))
    _check_no_pattern(r"INSERT\s+INTO\s+users\b", src)


def test_migration_0004_has_no_update_users() -> None:
    """Migration file must not contain UPDATE users SET."""
    src = _strip_comments(_MIGRATION_FILE.read_text(encoding="utf-8"))
    _check_no_pattern(r"UPDATE\s+users\s+SET\b", src)


def test_migration_0004_has_no_delete_from_users() -> None:
    """Migration file must not contain DELETE FROM users."""
    src = _strip_comments(_MIGRATION_FILE.read_text(encoding="utf-8"))
    _check_no_pattern(r"DELETE\s+FROM\s+users\b", src)
