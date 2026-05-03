"""Static verification that migrations contain no user-row mutations.

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

_MIGRATION_0021_FILE = (
    pathlib.Path(__file__).resolve().parents[2]
    / "journalctl"
    / "alembic"
    / "versions"
    / "20260503_0021_perf_audit_log_actor_idx.py"
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


# ---- Migration 0021: composite index on idx_audit_log_actor_id -----------


def test_migration_0021_upgrade_drops_old_bare_index() -> None:
    """Upgrade must drop the old bare idx_audit_log_actor_id index."""
    src = _strip_comments(_MIGRATION_0021_FILE.read_text(encoding="utf-8"))
    assert re.search(
        r"DROP\s+INDEX\s+(IF\s+EXISTS\s+)?idx_audit_log_actor_id\b",
        src,
        re.IGNORECASE,
    ), "upgrade() must drop idx_audit_log_actor_id"


def test_migration_0021_upgrade_creates_composite_index() -> None:
    """Upgrade must create the composite (actor_id, occurred_at DESC) index."""
    src = _strip_comments(_MIGRATION_0021_FILE.read_text(encoding="utf-8"))
    assert re.search(
        r"CREATE\s+INDEX.*?idx_audit_log_actor_id\b",
        src,
        re.IGNORECASE,
    ), "upgrade() must create idx_audit_log_actor_id index"
    assert re.search(
        r"audit_log\s*\(\s*actor_id\s*,\s*occurred_at\s+DESC\s*\)",
        src,
        re.IGNORECASE,
    ), "index columns must be (actor_id, occurred_at DESC)"


def test_migration_0021_upgrade_contains_both_drop_and_create() -> None:
    """Upgrade body (between def upgrade and def downgrade) has both ops."""
    raw = _MIGRATION_0021_FILE.read_text(encoding="utf-8")
    up_start = raw.find("def upgrade()")
    down_start = raw.find("def downgrade()")
    assert up_start >= 0, "upgrade() function not found in migration"
    assert down_start > up_start, "downgrade() must follow upgrade()"
    body = raw[up_start:down_start]
    has_drop = re.search(
        r"DROP\s+INDEX.*?idx_audit_log_actor_id\b",
        body,
        re.IGNORECASE | re.DOTALL,
    )
    has_create = re.search(
        r"CREATE\s+INDEX.*?ON\s+audit_log\s*\(.*?actor_id.*?occurred_at\s+DESC",
        body,
        re.IGNORECASE | re.DOTALL,
    )
    assert has_drop, "upgrade() must contain DROP for idx_audit_log_actor_id"
    assert has_create, "upgrade() must create composite (actor_id, occurred_at DESC) index"
