"""Regression guard: every table referenced by app SQL must be accounted for.

Either the table is in _TENANT_TABLES (gets RLS) or it is in ADMIN_ONLY_ALLOWLIST
(explicitly confirmed to bypass RLS with justification).

If this test fails:
  1. Identify the table name printed in the assertion message.
  2. If the table SHOULD have RLS: add it to _TENANT_TABLES via a new migration
     and a matching RLS policy.
  3. If the table is admin-only (no row-level filtering needed): add it to
     ADMIN_ONLY_ALLOWLIST below with a one-line justification comment.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

_TENANT_TABLES = frozenset(
    {
        "topics",
        "entries",
        "conversations",
        "messages",
        "entry_embeddings",
        "users",  # added by migration 0019_rls_users (m234 C-7)
    }
)

ADMIN_ONLY_ALLOWLIST = frozenset(
    {
        "audit_log",  # INSERT-only for journal_app (migration 0010); immutability trigger blocks UPDATE/DELETE; no per-user filtering needed
        "alembic_version",  # Alembic internal tracking table; never touched by app code at runtime
        # OAuth SQLite tables (oauth.db, not PostgreSQL) -- see journalctl/oauth/storage.py
        "access_tokens",
        "auth_codes",
        "clients",
        "refresh_tokens",
        "token_pairs",
        "rate_limit_events",
        "journal_app",  # PostgreSQL role used in GRANT/REVOKE statements; not a table
    }
)

_SKIP_KEYWORDS = frozenset(
    {
        "public",
        "true",
        "false",
        "null",
        "and",
        "or",
        "not",
        "where",
        "set",
        "on",
        "as",
        "is",
        "by",
        "in",
        "at",
        "to",
        "of",
        # Additional SQL DDL/DML keywords
        "before",
        "cascade",
        "exclusive",
        "no",
        "json",
        "sql",
        # CTE aliases used in app SQL (not table names)
        "deleted",
        "new_entry",
        "updated",
        # Singular noun forms that appear in string literals (e.g. title="Update Entry")
        "entry",
        # Common English word that appears as false positive after FROM in prose
        "the",
        # PL/pgSQL variables captured by INTO (DO-block locals, not tables)
        "nxt",
    }
)

_SQL_KEYWORD_RE = re.compile(
    r"(?:FROM|INTO|UPDATE|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)",
    re.IGNORECASE | re.MULTILINE,
)


def _extract_table_names(source: str) -> set[str]:
    """Parse a SQL string and return the set of table names referenced."""
    tables: set[str] = set()
    for match in _SQL_KEYWORD_RE.finditer(source):
        name = match.group(1)
        if name.lower() in _SKIP_KEYWORDS:
            continue
        if name.startswith("pg_") or name == "information_schema":
            continue
        tables.add(name)
    return tables


def _collect_sql_tables(root: Path) -> set[str]:
    """Walk ``root`` for .py files, parse AST, and collect SQL table references.

    Skips docstrings (standalone expression statements) to avoid matching
    English words that co-incidentally follow SQL keywords in prose.
    """
    found: set[str] = set()
    for py_file in sorted(root.rglob("*.py")):
        try:
            source_text = py_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        try:
            tree = ast.parse(source_text, filename=str(py_file))
        except SyntaxError:
            continue

        # Build parent map to identify docstrings (Expr -> Constant)
        parent_map: dict[int, ast.AST] = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parent_map[id(child)] = node

        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            # Skip docstrings: string constants whose immediate parent is an
            # Expr node are standalone expression statements (docstrings).
            parent = parent_map.get(id(node), None)
            if isinstance(parent, ast.Expr):
                continue

            val = node.value
            if not any(kw in val.upper() for kw in ("SELECT", "INSERT", "UPDATE", "DELETE")):
                continue
            if not any(kw in val.upper() for kw in ("FROM", "INTO")) and not re.search(
                r"UPDATE\s+\w+", val, re.IGNORECASE
            ):
                continue
            found |= _extract_table_names(val)
    return found


def test_all_app_sql_tables_are_accounted_for() -> None:
    """Assert all tables referenced in app SQL are in _TENANT_TABLES or ADMIN_ONLY_ALLOWLIST."""
    src_root = Path(__file__).resolve().parents[2] / "journalctl"
    found_tables = _collect_sql_tables(src_root)
    unknown = found_tables - _TENANT_TABLES - ADMIN_ONLY_ALLOWLIST
    assert not unknown, (
        "SQL references tables not in _TENANT_TABLES or ADMIN_ONLY_ALLOWLIST:\n"
        + "\n".join(
            f"  {t!r} -- either add RLS migration or add to ADMIN_ONLY_ALLOWLIST with justification"
            for t in sorted(unknown)
        )
    )


def test_users_table_in_tenant_tables() -> None:
    """Regression guard: users table must be in _TENANT_TABLES (m234 C-7)."""
    assert (
        "users" in _TENANT_TABLES
    ), "users table must be in _TENANT_TABLES -- migration 0019_rls_users (m234 C-7)"
