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

import pytest

_TENANT_TABLES = frozenset(
    {
        "topics",
        "entries",
        "conversations",
        "messages",
        "entry_embeddings",
        "users",  # added by migration 0019_rls_users
        "extraction_jobs",  # added by migration 0023_extraction_jobs_relocate; RLS policy extraction_jobs_user_isolation
    }
)

ADMIN_ONLY_ALLOWLIST = frozenset(
    {
        "audit_log",  # INSERT-only for journal_app (migration 0010); immutability trigger blocks UPDATE/DELETE; no per-user filtering needed
        "alembic_version",  # Alembic internal tracking table; never touched by app code at runtime
        # OAuth SQLite tables (oauth.db, not PostgreSQL) -- see gubbi/oauth/storage.py
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
        # Singular noun forms that appear in string literals (e.g. title="Update Entry")
        "entry",
        # Common English word that appears as false positive after FROM in prose
        "the",
        # Common English pronoun that appears after UPDATE in prose
        # (e.g. ``"... cross-user UPDATE that a user-scoped pool ..."``)
        "that",
        # PL/pgSQL variables captured by INTO (DO-block locals, not tables)
        "nxt",
        # Reserved words after UPDATE in row-level locking clauses
        # (``FOR UPDATE SKIP LOCKED`` -- ``skip`` is not a table)
        "skip",
    }
)

_SQL_KEYWORD_RE = re.compile(
    r"(?:FROM|INTO|UPDATE|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)",
    re.IGNORECASE | re.MULTILINE,
)


_CTE_ALIAS_RE = re.compile(
    r"(?:\bWITH(?:\s+RECURSIVE)?|,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s+AS\s*(?:NOT\s+)?(?:MATERIALIZED\s*)?\(",
    re.IGNORECASE,
)


def _cte_aliases(source: str) -> set[str]:
    """Return the lowercased CTE names a SQL string defines in its WITH clause."""
    if not re.search(r"\bWITH\b", source, re.IGNORECASE):
        return set()
    return {m.group(1).lower() for m in _CTE_ALIAS_RE.finditer(source)}


def _extract_table_names(source: str, statement: str | None = None) -> set[str]:
    """Parse a SQL string and return the set of table names referenced.

    CTE names defined in ``statement`` (default: ``source`` itself) are
    subtracted, so an alias defined in one statement never hides a real
    table of the same name in another.
    """
    aliases = _cte_aliases(source if statement is None else statement)
    tables: set[str] = set()
    for match in _SQL_KEYWORD_RE.finditer(source):
        name = match.group(1)
        if name.lower() in _SKIP_KEYWORDS or name.lower() in aliases:
            continue
        if name.startswith("pg_") or name == "information_schema":
            continue
        # SQLite system catalogue (analog of pg_* / information_schema):
        # used by gubbi/oauth/_rate_limit.py for schema-existence probes.
        if name == "sqlite_master":
            continue
        tables.add(name)
    return tables


def _enclosing_statement(node: ast.Constant, parent: ast.AST | None) -> str | None:
    """Return the literal text of the f-string ``node`` belongs to, if any.

    An f-string splits into one constant per literal segment; a CTE defined
    before an interpolation is referenced in a later segment.
    """
    if not isinstance(parent, ast.JoinedStr):
        return None
    return " ".join(
        part.value
        for part in parent.values
        if isinstance(part, ast.Constant) and isinstance(part.value, str)
    )


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
            parent = parent_map.get(id(node))
            if isinstance(parent, ast.Expr):
                continue

            val = node.value
            if not any(kw in val.upper() for kw in ("SELECT", "INSERT", "UPDATE", "DELETE")):
                continue
            if not any(kw in val.upper() for kw in ("FROM", "INTO")) and not re.search(
                r"UPDATE\s+\w+", val, re.IGNORECASE
            ):
                continue
            found |= _extract_table_names(val, _enclosing_statement(node, parent))
    return found


def test_all_app_sql_tables_are_accounted_for() -> None:
    """Assert all tables referenced in app SQL are in _TENANT_TABLES or ADMIN_ONLY_ALLOWLIST."""
    src_root = Path(__file__).resolve().parents[2] / "gubbi"
    found_tables = _collect_sql_tables(src_root)
    unknown = found_tables - _TENANT_TABLES - ADMIN_ONLY_ALLOWLIST
    assert not unknown, (
        "SQL references tables not in _TENANT_TABLES or ADMIN_ONLY_ALLOWLIST:\n"
        + "\n".join(
            f"  {t!r} -- either add RLS migration or add to ADMIN_ONLY_ALLOWLIST with justification"
            for t in sorted(unknown)
        )
    )


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        pytest.param(
            "WITH released AS (UPDATE extraction_jobs SET run_after = now() RETURNING status) "
            "SELECT status FROM released",
            {"extraction_jobs"},
            id="cte-alias-subtracted",
        ),
        pytest.param(
            "WITH counts AS (SELECT user_id FROM entries), _upd AS (UPDATE topics SET x = 1) "
            "SELECT * FROM counts JOIN _upd ON true",
            {"entries", "topics"},
            id="multiple-cte-aliases-subtracted",
        ),
        pytest.param(
            "WITH released AS (SELECT id FROM extraction_jobs) "
            "SELECT * FROM released JOIN unscoped_table ON true",
            {"extraction_jobs", "unscoped_table"},
            id="real-table-beside-cte-still-flagged",
        ),
    ],
)
def test_extract_table_names_subtracts_only_own_cte_aliases(sql: str, expected: set[str]) -> None:
    assert _extract_table_names(sql) == expected


def test_cte_alias_does_not_hide_same_named_table_in_another_statement(tmp_path: Path) -> None:
    (tmp_path / "queries.py").write_text(
        "A = 'WITH ledger AS (SELECT id FROM entries) SELECT id FROM ledger'\n"
        "B = 'SELECT id FROM ledger'\n",
        encoding="utf-8",
    )

    found = _collect_sql_tables(tmp_path)

    assert found == {"entries", "ledger"}


def test_cte_alias_spans_fstring_interpolation(tmp_path: Path) -> None:
    (tmp_path / "queries.py").write_text(
        "W = 'x'\nA = f'WITH ledger AS (SELECT id FROM entries WHERE {W}) SELECT id FROM ledger'\n",
        encoding="utf-8",
    )

    found = _collect_sql_tables(tmp_path)

    assert found == {"entries"}


def test_users_table_in_tenant_tables() -> None:
    """Regression guard: users table must be in _TENANT_TABLES."""
    assert "users" in _TENANT_TABLES, (
        "users table must be in _TENANT_TABLES -- migration 0019_rls_users"
    )
