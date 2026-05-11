"""Drift guard between conftest's TRUNCATE list and ``_TENANT_TABLES``.

``tests/conftest.py`` truncates a hand-maintained list of tenant tables
in the ``clean_rls_db`` fixture (and a smaller subset in ``clean_pool``).
The canonical list of tenant tables lives in
``tests/unit/test_tenant_table_coverage.py`` as ``_TENANT_TABLES``.

If a new tenant table is added (migration + RLS policy) without also
extending the conftest TRUNCATE statements, the next integration test
that relies on isolation will leak rows across cases.  This guard fails
loudly on that drift instead of letting tests pass-then-flap.

The TRUNCATE list is parsed from the conftest source rather than
introspected, because the SQL is currently a literal string and there is
no shared constant to import.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.unit.test_tenant_table_coverage import _TENANT_TABLES

# Tables that are allowed to appear in conftest's TRUNCATE list without
# being in ``_TENANT_TABLES`` (audit-only writes, no RLS, but still need
# clearing between integration tests).  Keep this set tiny and documented.
_TRUNCATE_EXTRAS = frozenset({"audit_log"})

_CONFTEST = Path(__file__).resolve().parents[1] / "conftest.py"

# ``TRUNCATE a, b, c RESTART IDENTITY CASCADE`` (single-line or wrapped).
# We capture each TRUNCATE statement's table list and union them.
_TRUNCATE_RE = re.compile(
    r"TRUNCATE\s+([A-Za-z0-9_,\s]+?)\s+RESTART\s+IDENTITY",
    re.IGNORECASE,
)


def _parse_truncate_tables() -> frozenset[str]:
    """Return the union of tables truncated across all TRUNCATE statements.

    The conftest's TRUNCATE SQL is sometimes split across two adjacent
    Python string literals (implicit concatenation across a newline).
    Normalise quote+whitespace+quote to a single space before scanning so
    the regex can span those splits.
    """
    raw_source = _CONFTEST.read_text()
    # Collapse adjacent string-literal concatenation: `..."<newline>spaces"...`
    normalised = re.sub(r'"\s*\n\s*"', " ", raw_source)
    tables: set[str] = set()
    for match in _TRUNCATE_RE.finditer(normalised):
        for raw in match.group(1).split(","):
            name = raw.strip()
            if name:
                tables.add(name)
    return frozenset(tables)


@pytest.mark.integration
def test_conftest_truncate_covers_tenant_tables() -> None:
    """Conftest's TRUNCATE list must include every tenant table."""
    truncated = _parse_truncate_tables()
    assert truncated, "Failed to parse any TRUNCATE statement from tests/conftest.py"

    missing = _TENANT_TABLES - truncated
    assert not missing, (
        "tests/conftest.py TRUNCATE list is missing tenant tables -- integration "
        "test isolation will leak rows across cases. Add to the TRUNCATE "
        f"statements in clean_pool/clean_rls_db: {sorted(missing)}"
    )


@pytest.mark.integration
def test_conftest_truncate_has_no_unknown_tables() -> None:
    """Conftest must not truncate tables outside _TENANT_TABLES + allowlist."""
    truncated = _parse_truncate_tables()
    unknown = truncated - _TENANT_TABLES - _TRUNCATE_EXTRAS
    assert not unknown, (
        "tests/conftest.py truncates tables that are neither in _TENANT_TABLES "
        "nor in the audit-only allowlist. Either add them to _TENANT_TABLES "
        "(plus a migration + RLS policy) or to _TRUNCATE_EXTRAS in this test "
        f"with a one-line justification: {sorted(unknown)}"
    )
