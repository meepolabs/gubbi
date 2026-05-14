"""Shared helpers for alembic migrations.

This module centralises low-level dance steps that several migrations need
but that are too small to live in a third-party package (gubbi-common would
introduce cross-repo coupling for migrations).

The current API surface is one context manager:

    autocommit_block(conn) -> Iterator[psycopg.Connection]

which is used by ``CREATE INDEX CONCURRENTLY`` migrations to step out of
Alembic's wrapping transaction safely.

Pinned to psycopg 3.x semantics: ``conn.connection.autocommit = True/False``.
If the pyproject psycopg pin moves to 4.x, audit this contract -- 4.x may
switch to ``set_autocommit()`` method form or context-manager-only.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


@contextmanager
def autocommit_block(conn: Any) -> Iterator[Any]:
    """Run a CREATE INDEX CONCURRENTLY (or similar) outside the Alembic txn.

    ``CREATE INDEX CONCURRENTLY`` cannot run inside a transaction block.
    Alembic wraps every ``upgrade()`` / ``downgrade()`` in one, so the
    pattern in concurrency-using migrations is:

        op.execute("COMMIT")            # exit the Alembic transaction
        conn = op.get_bind()
        with autocommit_block(conn) as raw:
            raw.execute("CREATE INDEX CONCURRENTLY ...")

    The context manager flips the underlying psycopg connection to
    autocommit mode on entry and restores the prior mode on exit -- on
    success and on exception both. Capturing the prior value (rather than
    blindly resetting to ``False``) keeps the helper defensive against a
    future psycopg version that starts in a different mode.

    The context-manager shape also makes the autocommit-True window
    obvious in diff and easier to spot if a future psycopg version
    changes the underlying mechanism.

    Pinned to psycopg 3.x semantics: ``conn.connection.autocommit = True/False``.
    If the pyproject psycopg pin moves to 4.x, audit this contract -- 4.x
    may switch to ``set_autocommit()`` method form or context-manager-only.

    Args:
        conn: SQLAlchemy connection from ``op.get_bind()``. Its
            ``.connection`` attribute is the raw psycopg connection.

    Yields:
        The raw psycopg connection (autocommit=True) for direct
        ``.execute(...)`` calls.
    """
    raw = conn.connection  # psycopg.Connection
    prev = getattr(raw, "autocommit", False)
    raw.autocommit = True
    try:
        yield raw
    finally:
        raw.autocommit = prev
