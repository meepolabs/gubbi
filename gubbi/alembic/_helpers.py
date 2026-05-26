"""Shared helpers for alembic migrations.

This module centralises low-level dance steps that several migrations need
but that are too small to live in a third-party package (gubbi-common would
introduce cross-repo coupling for migrations).

The current API surface is one context manager:

    autocommit_block(conn) -> Iterator[psycopg.Connection]

which is used by ``CREATE INDEX CONCURRENTLY`` migrations to step out of
Alembic's wrapping transaction safely.

Pinned to psycopg 3.x semantics: the underlying psycopg connection exposes
``autocommit`` as a settable property. The change that introduced
``driver_connection`` resolution removed the proxy-vs-driver coupling, but
the contract still depends on psycopg's ``autocommit`` property setter -- a
psycopg 4.0 bump may switch to ``set_autocommit()`` method form or
context-manager-only, which would require updating this helper.
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

    Why ``driver_connection`` is required:
        ``op.get_bind()`` returns a SQLAlchemy ``Connection`` whose
        ``.connection`` attribute is a ``_ConnectionFairy``
        (PoolProxiedConnection), NOT the raw psycopg connection. The
        fairy does not define ``__setattr__`` to forward writes -- so
        ``proxy.autocommit = True`` lands on the proxy instance and the
        underlying psycopg connection's ``autocommit`` stays at its prior
        value (typically False). The result: ``CREATE INDEX CONCURRENTLY``
        can fail mid-migration because Alembic's transaction is still
        open on the real connection.

        SQLAlchemy 2.0's canonical accessor for the underlying DB-API
        connection on a proxy is ``driver_connection``;
        ``dbapi_connection`` is the SQLAlchemy 1.x alias and is kept here
        as a fallback. The final ``or proxy`` fallback covers tests that
        pass a raw psycopg-shaped fake (no proxy wrapper).

    Pinned to psycopg 3.x semantics: ``raw.autocommit = True/False``.
    If the pyproject psycopg pin moves to 4.x, audit this contract -- 4.x
    may switch to ``set_autocommit()`` method form or context-manager-only.

    Args:
        conn: SQLAlchemy connection from ``op.get_bind()``. Its
            ``.connection`` attribute is a PoolProxiedConnection
            (``_ConnectionFairy``) whose ``.driver_connection`` exposes
            the raw psycopg connection.

    Yields:
        The raw psycopg connection (autocommit=True) for direct
        ``.execute(...)`` calls.
    """
    proxy = conn.connection
    # SQLAlchemy 2.0 PoolProxiedConnection: ``driver_connection`` is the
    # canonical accessor for the underlying psycopg connection. The fairy
    # does not forward attribute writes via ``__setattr__``, so writing
    # ``autocommit`` on the proxy itself silently stays on the proxy and
    # never reaches psycopg. ``dbapi_connection`` is the SQLAlchemy 1.x
    # alias. The final ``or proxy`` fallback covers tests that pass a raw
    # psycopg-shaped fake.
    raw = (
        getattr(proxy, "driver_connection", None)
        or getattr(proxy, "dbapi_connection", None)
        or proxy
    )
    prev = getattr(raw, "autocommit", False)
    raw.autocommit = True
    try:
        yield raw
    finally:
        raw.autocommit = prev
