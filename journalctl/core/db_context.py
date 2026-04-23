"""User-scoped DB connection helper for RLS (TASK-02.06).

Every request handled as a tenant user wraps its DB work with
``user_scoped_connection``. The helper:

1. Acquires a connection from the app pool (``journal_app`` role — NOT BYPASSRLS).
2. Opens a transaction.
3. Sets ``app.current_user_id`` (the GUC read by RLS policies) and
   ``hnsw.ef_search`` (raised to compensate for post-index RLS filtering)
   via ``SELECT set_config(..., true)`` — the parameter-safe, transaction-
   scoped equivalent of ``SET LOCAL``.
4. Yields the scoped connection.

``SET LOCAL name = value`` cannot be parameterized in asyncpg because ``SET``
is a PostgreSQL utility statement, not a DML query. ``set_config(name, value,
is_local=true)`` is the SQL-function form that accepts bound parameters,
which both avoids injection risk and keeps the statement prepared-plan safe.
``user_id`` is additionally validated as a ``UUID`` instance before the call.

Admin/worker paths that must cross tenants (``journal_reindex``, cleanup jobs)
connect via a separate BYPASSRLS pool — see ``AppContext.admin_pool`` — and
do NOT use this helper.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID

import asyncpg

from journalctl.core.auth_context import current_user_id

logger = logging.getLogger(__name__)

# Pgvector HNSW default ef_search is 40. With RLS the HNSW scan returns
# candidates without user_id awareness; the RLS filter then drops rows
# belonging to other tenants. A larger ef_search gives the post-filter
# more candidates to keep, so semantic-search recall survives multi-tenancy.
# 100 is conservative — tune once multi-tenant scale metrics exist.
DEFAULT_HNSW_EF_SEARCH = 100


class MissingUserIdError(RuntimeError):
    """Raised when a scoped connection is requested without an authenticated user."""


@asynccontextmanager
async def user_scoped_connection(
    pool: asyncpg.Pool,
    user_id: UUID | None = None,
    *,
    hnsw_ef_search: int = DEFAULT_HNSW_EF_SEARCH,
) -> AsyncIterator[asyncpg.Connection]:
    """Acquire a pool connection with ``app.current_user_id`` bound to ``user_id``.

    Parameters
    ----------
    pool:
        The app-role asyncpg pool. Must connect as a role WITHOUT
        ``BYPASSRLS`` (``journal_app``) or RLS silently no-ops.
    user_id:
        UUID of the tenant user. If ``None``, reads from the
        ``current_user_id`` ContextVar set by ``BearerAuthMiddleware``.
        Raises ``MissingUserIdError`` if neither is set — default-deny
        rather than silently bind to NULL.
    hnsw_ef_search:
        Value for the transaction-scoped ``hnsw.ef_search`` GUC. Applies
        to every pgvector HNSW scan inside the yielded transaction.

    Both GUCs are set via ``set_config(..., is_local=true)``. They are
    automatically cleared at COMMIT or ROLLBACK, so no connection state
    leaks back into the pool when the connection is returned.
    """
    resolved = user_id if user_id is not None else current_user_id.get()
    if resolved is None:
        # Log before raising so the operator can tell an auth-misconfig
        # (operator_user_id=None, operator-identity auth path) from a code bug
        # (tool handler skipped the helper and called .get() elsewhere).
        logger.error(
            "user_scoped_connection: no authenticated user",
            extra={"user_id_arg_provided": user_id is not None},
        )
        raise MissingUserIdError(
            "user_scoped_connection called with no authenticated user — "
            "check BearerAuthMiddleware wiring or pass user_id explicitly"
        )
    if not isinstance(resolved, UUID):
        # Defense-in-depth: ContextVar and signature are typed but not runtime-enforced.
        raise TypeError(f"user_id must be UUID, got {type(resolved).__name__}")

    # pgvector permits hnsw.ef_search in [1, 1000]; reject nonsense early so the
    # failure surface is a clear ValueError instead of an asyncpg GUC-parse error.
    ef_search = int(hnsw_ef_search)
    if not (1 <= ef_search <= 1000):
        raise ValueError(f"hnsw_ef_search must be in [1, 1000], got {hnsw_ef_search}")

    async with pool.acquire() as conn, conn.transaction():
        await conn.execute(
            "SELECT set_config('app.current_user_id', $1, true)",
            str(resolved),
        )
        await conn.execute(
            "SELECT set_config('hnsw.ef_search', $1, true)",
            str(ef_search),
        )
        yield conn
