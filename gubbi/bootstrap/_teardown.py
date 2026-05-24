"""Lifespan resource teardown helper.

One try/except wraps StartupRunner.run; on any exception (probe failure
or budget overrun) this helper closes resources in lifecycle-reverse
order, each guarded by ``suppress(Exception, asyncio.CancelledError)``.
Net effect: 8-9 duplicated teardown blocks in the old lifespan body
collapse to one helper call.

CancelledError is suppressed alongside Exception because it is a
BaseException (not Exception) since Python 3.8: a bare
``suppress(Exception)`` would let a cancelled close() escape and
clobber the original cancellation that the surrounding ``raise`` is
meant to re-raise.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import asyncpg
    import httpx
    import redis.asyncio as aioredis

    from gubbi.oauth.storage import OAuthStorage

__all__ = ["teardown_lifespan_resources"]


async def teardown_lifespan_resources(
    *,
    pool: asyncpg.Pool | None,
    admin_pool: asyncpg.Pool | None,
    redis_client: aioredis.Redis | None,
    redis_pool: aioredis.ConnectionPool | None,
    oauth_storage: OAuthStorage | None,
    hydra_http_client: httpx.AsyncClient | None,
) -> None:
    """Close every pre-yield resource the lifespan opened, best-effort.

    Order is lifecycle-reverse so a teardown that touches Redis (the
    audit fail-open path) lands before the underlying pool is closed.
    Each close is guarded so a teardown failure cannot mask the
    original exception or cancellation that triggered the unwind.
    """
    if redis_client is not None:
        with suppress(Exception, asyncio.CancelledError):
            await redis_client.aclose()
    if redis_pool is not None:
        with suppress(Exception, asyncio.CancelledError):
            await redis_pool.aclose()
    if oauth_storage is not None:
        with suppress(Exception, asyncio.CancelledError):
            await oauth_storage.close()
    if hydra_http_client is not None:
        with suppress(Exception, asyncio.CancelledError):
            await hydra_http_client.aclose()
    if pool is not None:
        with suppress(Exception, asyncio.CancelledError):
            await pool.close()
    if admin_pool is not None:
        with suppress(Exception, asyncio.CancelledError):
            await admin_pool.close()
