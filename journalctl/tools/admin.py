"""MCP tool: journal_reindex."""

import asyncio
import logging
import time
from datetime import UTC
from datetime import datetime as datetime_cls

import asyncpg
from mcp.server.fastmcp import FastMCP

from journalctl.core.context import AppContext
from journalctl.storage import pg_setup
from journalctl.storage.repositories import entries as entry_repo
from journalctl.tools.constants import REINDEX_BATCH_SIZE

logger = logging.getLogger(__name__)

_REINDEX_ADVISORY_LOCK_KEY = 3141592653  # stable app-wide key
_REINDEX_COOLDOWN_SECONDS = 60


async def _db_reindex_cooldown(pool: asyncpg.Pool) -> int | None:
    """Return seconds until cooldown expires, or None if reindex is allowed.

    Uses MAX(indexed_at) from entries as a shared proxy for last reindex time —
    accurate across all workers since the value lives in PostgreSQL.
    """
    async with pool.acquire() as conn:
        max_indexed = await entry_repo.get_max_indexed_at(conn)
    if max_indexed is None:
        return None
    now_utc = datetime_cls.now(UTC)
    if max_indexed.tzinfo is None:
        max_indexed = max_indexed.replace(tzinfo=UTC)
    elapsed = (now_utc - max_indexed).total_seconds()
    if elapsed < _REINDEX_COOLDOWN_SECONDS:
        return int(_REINDEX_COOLDOWN_SECONDS - elapsed)
    return None


def register(mcp: FastMCP, app_ctx: AppContext) -> None:
    """Register admin tools on the MCP server."""

    @mcp.tool()
    async def journal_reindex() -> dict:
        """Repair the search index — use when journal_search returns wrong or missing results.

        Only needed if search results seem stale or incomplete.
        Rarely needed during normal use.

        Returns:
            Number of embeddings generated and duration.
        """
        # Reindex rebuilds embeddings for ALL tenants' entries, so it must run
        # under the BYPASSRLS admin pool. Fallback to app pool is only safe in
        # single-tenant dev (pre-RLS). Once 02.05 is applied in prod, running
        # reindex against the app pool would silently process zero rows, so
        # log a loud warning when the fallback is taken.
        pool = app_ctx.admin_pool
        if pool is None:
            logger.warning(
                "journal_reindex: admin_pool not configured — falling back to app pool. "
                "With RLS active this will process zero rows. "
                "Set JOURNAL_DATABASE_URL_ADMIN (BYPASSRLS DSN) to fix."
            )
            pool = app_ctx.pool
        retry_after = await _db_reindex_cooldown(pool)
        if retry_after is not None:
            return {
                "status": "cooldown",
                "message": f"Reindex completed recently. Try again in {retry_after}s.",
                "retry_after": retry_after,
            }

        async with pool.acquire() as lock_conn:
            acquired = await pg_setup.try_advisory_lock(lock_conn, _REINDEX_ADVISORY_LOCK_KEY)
            if not acquired:
                return {
                    "status": "already_running",
                    "message": "A reindex is already in progress.",
                }
            try:
                return await _run_reindex(app_ctx, pool)
            finally:
                await pg_setup.advisory_unlock(lock_conn, _REINDEX_ADVISORY_LOCK_KEY)


async def _run_reindex(app_ctx: AppContext, pool: asyncpg.Pool) -> dict:
    start = time.monotonic()

    # tsvector columns are GENERATED ALWAYS — always up-to-date.
    # journal_reindex only needs to rebuild semantic embeddings.
    async with pool.acquire() as conn:
        await entry_repo.reset_indexed_at(conn)

    embeddings_generated = 0
    embeddings_failed = 0
    last_id = 0
    semantic_status = "ok"

    while True:
        async with pool.acquire() as conn:
            batch = await entry_repo.get_unindexed(conn, last_id, REINDEX_BATCH_SIZE)

        if not batch:
            break

        succeeded_ids: list[int] = []
        for r in batch:
            try:
                content = r["content"] or ""
                # Encode outside the connection acquire — ONNX inference is CPU-bound
                # (10-200ms) and should not hold a pool connection during that time.
                embedding = await asyncio.to_thread(app_ctx.embedding_service.encode, content)
                async with pool.acquire() as conn:
                    await app_ctx.embedding_service.store_by_vector(conn, r["id"], embedding)
                succeeded_ids.append(r["id"])
                embeddings_generated += 1
            except Exception as e:
                embeddings_failed += 1
                logger.warning(
                    "Failed to embed entry %s during reindex: %s",
                    r["id"],
                    e,
                    exc_info=True,
                )

        # Batch-mark all succeeded entries as indexed in one query
        if succeeded_ids:
            async with pool.acquire() as conn:
                await entry_repo.mark_indexed_batch(conn, succeeded_ids)

        last_id = batch[-1]["id"]

    if embeddings_failed:
        semantic_status = "partial"

    duration = round(time.monotonic() - start, 2)
    return {
        "status": "rebuilt",
        "semantic_status": semantic_status,
        "embeddings_generated": embeddings_generated,
        "embeddings_failed": embeddings_failed,
        "duration_seconds": duration,
    }
