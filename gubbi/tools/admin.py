"""Reindex primitives (library). Used by the admin API; no longer exposed as an MCP tool."""

import asyncio
import logging
import time
from datetime import UTC
from datetime import datetime as datetime_cls

import asyncpg

from gubbi.core.context import AppContext
from gubbi.core.crypto import ContentCipher
from gubbi.storage.repositories import entries as entry_repo
from gubbi.tools.constants import REINDEX_BATCH_SIZE

logger = logging.getLogger(__name__)

# PostgreSQL advisory lock key for reindex coordination.
# Must be unique across all advisory locks used by this application so that a
# concurrent reindex never collides with backup, migration, or maintenance
# locks picked by other subsystems.  The value is a large prime (no special
# meaning beyond "not any other constant we already use"); any integer >= 2^31
# works but must be positive to avoid conflicting with PG's negative-range API.
_REINDEX_ADVISORY_LOCK_KEY: int = 2048976971  # large prime, not pi

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


async def _run_reindex(app_ctx: AppContext, pool: asyncpg.Pool, cipher: ContentCipher) -> dict:
    """Rebuild semantic embeddings for unindexed entries.

    Callers MUST acquire ``pg_try_advisory_lock(_REINDEX_ADVISORY_LOCK_KEY)``
    before invoking and release it after; this function has no internal
    concurrency protection. The cooldown check in ``_db_reindex_cooldown``
    is time-based and does not serialize concurrent callers on its own.
    """
    start = time.monotonic()

    # tsvector columns are GENERATED ALWAYS — always up-to-date.
    # Only needs to rebuild semantic embeddings (tsvector stays current).
    async with pool.acquire() as conn:
        await entry_repo.reset_indexed_at(conn)

    embeddings_generated = 0
    embeddings_failed = 0
    last_id = 0
    semantic_status = "ok"

    while True:
        async with pool.acquire() as conn:
            batch = await entry_repo.get_unindexed(conn, cipher, last_id, REINDEX_BATCH_SIZE)

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
