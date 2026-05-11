"""Reindex primitives (library). Used by the admin API; no longer exposed as an MCP tool."""

import asyncio
import time
from datetime import UTC
from datetime import datetime as datetime_cls
from typing import Any

import asyncpg
import structlog

from gubbi.app_context import AppContext
from gubbi.crypto.cipher import ContentCipher
from gubbi.storage.connection import safe_acquire
from gubbi.storage.repositories import entries as entry_repo
from gubbi.tools.constants import REINDEX_BATCH_SIZE

# All functions are private; registered via MCP tool registry, not direct import.
__all__: list[str] = []

logger = structlog.get_logger(__name__)

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

    Uses MAX(indexed_at) from entries as a shared proxy for last reindex time -
    accurate across all workers since the value lives in PostgreSQL.
    """
    async with safe_acquire(pool) as conn:
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


async def _run_reindex(
    app_ctx: AppContext, admin_pool: asyncpg.Pool, cipher: ContentCipher
) -> dict[str, Any]:
    """Process the unindexed-entries queue (entries.indexed_at IS NULL).

    Does NOT reset already-indexed rows. Use ``_reset_all_indexed_at`` for
    the wipe-and-re-embed-everything path.

    Callers MUST acquire ``pg_try_advisory_lock(_REINDEX_ADVISORY_LOCK_KEY)``
    before invoking and release it after; the advisory lock is kept as
    defense-in-depth so a stray manual ``_run_reindex`` invocation does
    not race the production caller. The cooldown check in
    ``_db_reindex_cooldown`` is time-based and does not serialize
    concurrent callers on its own.

    The ``FOR UPDATE SKIP LOCKED`` clause in ``get_unindexed`` is the
    correctness boundary for multi-worker safety; the advisory lock at
    the entry of this function reduces contention by preventing duplicate
    scan passes.

    Each iteration uses claim-then-process semantics so the row-level
    lock is held only for the SELECT (the embedding encode is CPU-bound
    via ONNX and must not pin a connection):

    1. Open a single transaction; ``get_unindexed`` issues
       ``FOR UPDATE SKIP LOCKED`` to take row locks; immediately
       stamp ``indexed_at = now()`` for the claimed ids via
       ``mark_indexed_batch``. Commit on transaction exit.
    2. Encode + ``save_by_vector`` happen OUTSIDE the transaction
       (acquire-per-save preserved).
    3. On encode/save failure, accumulate the failed ids and run a
       compensating ``reset_indexed_at_for_ids`` so the next reindex
       picks them up again. ``last_id`` advances past the failed ids
       only after the compensating reset succeeds; if the reset itself
       fails, the error is logged at ERROR level and re-raised so the
       caller can see the partial-state and retry on the same cursor.

    The ``admin_pool`` parameter MUST be the BYPASSRLS pool: this
    function issues UPDATEs that span multiple users
    (``mark_indexed_batch``, ``reset_indexed_at_for_ids``) and
    ``get_unindexed`` reads encrypted content across all tenants.
    A user-scoped (RLS-enforced) pool would silently drop rows.
    """
    # Hard guard (not ``assert`` -- ``-O`` strips asserts and the BYPASSRLS
    # contract is security-relevant). Both the supplied pool AND the
    # context's pool must be present and identical -- a vacuous
    # ``None == None`` would have passed the previous assert in
    # single-tenant dev mode where ``app_ctx.admin_pool`` is typed
    # ``Pool | None``.
    if admin_pool is None or admin_pool is not app_ctx.admin_pool:
        raise ValueError(
            "_run_reindex requires app_ctx.admin_pool (BYPASSRLS) -- "
            "mark_indexed_batch and reset_indexed_at_for_ids issue UPDATEs "
            "across users."
        )

    start = time.monotonic()

    embeddings_generated = 0
    embeddings_failed = 0
    last_id = 0
    semantic_status = "ok"

    while True:
        # Claim step: SELECT FOR UPDATE SKIP LOCKED + mark_indexed_batch
        # in one transaction so concurrent callers cannot reclaim the same
        # rows. The row locks release on commit; from here on the rows are
        # protected by the ``indexed_at = now()`` stamp.
        async with safe_acquire(admin_pool) as conn, conn.transaction():
            batch = await entry_repo.get_unindexed(conn, cipher, last_id, REINDEX_BATCH_SIZE)
            if batch:
                await entry_repo.mark_indexed_batch(conn, [r["id"] for r in batch])

        if not batch:
            break

        succeeded_ids: list[int] = []
        failed_ids: list[int] = []
        for r in batch:
            try:
                content = r["content"] or ""
                # Encode outside the connection acquire - ONNX inference is CPU-bound
                # (10-200ms) and should not hold a pool connection during that time.
                embedding = await asyncio.to_thread(app_ctx.embedding_service.encode, content)
                async with safe_acquire(admin_pool) as conn:
                    await app_ctx.embedding_service.save_by_vector(conn, r["id"], embedding)
                succeeded_ids.append(r["id"])
                embeddings_generated += 1
            except Exception as exc:
                embeddings_failed += 1
                failed_ids.append(r["id"])
                await logger.warning(
                    "Failed to embed entry during reindex",
                    entry_id=r["id"],
                    error=str(exc),
                    exc_info=True,
                )

        # Compensating reset: clear ``indexed_at`` for ids whose encode/save
        # failed so a future reindex retries them. If the reset itself
        # fails we log at ERROR level and re-raise -- the upstream caller
        # must see the partial-state failure rather than have ``last_id``
        # silently strand the failed batch past its retry window.
        if failed_ids:
            try:
                async with safe_acquire(admin_pool) as conn:
                    await entry_repo.reset_indexed_at_for_ids(conn, failed_ids)
            except Exception as exc:
                await logger.error(
                    "Compensating reset_indexed_at_for_ids failed; "
                    "not advancing last_id past failed batch",
                    failed_ids=failed_ids,
                    last_id=last_id,
                    error=str(exc),
                    exc_info=True,
                )
                raise

        # Advance ``last_id`` only past rows we actually finished. If the
        # whole batch failed, the compensating reset cleared ``indexed_at``
        # on every id in the batch -- a naive ``last_id = batch[-1]["id"]``
        # would put the cursor past those reset rows, and the next
        # iteration's ``get_unindexed`` (cursor: ``id > $1``) would skip
        # them forever. Stranded rows must keep showing up to the next
        # claim pass.
        if succeeded_ids:
            last_id = max(succeeded_ids)
        else:
            # Whole-batch failure with successful compensating reset: the
            # rows we just reset still have ``id <= batch[-1]["id"]``, so
            # leaving ``last_id`` put would cause the next ``get_unindexed``
            # call to re-claim the same batch and (likely) keep failing on
            # the same encode error. Break out so the upstream caller can
            # observe the partial-state and decide whether to retry. The
            # row-level ``indexed_at IS NULL`` state ensures a future
            # reindex pass picks them up; this loop just stops grinding.
            await logger.error(
                "Whole reindex batch failed; aborting loop to avoid infinite retry "
                "on the same cursor (rows remain with indexed_at IS NULL for next pass)",
                last_id=last_id,
                failed_ids=failed_ids,
            )
            break

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


async def _reset_all_indexed_at(admin_pool: asyncpg.Pool, app_ctx: AppContext) -> dict[str, int]:
    """Wipe ``indexed_at`` on every non-deleted entry so the next reindex re-embeds them all.

    This is the explicit "wipe and re-embed everything" admin action that
    used to live as an implicit side effect at the head of ``_run_reindex``.
    It is split out so callers must opt in: a normal ``_run_reindex`` no
    longer destroys already-indexed rows.

    Note: intentionally private. Not yet wired as an MCP tool; the
    ``_run_reindex`` docstring refers to ``_reset_all_indexed_at`` (with
    underscore) so the symbol name actually exists. Future work: register
    as an admin-only MCP tool. For now, callers invoke from a Python REPL
    or admin script.

    Acquires ``pg_try_advisory_xact_lock(_REINDEX_ADVISORY_LOCK_KEY)``
    (transaction-scoped) so the lock is released atomically with the
    UPDATE on commit/rollback. Session-scoped advisory locks were used
    previously, but a manual ``pg_advisory_unlock`` in ``finally`` could
    leak the lock if ``unlock`` itself raised, and asyncpg pool init/reset
    callbacks running ``RESET ALL`` between checkouts can silently release
    session locks. The xact-scoped variant ties the lock lifetime to the
    transaction, removing both hazards.

    Note: ``pg_try_advisory_xact_lock`` shares the same lock-key namespace
    as session-level ``pg_try_advisory_lock``, so existing session-held
    locks (e.g. a concurrent ``_run_reindex`` caller) still serialize.

    The ``admin_pool`` parameter MUST be the BYPASSRLS pool: ``reset_indexed_at``
    issues a single cross-user UPDATE so a user-scoped (RLS-enforced) pool
    would silently restrict the rowcount to zero. ``app_ctx`` is required
    (no default) so the identity check ``admin_pool is app_ctx.admin_pool``
    runs unconditionally -- the previous ``app_ctx: AppContext | None = None``
    signature let callers pass an RLS-scoped pool with ``app_ctx=None`` and
    bypass the security contract entirely. Mirrors ``_run_reindex``'s guard.
    """
    # Hard guard (not ``assert`` -- ``-O`` strips asserts and the BYPASSRLS
    # contract is security-relevant). Both the supplied pool AND the
    # context's pool must be present and identical -- a vacuous
    # ``None == None`` would have passed the previous assert in
    # single-tenant dev mode where ``app_ctx.admin_pool`` is typed
    # ``Pool | None``.
    if admin_pool is None or admin_pool is not app_ctx.admin_pool:
        raise ValueError(
            "_reset_all_indexed_at requires app_ctx.admin_pool (BYPASSRLS) -- "
            "reset_indexed_at issues a cross-user UPDATE that a user-scoped "
            "(RLS-enforced) pool would silently restrict to zero rows."
        )

    async with safe_acquire(admin_pool) as conn, conn.transaction():
        locked = await conn.fetchval(
            "SELECT pg_try_advisory_xact_lock($1)", _REINDEX_ADVISORY_LOCK_KEY
        )
        if not locked:
            # Returning early from inside the transaction commits an empty
            # tx; the xact-scoped lock would not have been taken anyway.
            return {"status_locked": 1, "rows_reset": 0}
        # Mirror reset_indexed_at's WHERE clause to count affected rows;
        # the helper itself does not return a count. Run the SELECT first
        # so the count reflects rows that would be reset.
        rows_reset = int(
            await conn.fetchval(
                "SELECT count(*) FROM entries "
                "WHERE deleted_at IS NULL AND indexed_at IS NOT NULL"
            )
            or 0
        )
        await entry_repo.reset_indexed_at(conn)

    return {"status_locked": 0, "rows_reset": rows_reset}
