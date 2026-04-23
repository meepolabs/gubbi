"""One-time backfill: encrypt legacy plaintext columns with ContentCipher.

Usage:
    JOURNAL_ENCRYPTION_MASTER_KEY_V1="base64-key..." \\
        JOURNAL_DATABASE_URL_ADMIN="postgresql://admin@localhost:5432/journal" \\
        poetry run python -m journalctl.scripts.backfill_encrypt

Add ``--dry-run`` to preview counts without writing.

What it does:
1. Read every entries.content, entries.reasoning, messages.content row
   where plaintext IS NOT NULL and the matching encrypted column is NULL.
2. Encrypt each value with ContentCipher (AES-256-GCM).
3. Write encrypted + nonce columns (plus search_text for content columns).
4. Verify zero rows remain unencrypted before exiting 0.

Idempotent: the ``<col>_encrypted IS NULL`` WHERE clause means re-running
after a partial failure skips already-encrypted rows.

Safe to run against a live read-serving DB: the script only writes to
the new encrypted / nonce / search_text columns, never touches the
legacy plaintext, and uses the admin (BYPASSRLS) DSN so it sees every
tenant's rows without going through the app pool. Dual-read in the repo
layer means existing readers see consistent data throughout.

Post-script: after this run exits 0 on every environment, apply
migration 0008 (``alembic upgrade head``) to drop the legacy plaintext
columns. Running 0008 before a clean backfill will drop rows whose
content was never encrypted.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

import asyncpg

from journalctl.core.crypto import ContentCipher, load_master_keys_from_env
from journalctl.storage.pg_setup import _init_connection


def _fail(msg: str) -> None:
    sys.stderr.write(f"{msg}\n")
    sys.exit(1)


async def _count(pool: asyncpg.Pool, table: str, col: str, enc_col: str) -> int:
    """Return the number of plaintext rows still awaiting encryption."""
    query = f"SELECT count(*) FROM {table} WHERE {col} IS NOT NULL AND {enc_col} IS NULL"  # noqa: S608 -- identifiers are call-site literals
    return int(await pool.fetchval(query))


async def _encrypt_pass(
    pool: asyncpg.Pool,
    cipher: ContentCipher,
    label: str,
    table: str,
    col: str,
    enc_col: str,
    nonce_col: str,
    also_write_search_text: bool,
    dry_run: bool,
) -> int:
    """Batch-encrypt every row with plaintext set and ciphertext NULL.

    ``also_write_search_text`` is True for the two content passes (entries
    and messages) because search_text is derived from content. It is False
    for the reasoning pass; reasoning has no FTS role and no search_text
    column of its own.
    """
    total = await _count(pool, table, col, enc_col)
    print(f"{label}: {total} rows to encrypt")  # noqa: T201
    if total == 0 or dry_run:
        return 0

    encrypted = 0
    batch_size = 1000
    total_batches = (total + batch_size - 1) // batch_size
    batch_num = 0

    if also_write_search_text:
        update_sql = (
            f"UPDATE {table} "  # noqa: S608 -- identifiers are call-site literals
            f"SET {enc_col} = $1, {nonce_col} = $2, search_text = $3 WHERE id = $4"
        )
    else:
        update_sql = (
            f"UPDATE {table} "  # noqa: S608 -- identifiers are call-site literals
            f"SET {enc_col} = $1, {nonce_col} = $2 WHERE id = $3"
        )

    select_sql = (
        f"SELECT id, {col} AS plaintext FROM {table} "  # noqa: S608 -- identifiers are call-site literals
        f"WHERE {col} IS NOT NULL AND {enc_col} IS NULL ORDER BY id LIMIT {batch_size}"
    )

    while True:
        async with pool.acquire() as conn:
            rows = await conn.fetch(select_sql)
            if not rows:
                break
            batch_num += 1
            async with conn.transaction():
                for r in rows:
                    plaintext = r["plaintext"]
                    ct, nonce = cipher.encrypt(plaintext)
                    if also_write_search_text:
                        await conn.execute(update_sql, ct, nonce, plaintext, r["id"])
                    else:
                        await conn.execute(update_sql, ct, nonce, r["id"])
                    encrypted += 1
        pct = encrypted * 100 // total if total else 100
        sys.stderr.write(
            f"  {label} [{pct}%] {encrypted}/{total} " f"(batch {batch_num}/{total_batches})\n"
        )
    return encrypted


async def _run(pool: asyncpg.Pool, cipher: ContentCipher, dry_run: bool) -> None:
    t0 = time.monotonic()

    ec = await _encrypt_pass(
        pool,
        cipher,
        "entries.content",
        table="entries",
        col="content",
        enc_col="content_encrypted",
        nonce_col="content_nonce",
        also_write_search_text=True,
        dry_run=dry_run,
    )
    er = await _encrypt_pass(
        pool,
        cipher,
        "entries.reasoning",
        table="entries",
        col="reasoning",
        enc_col="reasoning_encrypted",
        nonce_col="reasoning_nonce",
        also_write_search_text=False,
        dry_run=dry_run,
    )
    mc = await _encrypt_pass(
        pool,
        cipher,
        "messages.content",
        table="messages",
        col="content",
        enc_col="content_encrypted",
        nonce_col="content_nonce",
        also_write_search_text=True,
        dry_run=dry_run,
    )

    # Verification: every plaintext row should now have a ciphertext.
    if not dry_run:
        for label, table, col, enc_col in [
            ("entries.content", "entries", "content", "content_encrypted"),
            ("entries.reasoning", "entries", "reasoning", "reasoning_encrypted"),
            ("messages.content", "messages", "content", "content_encrypted"),
        ]:
            remaining = await _count(pool, table, col, enc_col)
            if remaining > 0:
                _fail(f"VERIFICATION FAILED: {label} has {remaining} unencrypted rows")

    elapsed = time.monotonic() - t0
    print(f"entries.content:    {ec} rows encrypted")  # noqa: T201
    print(f"entries.reasoning:  {er} rows encrypted")  # noqa: T201
    print(f"messages.content:   {mc} rows encrypted")  # noqa: T201
    print(f"Elapsed:            {elapsed:.1f}s")  # noqa: T201
    if dry_run:
        print("(dry-run: no rows were modified)")  # noqa: T201


def _resolve_admin_dsn() -> str:
    dsn = os.environ.get("JOURNAL_DB_ADMIN_URL")
    if dsn:
        return dsn
    try:
        from journalctl.config import get_settings  # noqa: PLC0415

        settings = get_settings()
        if settings.db_admin_url:
            return str(settings.db_admin_url)
    except Exception as exc:
        sys.stderr.write(f"Could not resolve admin DSN from settings: {exc}\n")
    _fail(
        "JOURNAL_DB_ADMIN_URL environment variable is required. "
        "This DSN provides the BYPASSRLS role needed to read all tenants' "
        "rows. Set it before running the backfill."
    )
    raise RuntimeError("unreachable")  # pragma: no cover


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill plaintext content/reasoning columns with AES-256-GCM encryption."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report row counts, do not modify any rows.",
    )
    args = parser.parse_args()

    admin_dsn = _resolve_admin_dsn()
    keys = load_master_keys_from_env(os.environ)
    if not keys:
        _fail(
            "No master encryption keys found. Set JOURNAL_ENCRYPTION_MASTER_KEY_V1 "
            "(or any V<N>) to a base64-encoded 32-byte key before running."
        )
    cipher = ContentCipher(keys)

    async def _entry() -> None:
        pool: asyncpg.Pool = await asyncpg.create_pool(
            admin_dsn,
            statement_cache_size=0,
            init=_init_connection,
            min_size=1,
            max_size=2,
        )
        try:
            await _run(pool, cipher, args.dry_run)
        except Exception as exc:
            # Log only exception TYPE + message -- the full traceback can
            # include plaintext query parameters captured in the asyncpg stack
            # frames, which would leak tenant content through stderr/log
            # aggregators. Operators who need the full traceback can set
            # BACKFILL_TRACEBACK=1.
            sys.stderr.write(f"BACKFILL FAILED: {type(exc).__name__}: {exc}\n")
            if os.environ.get("BACKFILL_TRACEBACK") == "1":
                import traceback  # noqa: PLC0415

                traceback.print_exc(file=sys.stderr)
            sys.exit(1)
        finally:
            await pool.close()

    asyncio.run(_entry())


if __name__ == "__main__":
    main()
