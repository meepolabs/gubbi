"""Backfill entries/messages search_vector from decrypted content.

One-shot deploy script paired with migrations 0013 + 0014. After every prod
+ dev DB has been backfilled and 0014 has landed, **delete this file**.

Lives in ``deployment/`` rather than ``journalctl/scripts/`` because this is
a one-shot tool meant to be deleted after operator runs it on prod (matches
``cleanup_encrypted_xml_spill.py`` pattern).

Usage::

    cd journalctl
    JOURNAL_ENCRYPTION_MASTER_KEY_V1="base64-key..." \\
        JOURNAL_DB_ADMIN_URL="postgresql://admin@localhost:5432/journal" \\
        poetry run python deployment/backfill_to_tsvector.py

Add ``--dry-run`` to print pending row counts without writes.
Add ``--verify`` to compare stored tsvector values against recomputed values.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

import asyncpg

from journalctl.core.crypto import ContentCipher, decrypt_or_raise, load_master_keys_from_env
from journalctl.storage.pg_setup import _init_connection

_BATCH_SIZE = 1000
_VERIFY_SAMPLE = 50


def _fail(msg: str) -> None:
    sys.stderr.write(f"{msg}\n")
    sys.exit(1)


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
    _fail("JOURNAL_DB_ADMIN_URL is required for cross-tenant backfill.")
    raise RuntimeError("unreachable")


async def _count_pending(pool: asyncpg.Pool, table: str) -> int:
    query = f"SELECT COUNT(*) FROM {table} WHERE search_vector IS NULL"  # noqa: S608
    return int(await pool.fetchval(query) or 0)


async def _backfill_table(
    pool: asyncpg.Pool,
    cipher: ContentCipher,
    *,
    table: str,
    dry_run: bool,
) -> int:
    total = await _count_pending(pool, table)
    print(f"{table}: {total} rows pending")  # noqa: T201
    if total == 0 or dry_run:
        return 0

    select_sql = (
        f"SELECT id, content_encrypted, content_nonce FROM {table} "  # noqa: S608
        "WHERE search_vector IS NULL ORDER BY id LIMIT $1"
    )
    update_sql = f"UPDATE {table} SET search_vector = to_tsvector('english', $1) WHERE id = $2"  # noqa: S608

    done = 0
    batches = (total + _BATCH_SIZE - 1) // _BATCH_SIZE
    batch_num = 0
    while True:
        async with pool.acquire() as conn:
            rows = await conn.fetch(select_sql, _BATCH_SIZE)
            if not rows:
                break
            batch_num += 1
            async with conn.transaction():
                for row in rows:
                    plaintext = decrypt_or_raise(
                        cipher,
                        bytes(row["content_encrypted"]),
                        bytes(row["content_nonce"]),
                    )
                    await conn.execute(update_sql, plaintext, int(row["id"]))
                    done += 1
        pct = done * 100 // total if total else 100
        sys.stderr.write(f"  {table} [{pct}%] {done}/{total} (batch {batch_num}/{batches})\n")

    return done


async def _verify_table(pool: asyncpg.Pool, cipher: ContentCipher, *, table: str) -> None:
    async with pool.acquire() as conn:
        query = f"SELECT COUNT(*) FROM {table} WHERE search_vector IS NULL"  # noqa: S608
        pending = int(await conn.fetchval(query) or 0)
        if pending:
            _fail(f"VERIFY FAILED: {table} has {pending} rows with NULL search_vector")
        rows = await conn.fetch(
            f"SELECT id, content_encrypted, content_nonce, search_vector::text AS sv "  # noqa: S608
            f"FROM {table} ORDER BY id DESC LIMIT $1",
            _VERIFY_SAMPLE,
        )
        for row in rows:
            plaintext = decrypt_or_raise(
                cipher,
                bytes(row["content_encrypted"]),
                bytes(row["content_nonce"]),
            )
            expected = await conn.fetchval("SELECT to_tsvector('english', $1)::text", plaintext)
            if str(row["sv"] or "") != str(expected or ""):
                _fail(f"VERIFY FAILED: {table} id={int(row['id'])} has mismatched search_vector")


async def _run(pool: asyncpg.Pool, cipher: ContentCipher, dry_run: bool, verify: bool) -> None:
    t0 = time.monotonic()
    entries_done = await _backfill_table(pool, cipher, table="entries", dry_run=dry_run)
    messages_done = await _backfill_table(pool, cipher, table="messages", dry_run=dry_run)

    if verify and not dry_run:
        await _verify_table(pool, cipher, table="entries")
        await _verify_table(pool, cipher, table="messages")

    elapsed = time.monotonic() - t0
    print(f"entries:  {entries_done} rows updated")  # noqa: T201
    print(f"messages: {messages_done} rows updated")  # noqa: T201
    print(f"elapsed:  {elapsed:.1f}s")  # noqa: T201
    if dry_run:
        print("(dry-run: no rows were modified)")  # noqa: T201


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill search_vector for entries/messages.")
    parser.add_argument("--dry-run", action="store_true", help="Print counts without writing rows.")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Recompute and compare tsvector values on a sample after writes.",
    )
    args = parser.parse_args()

    keys = load_master_keys_from_env(os.environ)
    if not keys:
        _fail("No encryption master keys found in JOURNAL_ENCRYPTION_MASTER_KEY_V* env vars.")
    cipher = ContentCipher(keys)
    admin_dsn = _resolve_admin_dsn()

    async def _entry() -> None:
        pool = await asyncpg.create_pool(
            admin_dsn,
            statement_cache_size=0,
            init=_init_connection,
            min_size=1,
            max_size=2,
        )
        try:
            await _run(pool, cipher, args.dry_run, args.verify)
        except Exception as exc:
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
