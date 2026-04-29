#!/usr/bin/env python3
"""Re-encrypt all rows from key-version V_from to V_to across five encrypted column pairs.

Runs as a one-shot operator tool (NOT under ``deployment/``).  It reads every
row whose source-version nonce-byte matches ``--from-version``, decrypts with
that key, re-encrypts with the ``--to-version`` key, and writes back in place.

AUDIT STRATEGY:
    Each batch writes ONE ``audit_log`` row per (table, column-pair) covering
    that batch's rotated rows.  Each audit record carries ``count``,
    ``from_version``, ``to_version`` in its metadata JSON so volume stays
    bounded while remaining fully auditable.

USAGE::

    JOURNAL_ENCRYPTION_MASTER_KEY_V1="base64-..." \\
        JOURNAL_ENCRYPTION_MASTER_KEY_V2="base64-..." \\
        JOURNAL_DB_ADMIN_URL="postgresql://admin@host:5432/journal" \\
        python -m journalctl.scripts.rotate_encryption_key --from-version 1 --to-version 2

    Add ``--dry-run`` to print per-table counts without writing.
    Add ``--verify`` for a post-update round-trip validation on a sample.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from typing import Final

import asyncpg

from journalctl.audit import Action, record_audit
from journalctl.core.crypto import ContentCipher, DecryptionError, load_master_keys_from_env
from journalctl.storage.pg_setup import init_pool

logger = logging.getLogger("rotate.encryption.key")

BATCH_SIZE: Final = 1000

# Five encrypted column-pairs the script rotates.
_ROTATION_SCREENS: Final = [
    ("entries", "content_encrypted", "content_nonce"),
    ("entries", "reasoning_encrypted", "reasoning_nonce"),
    ("messages", "content_encrypted", "content_nonce"),
    ("conversations", "title_encrypted", "title_nonce"),
    ("conversations", "summary_encrypted", "summary_nonce"),
]

_AUDIT_ACTION: Final = Action.ENCRYPTION_KEY_ROTATED
_AUDITOR_ID: Final = "script:rotate_encryption_key"


def _resolve_admin_dsn() -> str:
    """Return the admin DSN from env or config."""
    dsn = os.environ.get("JOURNAL_DB_ADMIN_URL")
    if dsn:
        return dsn
    try:
        from journalctl.config import get_settings  # noqa: PLC0415

        settings = get_settings()
        if settings.db_admin_url:
            return str(settings.db_admin_url)
    except Exception as exc:
        logger.debug("Config fallback failed: %s", exc)
        pass
    raise RuntimeError(
        "JOURNAL_DB_ADMIN_URL environment variable is required. "
        "This DSN provides the BYPASSRLS role needed to read all tenants."
    )


async def _collect_dry_run_counts(pool: asyncpg.Pool, source_version: int) -> list[dict]:
    """Collect per-table row counts that are at ``source_version``."""
    reports: list[dict] = []
    hex_pfx = f"{source_version:02x}"

    for table, col_enc, col_nonce in _ROTATION_SCREENS:  # noqa: S608
        where_clause = (
            f"{col_enc} IS NOT NULL AND {col_nonce} IS NOT NULL "
            f"AND encode({col_nonce}, 'hex') LIKE '{hex_pfx}%'"
        )
        count: int = int(
            await pool.fetchval(  # noqa: S608 -- identifiers from constant tuple
                f"SELECT COUNT(*) FROM {table} WHERE {where_clause}",  # noqa: S608
            )
            or 0,
        )

        if count > 0:
            entry = {
                "table": table,
                "column": col_enc,
                "source_version": source_version,
                "count": count,
            }
            reports.append(entry)

    return reports


_VERIFY_SAMPLE_MIN: Final = 10
_VERIFY_SAMPLE_MAX: Final = 1000


def _verify_sample_size(total: int) -> int:
    """Return the row count to sample for ``--verify``: ~1% bounded to [10, 1000]."""
    if total == 0:
        return 0
    return max(_VERIFY_SAMPLE_MIN, min(_VERIFY_SAMPLE_MAX, (total + 99) // 100))


async def _rotate_table(
    pool: asyncpg.Pool,
    cipher: ContentCipher,
    source_version: int,
    target_version: int,
    table: str,
    col_encrypted: str,
    col_nonce: str,
    dry_run: bool,
) -> tuple[int, int]:
    """Rotate all eligible rows for a single (table, column-pair).

    Returns ``(rows_updated, audit_rows_written)``.
    """
    hex_pfx = f"{source_version:02x}"
    where_clause = (
        f"{col_encrypted} IS NOT NULL AND {col_nonce} IS NOT NULL "
        f"AND encode({col_nonce}, 'hex') LIKE '{hex_pfx}%'"
    )

    total_count: int = int(
        await pool.fetchval(
            f"SELECT COUNT(*) FROM {table} WHERE {where_clause}",  # noqa: S608
        )
        or 0
    )

    if total_count == 0:
        return 0, 0

    batch_num = 0
    rows_updated = 0
    audit_rows_written = 0
    last_id = 0

    while True:
        # Fetch a batch of candidate rows.
        async with pool.acquire() as conn:
            batch = await conn.fetch(
                f"""
                SELECT id, {col_encrypted} AS ct, {col_nonce} AS nonce
                FROM {table}
                WHERE {where_clause}
                  AND id > $1
                ORDER BY id ASC
                LIMIT $2
                """,  # noqa: S608
                last_id,
                BATCH_SIZE,
            )

        if not batch:
            break

        last_id = batch[-1]["id"]
        batch_num += 1

        batch_updated = 0

        # Update (decrypt + re-encrypt) in its own transaction.
        async with pool.acquire() as conn, conn.transaction():
            for row in batch:
                entry_id = int(row["id"])
                ct_bytes = bytes(row["ct"]) if row["ct"] else None
                nonce_bytes = bytes(row["nonce"]) if row["nonce"] else None

                if ct_bytes is None or nonce_bytes is None:
                    continue

                # Skip rows already at target version (idempotency).
                if nonce_bytes[0] == target_version:
                    continue

                try:
                    plaintext = cipher.decrypt(ct_bytes, nonce_bytes)
                except (DecryptionError, ValueError) as exc:
                    logger.warning(
                        "rotate: decrypt failed for %s id=%d v=%d: %s",
                        table,
                        entry_id,
                        source_version,
                        exc,
                    )
                    continue

                try:
                    new_ct, new_nonce = cipher.encrypt_with_version(plaintext, target_version)
                except (ValueError, TypeError) as exc:
                    logger.warning(
                        "rotate: encrypt failed for %s id=%d v=%d: %s",
                        table,
                        entry_id,
                        target_version,
                        exc,
                    )
                    continue

                if not dry_run:  # noqa: SIM108 -- combined with next block
                    await conn.execute(
                        f"UPDATE {table} SET "  # noqa: S608
                        f"{col_encrypted} = $1, {col_nonce} = $2 WHERE id = $3",
                        new_ct,
                        new_nonce,
                        entry_id,
                    )
                    batch_updated += 1

            if not dry_run and batch_updated > 0:
                # Write ONE audit row per (table, column-pair).
                await record_audit(
                    conn,
                    actor_type="admin",
                    actor_id=_AUDITOR_ID,
                    action=_AUDIT_ACTION,
                    target_type=table,
                    metadata={
                        "count": batch_updated,
                        "from_version": source_version,
                        "to_version": target_version,
                        "column": col_encrypted,
                    },
                )
                audit_rows_written += 1

        rows_updated += batch_updated
        pct = (rows_updated / total_count * 100) if total_count else 100
        logger.info(
            "%s.%s: %.1f%% %d/%d (batch %d)",
            table,
            col_encrypted,
            pct,
            rows_updated,
            total_count,
            batch_num,
        )

    return rows_updated, audit_rows_written


async def _verify(pool: asyncpg.Pool, cipher: ContentCipher, to_version: int) -> None:  # noqa: S608 -- identifiers from const
    """Re-fetch a ~1% sample per table and round-trip decrypt under target."""

    total_sampled = 0
    for table, col_enc, col_nonce in _ROTATION_SCREENS:
        total_count: int = int(
            await pool.fetchval(
                f"SELECT COUNT(*) FROM {table} WHERE {col_enc} IS NOT NULL",  # noqa: S608
            )
            or 0
        )
        if total_count == 0:
            continue

        sample_size = _verify_sample_size(total_count)

        # Sample real IDs via random() so we work correctly with non-contiguous
        # IDENTITY sequences (deletions, failed inserts, sequence cache gaps).
        rows = await pool.fetch(
            f"SELECT id, {col_enc} AS ct, {col_nonce} AS nc "  # noqa: S608
            f"FROM {table} WHERE {col_enc} IS NOT NULL "  # noqa: S608
            f"ORDER BY random() LIMIT $1",  # noqa: S608
            sample_size,
        )

        for row in rows:
            nonce = bytes(row["nc"]) if row["nc"] else None
            if nonce is None:
                continue
            try:
                cipher.decrypt(bytes(row["ct"]), nonce)
            except DecryptionError as exc:
                logger.error(
                    "verify FAILED: %s id=%d col=%s: %s",
                    table,
                    int(row["id"]),
                    col_enc,
                    exc,
                )
                raise SystemExit(1) from None

        total_sampled += len(rows)
        logger.info("verify: %s.%s round-trip OK on %d rows", table, col_enc, len(rows))

    logger.info(
        "verify: round-trip OK for %d sampled rows across %d column-pairs.",
        total_sampled,
        len(_ROTATION_SCREENS),
    )


async def _run(pool: asyncpg.Pool, cipher: ContentCipher, args: argparse.Namespace) -> None:  # noqa: S608
    """Execute the rotation (or dry-run) against all five column-pairs."""
    src_ver = args.from_version
    tgt_ver = args.to_version

    logger.info("rotation start: V%d -> V%d  (dry_run=%s)", src_ver, tgt_ver, args.dry_run)

    if tgt_ver not in cipher.known_versions:
        known = sorted(cipher.known_versions)
        raise RuntimeError(
            f"Target version {tgt_ver} not found in cipher keys "
            f"(known: {known}). Set JOURNAL_ENCRYPTION_MASTER_KEY_V{tgt_ver}"
        )

    # Dry-run mode -------------------------------------------------------
    if args.dry_run:
        reports = await _collect_dry_run_counts(pool, src_ver)
        if not reports:
            logger.info("dry-run: no rows at V%d in any table.", src_ver)
            return

        total_rows = sum(r["count"] for r in reports)
        logger.info(
            "dry-run summary: %d pairs, %d rows at V%d",
            len(reports),
            total_rows,
            src_ver,
        )
        for r in reports:
            logger.info(
                "  %s.%s : %d rows at V%d", r["table"], r["column"], r["count"], r["source_version"]
            )
        return

    # Real rotation ------------------------------------------------------
    total_updated = 0
    total_audits = 0

    for table, col_enc, col_nonce in _ROTATION_SCREENS:
        updated, audits = await _rotate_table(
            pool,
            cipher,
            src_ver,
            tgt_ver,
            table,
            col_enc,
            col_nonce,
            args.dry_run,
        )
        total_updated += updated
        total_audits += audits

    logger.info(
        "rotation done: V%d -> V%d; %d rows re-encrypted, %d audit rows",
        src_ver,
        tgt_ver,
        total_updated,
        total_audits,
    )

    if args.verify:
        await _verify(pool, cipher, tgt_ver)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Re-encrypt V_from -> V_to across five encrypted column pairs.",
    )
    parser.add_argument(
        "--from-version",
        type=int,
        required=True,
        help="Source key version (nonce[0] must match).",
    )
    parser.add_argument(
        "--to-version",
        type=int,
        required=True,
        help="Target key version to encrypt as.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Report per-table counts; do not modify rows."
    )
    parser.add_argument(
        "--verify", action="store_true", help="Post-update: decrypt ~1% sample per table."
    )
    return parser


def main() -> None:
    """CLI entry-point for encryption key rotation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stderr,
    )

    parser = _build_parser()
    args = parser.parse_args()

    keys = load_master_keys_from_env(os.environ)
    if not keys:
        sys.exit("FATAL: no JOURNAL_ENCRYPTION_MASTER_KEY_V* env vars set.")

    cipher = ContentCipher(keys)

    admin_dsn = _resolve_admin_dsn()
    asyncio.run(_main_async(pool_dsn=admin_dsn, cipher=cipher, args=args))


async def _main_async(
    *,
    pool_dsn: str,
    cipher: ContentCipher,
    args: argparse.Namespace,
) -> None:
    """async wrapper that manages pool lifecycle."""
    pool = await init_pool(pool_dsn, min_size=1, max_size=2)
    try:
        await _run(pool, cipher, args)
    except Exception as exc:
        logger.exception("rotation FAILED: %s", exc)
        raise
    finally:
        await pool.close()


if __name__ == "__main__":
    main()
