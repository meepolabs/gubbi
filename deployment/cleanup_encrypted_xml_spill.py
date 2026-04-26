"""One-shot cleanup: trim XML spill from encrypted entries.reasoning.

Usage::

    cd journalctl
    JOURNAL_ENCRYPTION_MASTER_KEY_V1="base64-key..." \\
        JOURNAL_DB_ADMIN_URL="postgresql://admin@localhost:5432/journal" \\
        poetry run python deployment/cleanup_encrypted_xml_spill.py

Add ``--dry-run`` to count without writing.

What it does:
1. Iterates entries where reasoning_encrypted IS NOT NULL in id-ordered batches.
2. Decrypts each value via ContentCipher (AES-256-GCM).
3. If the decrypted text contains '<parameter', trims at the first occurrence
   and attempts to recover tags from a <parameter name="tags"> section.
4. Re-encrypts and writes back reasoning_encrypted / reasoning_nonce (and
   optionally tags) for matching rows.
5. Inserts an audit_log row documenting each change (action='cleanup_xml_spill_v2').

Idempotent: a second run finds zero matching rows because the XML spill
pattern has been removed.

Lives in deployment/ rather than journalctl/scripts/ because this is a
one-shot tool meant to be deleted after the founder runs it on prod.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys

import asyncpg

from journalctl.audit import record_audit
from journalctl.core.crypto import ContentCipher, load_master_keys_from_env
from journalctl.storage.pg_setup import _init_connection

logger = logging.getLogger(__name__)

_XML_PARAM_RE = re.compile(r"<parameter\b")
_TAGS_RE = re.compile(r'<parameter\s+name=["\']tags["\']>(.*?)</parameter>', re.DOTALL)

_ACTION = "cleanup_xml_spill_v2"
_ACTOR_ID = "script:cleanup_encrypted_xml_spill"

_SELECT_QUERY = (
    "SELECT id, reasoning_encrypted AS cte, reasoning_nonce AS rn, tags "
    "FROM entries "
    "WHERE reasoning_encrypted IS NOT NULL AND id > $1 "
    "ORDER BY id ASC "
    "LIMIT $2"
)


def _fail(msg: str) -> None:
    sys.stderr.write(f"{msg}\n")
    sys.exit(1)


def _trim_reasoning(raw: str) -> tuple[str, list[str] | None]:
    """Trim raw reasoning at the first ``<parameter`` and recover tags.

    Returns ``(trimmed_reasoning, recovered_tags_or_None)``. If the text
    contains no ``<parameter``, returns ``(raw, None)`` (the caller skips it).
    """
    cut = raw.find("<parameter")
    if cut < 0:
        return raw, None

    trimmed = raw[:cut].rstrip()

    # Try to recover tags from the XML spill block.
    match = _TAGS_RE.search(raw[cut:])
    recovered_tags: list[str] | None = None
    if match:
        try:
            candidate = json.loads(match.group(1).strip())
            if isinstance(candidate, list):
                recovered_tags = candidate
        except (json.JSONDecodeError, ValueError):
            pass

    return trimmed, recovered_tags


async def _run(pool: asyncpg.Pool, cipher: ContentCipher, dry_run: bool) -> int:
    """Iterate encrypted rows, trim XML spill, update in place.

    Returns the number of rows that matched the spill pattern (whether or
    not they were written -- in dry-run, no write happens).

    Pagination uses id-ordered cursoring (WHERE id > last_id ORDER BY id ASC)
    to guarantee forward progress. Without this the loop could re-fetch the
    same already-cleaned rows endlessly.
    """
    batch_size = 1000
    last_id = 0
    total_matched = 0

    while True:
        async with pool.acquire() as conn:
            batch = await conn.fetch(_SELECT_QUERY, last_id, batch_size)
        if not batch:
            break
        last_id = batch[-1]["id"]

        to_update: list[tuple[int, bytes, bytes, list[str] | None]] = []
        audit_rows: list[dict] = []

        for row in batch:
            entry_id = row["id"]
            ct = row["cte"]
            nonce = row["rn"]

            try:
                decrypted = cipher.decrypt(bytes(ct), bytes(nonce))
            except Exception as exc:
                logger.error("decrypt failed for entry %d: %s", entry_id, exc)
                continue

            trimmed, recovered_tags = _trim_reasoning(decrypted)
            if trimmed == decrypted:
                # No XML spill found; skip.
                continue

            new_ct, new_nonce = cipher.encrypt(trimmed)

            to_update.append((int(entry_id), new_ct, new_nonce, recovered_tags))
            audit_rows.append(
                {
                    "entry_id": int(entry_id),
                    "len_before": len(decrypted),
                    "len_after": len(trimmed),
                    "tags_recovered": recovered_tags is not None,
                }
            )

        if to_update and not dry_run:
            async with pool.acquire() as conn:  # noqa: SIM117 -- outer must be acquired before inner transaction
                async with conn.transaction():
                    for entry_id, new_ct, new_nonce, recovered_tags in to_update:
                        if recovered_tags is not None:
                            await conn.execute(
                                "UPDATE entries SET reasoning_encrypted = $1, "
                                "reasoning_nonce = $2, tags = $3 WHERE id = $4",
                                new_ct,
                                new_nonce,
                                recovered_tags,
                                entry_id,
                            )
                        else:
                            await conn.execute(
                                "UPDATE entries SET reasoning_encrypted = $1, "
                                "reasoning_nonce = $2 WHERE id = $3",
                                new_ct,
                                new_nonce,
                                entry_id,
                            )

                    for ar in audit_rows:
                        await record_audit(
                            conn,
                            actor_type="admin",
                            actor_id=_ACTOR_ID,
                            action=_ACTION,
                            target_type="entry",
                            target_id=str(ar["entry_id"]),
                            metadata={
                                "reasoning_len_before": ar["len_before"],
                                "reasoning_len_after": ar["len_after"],
                                "tags_recovered": ar["tags_recovered"],
                            },
                        )

        total_matched += len(to_update)
        prefix = "(dry-run) " if dry_run else ""
        for ar in audit_rows:
            logger.info(
                "%scleanup_xml_spill_v2: entry %d trimmed %d -> %d tags=%s",
                prefix,
                ar["entry_id"],
                ar["len_before"],
                ar["len_after"],
                ar["tags_recovered"],
            )

    return total_matched


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
        "rows. Set it before running the script."
    )
    raise RuntimeError("unreachable")  # pragma: no cover


async def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Trim XML spill from encrypted entries.reasoning at rest."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count matching rows and report without writing.",
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

    pool: asyncpg.Pool = await asyncpg.create_pool(
        admin_dsn,
        statement_cache_size=0,
        init=_init_connection,
        min_size=1,
        max_size=2,
    )
    try:
        matched = await _run(pool, cipher, dry_run=args.dry_run)
        logger.info(
            "cleanup_xml_spill_v2 finished: matched %d entries (%s)",
            matched,
            "dry-run" if args.dry_run else "committed",
        )
    except Exception as exc:
        sys.stderr.write(f"CLEANUP FAILED: {type(exc).__name__}: {exc}\n")
        sys.exit(1)
    finally:
        await pool.close()


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
