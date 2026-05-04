"""Integration tests for ``gubbi.scripts.rotate_encryption_key`` against the test DB.

Seeds rows encrypted at V1 into all five column-pairs, invokes the rotation
logic at the function level (not as subprocess), then asserts post-state:
every rotated row has nonce-Byte 0 == V2 and round-trips cleanly under a
_V2-only cipher_ (ensuring V1 plaintext never leaks after rotation).

Docker-backed: skipped automatically when Postgres is not reachable.
"""

from __future__ import annotations

import argparse
from datetime import UTC, date, datetime, timedelta
from uuid import UUID

import asyncpg
import pytest

from gubbi.core.crypto import ContentCipher
from gubbi.scripts.rotate_encryption_key import (
    _ROTATION_SCREENS,
    _run,
)

# Reuse the same deterministic V1 key used by tests/fixtures/tenants.py and
# tests/conftest.py so seeded rows decrypt cleanly.
_V1_KEY = bytes([1]) * 32
_V2_KEY = bytes([2]) * 32


@pytest.fixture
def dual_cipher() -> ContentCipher:
    """Cipher carrying both V1 (seed) and V2 (target) keys."""
    return ContentCipher({1: _V1_KEY, 2: _V2_KEY})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _seed_v1_rows(
    admin_pool: asyncpg.Pool,
    user_id: UUID,
) -> list[int]:
    """Insert ~10 encrypted rows per table keyed with V1.

    Mirrors ``tests/fixtures/tenants.py:seed_for`` but only writes encrypted
    columns (no embedding / search-vector logic needed).  Uses the same
    deterministic cipher key so rows round-trip under our dual_cipher.
    """
    now = datetime.now(UTC)
    today = date.today()

    # -- entries.content + reasoning pairs ----------------------------------
    async with admin_pool.acquire() as conn, conn.transaction():
        topic_id = await conn.fetchval(
            """
            INSERT INTO topics (path, title, description, user_id, created_at, updated_at)
            VALUES ('rotate-test', 'Rotation test', '', $1, $2, $2)
            RETURNING id
            """,  # noqa: S608
            user_id,
            now,
        )

        # entries: 10 rows with content + reasoning encrypted at V1
        entry_ids = []
        for i in range(10):
            plain_content = f"Entry {i} content"
            plain_reasoning = f"Entry {i} reasoning"
            ct_c, nc_c = ContentCipher({1: _V1_KEY}).encrypt_with_version(plain_content, version=1)
            ct_r, nc_r = ContentCipher({1: _V1_KEY}).encrypt_with_version(
                plain_reasoning, version=1
            )

            eid = await conn.fetchval(
                """
                INSERT INTO entries
                    (topic_id, user_id, date,
                     content_encrypted, content_nonce,
                     reasoning_encrypted, reasoning_nonce,
                     search_vector, tags, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, to_tsvector('english', $8), $9, $10, $10)
                RETURNING id
                """,  # noqa: S608
                topic_id,
                user_id,
                today - timedelta(days=i),
                ct_c,
                nc_c,
                ct_r,
                nc_r,
                plain_content,
                ["rotate-test"],
                now,
            )
            entry_ids.append(int(eid))

        # -- messages: 10 rows -----------------------------------------------
        conv_title_ct, conv_title_nc = ContentCipher({1: _V1_KEY}).encrypt_with_version(
            "Conv title", version=1
        )
        conv_summary_ct, conv_summary_nc = ContentCipher({1: _V1_KEY}).encrypt_with_version(
            "Summary", version=1
        )

        conv_id = await conn.fetchval(
            """
            INSERT INTO conversations
                (topic_id, user_id, title_encrypted, title_nonce, slug, source,
                 summary_encrypted, summary_nonce, tags,
                 participants, message_count, created_at, updated_at, json_path,
                 search_vector)
            VALUES ($1, $2, $3, $4, $5, 'rotate-test', $6, $7, $8,
                    $9, 0, $10, $10, '/tmp/rot.json', to_tsvector('english',''))
            RETURNING id
            """,  # noqa: S608
            topic_id,
            user_id,
            conv_title_ct,
            conv_title_nc,
            f"conv-rotate-{user_id.hex[:8]}",
            conv_summary_ct,
            conv_summary_nc,
            ["rotate-test"],
            now,
        )

        # Insert 10 messages.
        for i in range(10):
            ct, nc = ContentCipher({1: _V1_KEY}).encrypt_with_version(f"Msg {i}", version=1)
            await conn.execute(
                """
                INSERT INTO messages
                    (conversation_id, user_id, role,
                     content_encrypted, content_nonce, search_vector, position)
                VALUES ($1, $2, $3, $4, $5, to_tsvector('english', $6), $7)
                """,  # noqa: S608
                conv_id,
                user_id,
                "user",
                ct,
                nc,
                f"Msg {i}",
                i,
            )

    return entry_ids


def nonce_bytes_v1() -> bytes:
    """Return a 12-byte nonce with version byte = 1."""
    return bytes([1]) + b"\x00" * 11


async def _count_rows_at_version(
    pool: asyncpg.Pool,
    table: str,
    col_encrypted: str,
    col_nonce: str,
    version: int,
) -> int:
    """Count rows in *table* whose ``col_nonce`` first byte matches *version*.

    Table and column names come from module-level constants in
    ``_ROTATION_SCREENS``, not user input — safe for parameterized queries.
    """
    hex_prefix = f"{version:02x}"
    return int(  # noqa: S608 -- identifiers from constant tuple
        await pool.fetchval(
            f"""
            SELECT COUNT(*) FROM {table}
            WHERE {col_encrypted} IS NOT NULL
              AND {col_nonce} IS NOT NULL
              AND encode({col_nonce}, 'hex') LIKE '{hex_prefix}%'
            """,  # noqa: S608
        )
        or 0,
    )


async def _verify_all_at_version(
    pool: asyncpg.Pool,
    cipher_v2: ContentCipher,
    target_version: int,
) -> None:
    """Decrypt every encrypted row under a V2-only cipher to confirm migration."""
    for table, col_enc, col_nonce in _ROTATION_SCREENS:
        rows = await pool.fetch(  # noqa: S608 -- identifiers from constant
            f"SELECT id, {col_enc} AS ct, {col_nonce} AS nc "  # noqa: S608
            f"FROM {table} WHERE {col_enc} IS NOT NULL",  # noqa: S608
        )
        for row in rows:
            nonce = bytes(row["nc"])
            assert (
                nonce[0] == target_version
            ), f"{table}.{col_nonce}[0]={nonce[0]} expected {target_version}"
            # Round-trip decrypt must succeed under V2-only cipher.
            cipher_v2.decrypt(bytes(row[col_enc]), nonce)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.asyncio(loop_scope="session")


async def test_rotate_integration_full_flow(
    admin_pool: asyncpg.Pool,
    tenant_a: UUID,
    dual_cipher: ContentCipher,
) -> None:
    """Seed V1 rows into all five column-pairs, rotate to V2 with args.from=1/to=2, assert post-state.

    Verifies that every row encrypted at V1 has nonce[0]==V2 after rotation
    and round-trips cleanly under a cipher carrying ONLY the target key (V2).
    """
    _ = await _seed_v1_rows(admin_pool, tenant_a)

    # Pre-flight: confirm all seeded rows are at V1.
    for table, col_enc, col_nonce in _ROTATION_SCREENS:
        v1_count = await _count_rows_at_version(admin_pool, table, col_enc, col_nonce, 1)
        assert v1_count > 0, f"{table}.{col_enc} has no V1 rows"

    # Rotate.
    args = argparse.Namespace(from_version=1, to_version=2, dry_run=False, verify=False)
    await _run(admin_pool, dual_cipher, args)

    # Post-flight: every row must be at V2 (V1 count should be 0).
    for table, col_enc, col_nonce in _ROTATION_SCREENS:
        v1_count_after = await _count_rows_at_version(admin_pool, table, col_enc, col_nonce, 1)
        assert v1_count_after == 0, f"{table}.{col_enc} still has {v1_count_after} V1 rows"

    # All non-NULL rows should now have nonce[0] == 2 AND round-trip.
    cipher_v2_only = ContentCipher({2: _V2_KEY})
    await _verify_all_at_version(admin_pool, cipher_v2_only, target_version=2)


async def test_rotate_dry_run_writes_nothing(
    admin_pool: asyncpg.Pool,
    tenant_a: UUID,
    dual_cipher: ContentCipher,
) -> None:
    """Dry-run should report counts but leave nonce[0] unchanged."""
    await _seed_v1_rows(admin_pool, tenant_a)

    # Count V1 rows pre-dry-run.
    v1_pre = 0
    for table, col_enc, col_nonce in _ROTATION_SCREENS:
        v1_pre += await _count_rows_at_version(admin_pool, table, col_enc, col_nonce, 1)
    assert v1_pre > 0

    args = argparse.Namespace(from_version=1, to_version=2, dry_run=True, verify=False)
    await _run(admin_pool, dual_cipher, args)

    # Post-dry-run: V1 counts unchanged.
    for table, col_enc, col_nonce in _ROTATION_SCREENS:
        v1_post = await _count_rows_at_version(admin_pool, table, col_enc, col_nonce, 1)
        assert v1_post == v1_pre


async def test_decrypt_with_v1_only_cipher_after_rotation_raises(
    admin_pool: asyncpg.Pool,
    tenant_a: UUID,
    dual_cipher: ContentCipher,
) -> None:
    """After rotation to V2, decrypting with a stale V1-only cipher must fail.

    We read both ciphertext AND current nonce (now at V2), then attempt
    decryption via a cipher that carries only the V1 key -- this must
    raise because version 2 is not in known_versions.
    """
    await _seed_v1_rows(admin_pool, tenant_a)

    # Rotate to V2.
    args = argparse.Namespace(from_version=1, to_version=2, dry_run=False, verify=False)
    await _run(admin_pool, dual_cipher, args)

    # A V1-only cipher cannot decrypt rows that carry nonce[0]==V2.
    cipher_v1_only = ContentCipher({1: _V1_KEY})
    for table, col_enc, col_nonce in _ROTATION_SCREENS:
        rows = await admin_pool.fetch(  # noqa: S608 -- identifiers from constant
            f"SELECT {col_enc} AS ct, {col_nonce} AS nc "  # noqa: S608
            f"FROM {table} WHERE {col_enc} IS NOT NULL LIMIT 1",  # noqa: S608
        )
        for row in rows:
            # Decrypting V2-encrypted ciphertext with V1-only cipher raises
            # ValueError (unknown key version) because nonce[0] == 2.
            with pytest.raises(ValueError, match="unknown key version"):
                cipher_v1_only.decrypt(bytes(row["ct"]), bytes(row["nc"]))


async def test_rotate_skips_rows_already_at_target_version(
    admin_pool: asyncpg.Pool,
    tenant_a: UUID,
    dual_cipher: ContentCipher,
) -> None:
    """Rows already at --to-version are not re-encrypted (idempotent skip).

    Seeds entries.content rows directly at V2 (via the dual cipher's
    encrypt_with_version); after a V1 -> V2 rotation pass, those V2 rows
    must carry the SAME ciphertext+nonce they started with -- proof that
    the rotation loop did not touch them.
    """
    # Seed mixed V1 + V2 rows.  V1 rows are seeded by the helper; we then
    # add a handful of V2-from-the-start rows whose ciphertext we capture
    # so we can detect any unwanted re-encryption.
    await _seed_v1_rows(admin_pool, tenant_a)

    pristine_v2: dict[int, tuple[bytes, bytes]] = {}
    async with admin_pool.acquire() as conn, conn.transaction():
        topic_id = await conn.fetchval(
            "SELECT id FROM topics WHERE user_id = $1 LIMIT 1",
            tenant_a,
        )
        for i in range(3):
            ct, nc = dual_cipher.encrypt_with_version(f"v2-from-start-{i}", version=2)
            eid = await conn.fetchval(
                """
                INSERT INTO entries
                    (topic_id, user_id, date,
                     content_encrypted, content_nonce,
                     search_vector, tags, created_at, updated_at)
                VALUES ($1, $2, CURRENT_DATE, $3, $4,
                        to_tsvector('english', 'v2-marker'), '{}',
                        now(), now())
                RETURNING id
                """,  # noqa: S608
                topic_id,
                tenant_a,
                ct,
                nc,
            )
            pristine_v2[int(eid)] = (ct, nc)

    args = argparse.Namespace(from_version=1, to_version=2, dry_run=False, verify=False)
    await _run(admin_pool, dual_cipher, args)

    # Each V2-from-start row must carry the EXACT ciphertext+nonce we wrote
    # in.  Re-encrypting under V2 would generate a fresh nonce (CSPRNG bytes
    # 1..11) and therefore a different ciphertext, so byte equality is a
    # strong tamper signal.
    for eid, (orig_ct, orig_nc) in pristine_v2.items():
        row = await admin_pool.fetchrow(
            "SELECT content_encrypted AS ct, content_nonce AS nc FROM entries WHERE id = $1",
            eid,
        )
        assert row is not None
        assert bytes(row["ct"]) == orig_ct, f"entries.id={eid} ciphertext changed"
        assert bytes(row["nc"]) == orig_nc, f"entries.id={eid} nonce changed"


async def test_verify_flag_validates_real_sample_with_id_gaps(
    admin_pool: asyncpg.Pool,
    tenant_a: UUID,
    dual_cipher: ContentCipher,
) -> None:
    """``--verify`` must round-trip real sampled rows even when IDs are non-contiguous.

    Regression test for the prior implementation that constructed sample IDs
    via ``range(1, total+1)``: with deletions creating gaps, ``WHERE id IN
    (...)`` would match few or zero rows and verify would silently pass.
    Here we delete every other entry to manufacture gaps, rotate, and then
    invoke ``_run`` with ``verify=True`` -- the call must complete without
    raising SystemExit.
    """
    entry_ids = await _seed_v1_rows(admin_pool, tenant_a)

    # Manufacture ID gaps by deleting alternate rows.  This makes the
    # contiguous-PK assumption strictly false.
    async with admin_pool.acquire() as conn, conn.transaction():
        for eid in entry_ids[::2]:
            await conn.execute("DELETE FROM entries WHERE id = $1", eid)

    args = argparse.Namespace(from_version=1, to_version=2, dry_run=False, verify=True)
    # Must not raise SystemExit; verify path must sample real, surviving rows
    # and round-trip them under the dual cipher.
    await _run(admin_pool, dual_cipher, args)

    # Final state: zero V1 rows remain.
    for table, col_enc, col_nonce in _ROTATION_SCREENS:
        v1_after = await _count_rows_at_version(admin_pool, table, col_enc, col_nonce, 1)
        assert v1_after == 0, f"{table}.{col_enc} still has {v1_after} V1 rows after verify run"
