"""Integration test for the repo-layer DecryptionError contract (TASK-02.13).

TASK-02.11's ContentCipher distinguishes `ValueError` (unknown key version,
malformed nonce) from `cryptography.exceptions.InvalidTag` (tamper, wrong
key, truncation). Distinguishing those two in any response to the end user
would be a version-existence oracle: an attacker could learn which key
versions the server has loaded by mutating nonce[0] and observing the error.

The repo layer (TASK-02.13) flattens both into a single opaque
`DecryptionError`. This test proves the invariant end-to-end: it seeds an
encrypted entry via the admin pool (bypassing RLS), mutates the ciphertext
or nonce directly, then exercises the repo read path and asserts
`DecryptionError` -- with the ORIGINAL exception preserved on `__cause__`
for forensic logging.

Seeding goes through ``admin_pool`` (not through ``entry_repo.append``)
because the repo-layer INSERTs do not yet wire ``user_id`` through --
that is a pre-existing gap tracked against TASK-02.06 follow-up, not a
02.13 concern. Admin-pool seeding gives this test a stable baseline that
does not regress if / when the repo INSERT path is completed.

Docker-backed: skipped automatically when Postgres is not reachable.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from uuid import UUID

import asyncpg
import pytest
from cryptography.exceptions import InvalidTag

from journalctl.core.crypto import ContentCipher, DecryptionError
from journalctl.core.db_context import user_scoped_connection
from journalctl.storage.repositories import entries as entry_repo

# Session-scoped asyncpg pools (admin_pool, app_pool) require tests to
# run in the same event loop they were created in.
pytestmark = pytest.mark.asyncio(loop_scope="session")


async def _seed_encrypted_entry(
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    user_id: UUID,
    content: str = "original plaintext",
    topic_path: str = "encryption/contract",
) -> int:
    """Insert one topic + one encrypted entry owned by ``user_id``.

    Uses ``admin_pool`` (BYPASSRLS) so we can bind ``user_id`` directly --
    the repo INSERT path does not yet thread user_id through and is not
    what this test is exercising.

    ``topic_path`` MUST be unique per caller across the DB: ``topics.path``
    carries a global UNIQUE constraint (not per-user), so tests that seed
    two tenants must pass distinct paths.
    """
    now = datetime.now(UTC)
    today = date.today()
    content_ct, content_nonce = cipher.encrypt(content)

    async with admin_pool.acquire() as conn, conn.transaction():
        topic_id = await conn.fetchval(
            """
            INSERT INTO topics (path, title, description, user_id, created_at, updated_at)
            VALUES ($1, 'Encryption contract', '', $2, $3, $3)
            RETURNING id
            """,
            topic_path,
            user_id,
            now,
        )
        entry_id = await conn.fetchval(
            """
            INSERT INTO entries
                (topic_id, user_id, date, content_encrypted, content_nonce,
                 search_vector, tags, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, to_tsvector('english', $6), '{}', $7, $7)
            RETURNING id
            """,
            topic_id,
            user_id,
            today,
            content_ct,
            content_nonce,
            content,
            now,
        )
    return int(entry_id)


async def test_tampered_ciphertext_surfaces_as_decryption_error(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    tenant_a: UUID,
) -> None:
    """Flip one byte of ``content_encrypted`` and assert DecryptionError.

    Underlying ``__cause__`` must be ``InvalidTag`` (forensic signal
    preserved server-side), but the raised type is opaque.
    """
    entry_id = await _seed_encrypted_entry(admin_pool, cipher, tenant_a)

    async with admin_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT content_encrypted FROM entries WHERE id = $1",
            entry_id,
        )
        assert row is not None
        ct = bytes(row["content_encrypted"])
        tampered = bytes([ct[0] ^ 0xFF]) + ct[1:]
        await conn.execute(
            "UPDATE entries SET content_encrypted = $1 WHERE id = $2",
            tampered,
            entry_id,
        )

    with pytest.raises(DecryptionError) as exc_info:
        async with user_scoped_connection(app_pool, tenant_a) as conn:
            await entry_repo.get_text(conn, cipher, entry_id)

    assert isinstance(exc_info.value.__cause__, InvalidTag)


async def test_unknown_version_nonce_surfaces_as_decryption_error(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    tenant_a: UUID,
) -> None:
    """Mutate ``content_nonce[0]`` to an unknown version and assert DecryptionError.

    The underlying ``__cause__`` is ``ValueError`` (unknown key version)
    but the raised type is the SAME ``DecryptionError`` as the tamper
    case -- no distinguishing signal reaches the caller.
    """
    entry_id = await _seed_encrypted_entry(admin_pool, cipher, tenant_a)

    async with admin_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT content_nonce FROM entries WHERE id = $1",
            entry_id,
        )
        assert row is not None
        nonce = bytes(row["content_nonce"])
        bad_version = bytes([99]) + nonce[1:]
        await conn.execute(
            "UPDATE entries SET content_nonce = $1 WHERE id = $2",
            bad_version,
            entry_id,
        )

    with pytest.raises(DecryptionError) as exc_info:
        async with user_scoped_connection(app_pool, tenant_a) as conn:
            await entry_repo.get_text(conn, cipher, entry_id)

    assert isinstance(exc_info.value.__cause__, ValueError)


async def test_tamper_and_bad_version_raise_same_error_type(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    tenant_a: UUID,
    tenant_b: UUID,
) -> None:
    """Caller-observable type is identical across distinct failure causes.

    Two entries, two failure modes (tamper vs bad version). The public
    exception type and message must be indistinguishable; only
    ``__cause__`` differs, and that is a server-side forensic detail
    that is not returned to the client.
    """
    a_id = await _seed_encrypted_entry(
        admin_pool,
        cipher,
        tenant_a,
        content="content for A",
        topic_path="encryption/contract-a",
    )
    b_id = await _seed_encrypted_entry(
        admin_pool,
        cipher,
        tenant_b,
        content="content for B",
        topic_path="encryption/contract-b",
    )

    async with admin_pool.acquire() as conn:
        a_row = await conn.fetchrow("SELECT content_encrypted FROM entries WHERE id = $1", a_id)
        b_row = await conn.fetchrow("SELECT content_nonce FROM entries WHERE id = $1", b_id)
        assert a_row is not None
        assert b_row is not None

        a_ct = bytes(a_row["content_encrypted"])
        await conn.execute(
            "UPDATE entries SET content_encrypted = $1 WHERE id = $2",
            bytes([a_ct[0] ^ 0xFF]) + a_ct[1:],
            a_id,
        )

        b_nonce = bytes(b_row["content_nonce"])
        await conn.execute(
            "UPDATE entries SET content_nonce = $1 WHERE id = $2",
            bytes([99]) + b_nonce[1:],
            b_id,
        )

    with pytest.raises(DecryptionError) as a_exc:
        async with user_scoped_connection(app_pool, tenant_a) as conn:
            await entry_repo.get_text(conn, cipher, a_id)

    with pytest.raises(DecryptionError) as b_exc:
        async with user_scoped_connection(app_pool, tenant_b) as conn:
            await entry_repo.get_text(conn, cipher, b_id)

    assert type(a_exc.value) is type(b_exc.value)
    assert str(a_exc.value) == str(b_exc.value)
    assert isinstance(a_exc.value.__cause__, InvalidTag)
    assert isinstance(b_exc.value.__cause__, ValueError)


async def test_happy_path_round_trip_via_repo(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    tenant_a: UUID,
) -> None:
    """Sanity check: untampered data round-trips cleanly through the repo read.

    If this fails, the tampering tests above do not prove anything (because
    a broken baseline would also "catch" the mutation).
    """
    entry_id = await _seed_encrypted_entry(admin_pool, cipher, tenant_a, content="hello world")

    async with user_scoped_connection(app_pool, tenant_a) as conn:
        result = await entry_repo.get_text(conn, cipher, entry_id)

    assert result is not None
    content, reasoning = result
    assert content == "hello world"
    assert reasoning is None
