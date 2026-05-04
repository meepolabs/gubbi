"""Integration tests for entries encryption round-trip (TASK-02.16).

Covers the non-contract acceptance bullets of TASK-02.16 (the tamper +
unknown-version DecryptionError contract lives in
``test_encryption_contract.py``):

1. INSERT via the encrypted path -> raw SELECT confirms ``content_encrypted``
   is NOT the plaintext (ciphertext is opaque bytes, not the readable input).
2. ``entry_repo.read`` returns decrypted content equal to the original
   plaintext (round-trip via the repo layer).
3. ``entry_repo.update`` re-encrypts: new ciphertext differs from old,
   new nonce differs from old, and the repo read returns only updated plaintext.
4. tsvector FTS still matches encrypted entries via populated ``search_vector``.

All tests seed via ``admin_pool`` (BYPASSRLS + explicit ``user_id``)
because the repo-layer INSERT does not yet thread ``user_id`` through
-- a pre-existing 02.06 follow-up, NOT a 02.13/02.16 concern. The
read/update/FTS paths all run via the RLS-active ``app_pool`` +
``user_scoped_connection`` so the repo-layer code under test is the
production code path.

Docker-backed: skipped automatically when Postgres is not reachable.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import asyncpg
import pytest
from gubbi_common.db.user_scoped import user_scoped_connection

from gubbi.core.crypto import ContentCipher
from gubbi.storage.repositories import entries as entry_repo

# Session-scoped asyncpg pools require tests to run in the pools' event loop.
pytestmark = pytest.mark.asyncio(loop_scope="session")


async def _seed_topic(admin_pool: asyncpg.Pool, user_id: UUID, topic_path: str) -> int:
    """Insert a topic owned by ``user_id``. Returns ``topic_id``."""
    now = datetime.now(UTC)
    async with admin_pool.acquire() as conn:
        topic_id = await conn.fetchval(
            """
            INSERT INTO topics (path, title, description, user_id, created_at, updated_at)
            VALUES ($1, 'Encryption round-trip', '', $2, $3, $3)
            RETURNING id
            """,
            topic_path,
            user_id,
            now,
        )
    return int(topic_id)


async def _seed_entry(
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    user_id: UUID,
    topic_id: int,
    content: str,
    reasoning: str | None = None,
) -> int:
    """Insert one encrypted entry via ``admin_pool``. Returns ``entry_id``."""
    now = datetime.now(UTC)
    today = now.date()
    content_ct, content_nonce = cipher.encrypt(content)
    reasoning_ct: bytes | None
    reasoning_nonce: bytes | None
    if reasoning is not None:
        reasoning_ct, reasoning_nonce = cipher.encrypt(reasoning)
    else:
        reasoning_ct = None
        reasoning_nonce = None

    async with admin_pool.acquire() as conn:
        entry_id = await conn.fetchval(
            """
            INSERT INTO entries
                (topic_id, user_id, date, content_encrypted, content_nonce,
                 reasoning_encrypted, reasoning_nonce,
                 search_vector, tags, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, to_tsvector('english', $8), '{}', $9, $9)
            RETURNING id
            """,
            topic_id,
            user_id,
            today,
            content_ct,
            content_nonce,
            reasoning_ct,
            reasoning_nonce,
            content,
            now,
        )
    return int(entry_id)


async def test_insert_ciphertext_is_not_plaintext(
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    tenant_a: UUID,
) -> None:
    """Raw SELECT of ``content_encrypted`` returns bytes that are NOT the plaintext.

    Also asserts the nonce is 12 bytes and starts with version byte 1
    (the test fixture cipher's only loaded version).
    """
    plaintext = "Sensitive journal content that must not appear in ciphertext"
    topic_id = await _seed_topic(admin_pool, tenant_a, "roundtrip/plaintext-check")
    entry_id = await _seed_entry(admin_pool, cipher, tenant_a, topic_id, plaintext)

    async with admin_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT content_encrypted, content_nonce FROM entries WHERE id = $1",
            entry_id,
        )

    assert row is not None
    ciphertext = bytes(row["content_encrypted"])
    nonce = bytes(row["content_nonce"])

    assert ciphertext != plaintext.encode("utf-8")
    assert plaintext.encode("utf-8") not in ciphertext
    assert len(nonce) == 12
    assert nonce[0] == cipher.active_version


async def test_repo_read_returns_original_plaintext(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    tenant_a: UUID,
) -> None:
    """``entry_repo.read`` decrypts transparently; caller sees original plaintext."""
    plaintext = "The original entry content"
    reasoning = "Why this matters"
    topic_id = await _seed_topic(admin_pool, tenant_a, "roundtrip/read-check")
    await _seed_entry(admin_pool, cipher, tenant_a, topic_id, plaintext, reasoning)

    async with user_scoped_connection(app_pool, tenant_a) as conn:
        _meta, entries, total = await entry_repo.read(conn, cipher, "roundtrip/read-check")

    assert total == 1
    assert len(entries) == 1
    assert entries[0].content == plaintext
    assert entries[0].reasoning == reasoning


async def test_update_rotates_ciphertext_and_nonce(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    tenant_a: UUID,
) -> None:
    """Update re-encrypts: new ciphertext differs, new nonce differs."""
    original = "Original plaintext"
    updated = "Updated plaintext"
    topic_id = await _seed_topic(admin_pool, tenant_a, "roundtrip/update-check")
    entry_id = await _seed_entry(admin_pool, cipher, tenant_a, topic_id, original)

    async with admin_pool.acquire() as conn:
        before = await conn.fetchrow(
            "SELECT content_encrypted, content_nonce, search_vector::text AS sv "
            "FROM entries WHERE id = $1",
            entry_id,
        )
    assert before is not None
    old_ct = bytes(before["content_encrypted"])
    old_nonce = bytes(before["content_nonce"])

    async with user_scoped_connection(app_pool, tenant_a) as conn:
        await entry_repo.update(conn, cipher, entry_id, content=updated, mode="replace")

    async with admin_pool.acquire() as conn:
        after = await conn.fetchrow(
            "SELECT content_encrypted, content_nonce, search_vector::text AS sv "
            "FROM entries WHERE id = $1",
            entry_id,
        )
    assert after is not None
    new_ct = bytes(after["content_encrypted"])
    new_nonce = bytes(after["content_nonce"])

    assert new_ct != old_ct
    assert new_nonce != old_nonce
    # Nonce uniqueness per encryption is the whole GCM invariant; restate it.
    assert new_nonce[1:] != old_nonce[1:]

    # Third encryption in the same test -- catches a buggy implementation that
    # reuses random bytes or derives them from a counter. GCM requires all
    # three random portions to be pairwise distinct.
    third_ct, third_nonce = cipher.encrypt("third distinct value")
    assert len({bytes(old_nonce[1:]), bytes(new_nonce[1:]), bytes(third_nonce[1:])}) == 3
    assert after["sv"] is not None
    assert "updat" in str(after["sv"])
    # Repo read should return the updated plaintext.
    async with user_scoped_connection(app_pool, tenant_a) as conn:
        result = await entry_repo.get_text(conn, cipher, entry_id)
    assert result is not None
    content, _reasoning = result
    assert content == updated


async def test_fts_matches_on_search_vector(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    tenant_a: UUID,
) -> None:
    """tsvector search still finds encrypted entries via search_vector."""
    plaintext = "watermelon crocodile polyhedron"
    topic_id = await _seed_topic(admin_pool, tenant_a, "roundtrip/fts-check")
    entry_id = await _seed_entry(admin_pool, cipher, tenant_a, topic_id, plaintext)

    async with user_scoped_connection(app_pool, tenant_a) as conn:
        match_row = await conn.fetchrow(
            """
            SELECT id FROM entries
            WHERE search_vector @@ websearch_to_tsquery('english', $1)
              AND deleted_at IS NULL
            """,
            "crocodile",
        )
    assert match_row is not None
    assert int(match_row["id"]) == entry_id
