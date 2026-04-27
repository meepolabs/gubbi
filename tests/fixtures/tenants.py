"""Tenant fixtures for cross-tenant RLS tests (TASK-02.15).

These fixtures compose on top of ``conftest.py``'s ``app_pool`` and
``admin_pool``. They create two throwaway users (A and B), return their
UUIDs, and expose a ``seed_for`` helper that populates every tenant
table with rows owned by a specific user — using ``admin_pool`` so the
seed bypasses RLS (otherwise the WITH CHECK clause blocks cross-tenant
INSERTs, which is exactly what the isolation tests need to verify
separately).

Tests then assert against the data using ``app_pool`` +
``user_scoped_connection`` — that pool has no BYPASSRLS attribute so
the policies run.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from uuid import UUID, uuid4

import asyncpg
import pytest_asyncio

from journalctl.core.crypto import ContentCipher

# Fixed test cipher -- keep in sync with the ``cipher`` fixture in conftest
# so fixture-seeded rows can be decrypted by test code that imports that
# fixture. NEVER use this key outside tests.
_SEED_CIPHER = ContentCipher({1: bytes([1]) * 32})


def _encrypt(plaintext: str) -> tuple[bytes, bytes]:
    return _SEED_CIPHER.encrypt(plaintext)


@dataclass(frozen=True)
class TenantSeed:
    """Handle returned by ``seed_for`` so tests can reference what was written."""

    user_id: UUID
    topic_id: int
    topic_path: str
    entry_ids: tuple[int, ...]
    conversation_id: int | None
    message_ids: tuple[int, ...]
    # entry_embeddings is keyed by entry_id (PK) — one embedding per entry_id above.


async def _create_user(admin_pool: asyncpg.Pool, email: str) -> UUID:
    """Insert a row into ``users`` via the admin pool. Returns the generated UUID."""
    async with admin_pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO users (id, email, timezone, created_at, updated_at)
            VALUES (gen_random_uuid(), $1, 'UTC', now(), now())
            RETURNING id
            """,
            email,
        )
    if row is None:
        raise RuntimeError(f"Failed to insert test user with email={email}")
    return UUID(str(row["id"]))


@pytest_asyncio.fixture
async def tenant_a(clean_rls_db: asyncpg.Pool) -> UUID:
    """Throwaway user A. ``clean_rls_db`` truncates before/after so teardown is free."""
    return await _create_user(clean_rls_db, email="tenant-a@test.local")


@pytest_asyncio.fixture
async def tenant_b(clean_rls_db: asyncpg.Pool) -> UUID:
    """Throwaway user B. Shares the same TRUNCATE cycle as ``tenant_a``."""
    return await _create_user(clean_rls_db, email="tenant-b@test.local")


async def seed_for(
    admin_pool: asyncpg.Pool,
    user_id: UUID,
    *,
    topic_path: str,
    topic_title: str = "Test topic",
    n_entries: int = 3,
    include_conversation: bool = True,
    n_messages: int = 2,
) -> TenantSeed:
    """Populate every tenant table with rows owned by ``user_id``.

    Runs through ``admin_pool`` (BYPASSRLS) so the inserts ignore the
    ``tenant_isolation`` WITH CHECK clause — that's the whole point: the
    test harness needs to plant data the RLS-enforced pool will later
    refuse to reach. Everything writes in one transaction so a partial
    seed never leaves the DB inconsistent.
    """
    now = datetime.now(UTC)
    today = date.today()

    async with admin_pool.acquire() as conn, conn.transaction():
        topic_id = await conn.fetchval(
            """
            INSERT INTO topics (path, title, description, user_id, created_at, updated_at)
            VALUES ($1, $2, '', $3, $4, $4)
            RETURNING id
            """,
            topic_path,
            topic_title,
            user_id,
            now,
        )

        entry_ids: list[int] = []
        for i in range(n_entries):
            content_plain = f"Entry {i} for {topic_path}"
            reasoning_plain = f"Reasoning {i}"
            content_ct, content_nonce = _encrypt(content_plain)
            reasoning_ct, reasoning_nonce = _encrypt(reasoning_plain)
            entry_id = await conn.fetchval(
                """
                INSERT INTO entries
                    (topic_id, user_id, date,
                     content_encrypted, content_nonce,
                     reasoning_encrypted, reasoning_nonce,
                     search_vector, tags, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, to_tsvector('english', $8), $9, $10, $10)
                RETURNING id
                """,
                topic_id,
                user_id,
                today - timedelta(days=i),
                content_ct,
                content_nonce,
                reasoning_ct,
                reasoning_nonce,
                content_plain,
                [f"tag-{i}", "seed"],
                now,
            )
            entry_ids.append(int(entry_id))

        conv_id: int | None = None
        message_ids: list[int] = []
        if include_conversation:
            title_plain = f"Conversation for {topic_path}"
            summary_plain = "Seed conversation summary"
            title_ct, title_nonce = _encrypt(title_plain)
            summary_ct, summary_nonce = _encrypt(summary_plain)
            conv_id = await conn.fetchval(
                """
                INSERT INTO conversations
                    (topic_id, user_id, title_encrypted, title_nonce, slug, source,
                     summary_encrypted, summary_nonce, tags,
                     participants, message_count, created_at, updated_at, json_path,
                     search_vector)
                VALUES ($1, $2, $3, $4, $5, 'claude', $6, $7, $8, $9, $10, $11, $11, $12,
                        to_tsvector('english', $13))
                RETURNING id
                """,
                topic_id,
                user_id,
                title_ct,
                title_nonce,
                f"conv-{uuid4().hex[:8]}",
                summary_ct,
                summary_nonce,
                ["seed"],
                ["user", "assistant"],
                n_messages,
                now,
                f"conversations_json/{uuid4()}.json",
                f"{title_plain} {summary_plain}".strip(),
            )
            for i in range(n_messages):
                msg_plain = f"Message {i} content"
                msg_ct, msg_nonce = _encrypt(msg_plain)
                msg_id = await conn.fetchval(
                    """
                    INSERT INTO messages
                        (conversation_id, user_id, role,
                         content_encrypted, content_nonce, search_vector, position)
                    VALUES ($1, $2, $3, $4, $5, to_tsvector('english', $6), $7)
                    RETURNING id
                    """,
                    conv_id,
                    user_id,
                    "user" if i % 2 == 0 else "assistant",
                    msg_ct,
                    msg_nonce,
                    msg_plain,
                    i,
                )
                message_ids.append(int(msg_id))

        # Plant one embedding row per entry. entry_embeddings.entry_id IS the PK,
        # so there is no separate ``id`` column. A unit vector (first dim 1.0) keeps
        # cosine distance well-defined for HNSW ORDER BY tests — zero vectors would
        # produce NaN distances and mask the real ordering behaviour.
        unit_embedding = [1.0] + [0.0] * 383
        for entry_id in entry_ids:
            await conn.execute(
                """
                INSERT INTO entry_embeddings (entry_id, user_id, embedding, indexed_at)
                VALUES ($1, $2, $3, $4)
                """,
                entry_id,
                user_id,
                unit_embedding,
                now,
            )

    return TenantSeed(
        user_id=user_id,
        topic_id=int(topic_id),
        topic_path=topic_path,
        entry_ids=tuple(entry_ids),
        conversation_id=int(conv_id) if conv_id is not None else None,
        message_ids=tuple(message_ids),
    )


@pytest_asyncio.fixture
async def seeded_a(admin_pool: asyncpg.Pool, tenant_a: UUID) -> TenantSeed:
    """Pre-seeded rows for tenant A at topic ``tenant-a/notes``."""
    return await seed_for(admin_pool, tenant_a, topic_path="tenant-a/notes")


@pytest_asyncio.fixture
async def seeded_b(admin_pool: asyncpg.Pool, tenant_b: UUID) -> TenantSeed:
    """Pre-seeded rows for tenant B at topic ``tenant-b/notes``."""
    return await seed_for(admin_pool, tenant_b, topic_path="tenant-b/notes")
