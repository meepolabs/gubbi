"""Integration tests: entry provenance columns written by ``entries.append`` and
``entries.link_to_conversation`` under RLS."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import pytest
from gubbi_common.db.user_scoped import user_scoped_connection

from gubbi.storage.repositories import entries as entry_repo

if TYPE_CHECKING:
    from uuid import UUID

    import asyncpg

    from gubbi.crypto.cipher import ContentCipher
    from tests.fixtures.tenants import TenantSeed

pytestmark = [pytest.mark.asyncio(loop_scope="session"), pytest.mark.integration]

_PROVENANCE_SQL = (
    "SELECT date, client, duplicate_of_entry_id, source_conversation_id FROM entries WHERE id = $1"
)


async def test_append_round_trips_event_date_and_provenance(
    app_pool: asyncpg.Pool,
    cipher: ContentCipher,
    seeded_a: TenantSeed,
) -> None:
    near_match_id = seeded_a.entry_ids[0]

    async with user_scoped_connection(app_pool, user_id=seeded_a.user_id) as conn:
        entry_id = await entry_repo.append(
            conn,
            cipher,
            topic=seeded_a.topic_path,
            content="Chose the blue bike",
            date="2026-03-14",
            client="chatgpt",
            duplicate_of_entry_id=near_match_id,
            source_conversation_id=seeded_a.conversation_id,
        )
        row = await conn.fetchrow(_PROVENANCE_SQL, entry_id)

    assert row is not None
    assert row["date"] == date(2026, 3, 14)
    assert row["client"] == "chatgpt"
    assert row["duplicate_of_entry_id"] == near_match_id
    assert row["source_conversation_id"] == seeded_a.conversation_id


async def test_append_without_provenance_leaves_columns_null(
    app_pool: asyncpg.Pool,
    cipher: ContentCipher,
    seeded_a: TenantSeed,
) -> None:
    async with user_scoped_connection(app_pool, user_id=seeded_a.user_id) as conn:
        entry_id = await entry_repo.append(
            conn, cipher, topic=seeded_a.topic_path, content="Plain MCP entry"
        )
        row = await conn.fetchrow(_PROVENANCE_SQL, entry_id)

    assert row is not None
    assert row["client"] is None
    assert row["duplicate_of_entry_id"] is None
    assert row["source_conversation_id"] is None


async def test_link_to_conversation_sets_an_unlinked_entry(
    app_pool: asyncpg.Pool,
    seeded_a: TenantSeed,
) -> None:
    assert seeded_a.conversation_id is not None
    entry_id = seeded_a.entry_ids[0]

    async with user_scoped_connection(app_pool, user_id=seeded_a.user_id) as conn:
        linked = await entry_repo.link_to_conversation(conn, entry_id, seeded_a.conversation_id)
        stored = await conn.fetchval(
            "SELECT source_conversation_id FROM entries WHERE id = $1", entry_id
        )

    assert linked is True
    assert stored == seeded_a.conversation_id


async def test_link_to_conversation_never_overwrites_an_existing_link(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    seeded_a: TenantSeed,
) -> None:
    assert seeded_a.conversation_id is not None
    entry_id = seeded_a.entry_ids[0]
    async with admin_pool.acquire() as conn:
        other_conversation_id = await conn.fetchval(
            "INSERT INTO conversations"
            " (topic_id, user_id, title_encrypted, title_nonce, slug,"
            "  summary_encrypted, summary_nonce)"
            " SELECT topic_id, user_id, title_encrypted, title_nonce, 'second-conv',"
            "  summary_encrypted, summary_nonce FROM conversations WHERE id = $1"
            " RETURNING id",
            seeded_a.conversation_id,
        )
        await conn.execute(
            "UPDATE entries SET source_conversation_id = $2 WHERE id = $1",
            entry_id,
            seeded_a.conversation_id,
        )

    async with user_scoped_connection(app_pool, user_id=seeded_a.user_id) as conn:
        linked = await entry_repo.link_to_conversation(conn, entry_id, other_conversation_id)
        stored = await conn.fetchval(
            "SELECT source_conversation_id FROM entries WHERE id = $1", entry_id
        )

    assert linked is False
    assert stored == seeded_a.conversation_id


async def test_link_to_conversation_cannot_reach_another_users_entry(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    seeded_a: TenantSeed,
    tenant_b: UUID,
) -> None:
    assert seeded_a.conversation_id is not None
    entry_id = seeded_a.entry_ids[0]

    async with user_scoped_connection(app_pool, user_id=tenant_b) as conn:
        linked = await entry_repo.link_to_conversation(conn, entry_id, seeded_a.conversation_id)
    async with admin_pool.acquire() as conn:
        stored = await conn.fetchval(
            "SELECT source_conversation_id FROM entries WHERE id = $1", entry_id
        )

    assert linked is False
    assert stored is None
