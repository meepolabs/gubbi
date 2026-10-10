"""Integration tests: ``conversations.get_extraction_state`` and
``conversations.set_extraction_state`` under RLS."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from gubbi_common.db.user_scoped import user_scoped_connection

from gubbi.storage.repositories import conversations as conv_repo

if TYPE_CHECKING:
    from uuid import UUID

    import asyncpg

    from tests.fixtures.tenants import TenantSeed

pytestmark = [pytest.mark.asyncio(loop_scope="session"), pytest.mark.integration]

_HASH = "a" * 64


async def test_extraction_state_is_unset_before_first_extraction(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    seeded_a: TenantSeed,
) -> None:
    assert seeded_a.conversation_id is not None
    async with admin_pool.acquire() as conn:
        await conn.execute(
            "UPDATE conversations SET platform = 'chatgpt' WHERE id = $1",
            seeded_a.conversation_id,
        )
        created_at = await conn.fetchval(
            "SELECT created_at FROM conversations WHERE id = $1", seeded_a.conversation_id
        )

    async with user_scoped_connection(app_pool, user_id=seeded_a.user_id) as conn:
        state = await conv_repo.get_extraction_state(conn, seeded_a.conversation_id)

    assert state == conv_repo.ExtractionState(
        platform="chatgpt",
        source="claude",
        created_at=created_at,
        extracted_content_hash=None,
        extraction_version=None,
    )


async def test_set_extraction_state_round_trips(
    app_pool: asyncpg.Pool,
    seeded_a: TenantSeed,
) -> None:
    assert seeded_a.conversation_id is not None

    async with user_scoped_connection(app_pool, user_id=seeded_a.user_id) as conn:
        updated = await conv_repo.set_extraction_state(conn, seeded_a.conversation_id, _HASH, 3)
        state = await conv_repo.get_extraction_state(conn, seeded_a.conversation_id)

    assert updated is True
    assert state is not None
    assert state.extracted_content_hash == _HASH
    assert state.extraction_version == 3


async def test_extraction_state_is_invisible_to_another_user(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    seeded_a: TenantSeed,
    tenant_b: UUID,
) -> None:
    assert seeded_a.conversation_id is not None

    async with user_scoped_connection(app_pool, user_id=tenant_b) as conn:
        state = await conv_repo.get_extraction_state(conn, seeded_a.conversation_id)
        updated = await conv_repo.set_extraction_state(conn, seeded_a.conversation_id, _HASH, 3)
    async with admin_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT extracted_content_hash, extraction_version FROM conversations WHERE id = $1",
            seeded_a.conversation_id,
        )

    assert state is None
    assert updated is False
    assert row is not None
    assert row["extracted_content_hash"] is None
    assert row["extraction_version"] is None
