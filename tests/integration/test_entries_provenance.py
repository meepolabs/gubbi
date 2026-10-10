"""Integration tests: entry provenance columns written by ``entries.append`` and
``entries.link_to_conversation`` under RLS."""

from __future__ import annotations

import asyncio
from datetime import date
from typing import TYPE_CHECKING

import pytest
from gubbi_common.db.user_scoped import user_scoped_connection

from gubbi.storage.exceptions import ConversationNotFoundError, EntryNotFoundError
from gubbi.storage.repositories import entries as entry_repo

if TYPE_CHECKING:
    from uuid import UUID

    import asyncpg

    from gubbi.crypto.cipher import ContentCipher
    from tests.fixtures.tenants import TenantSeed

pytestmark = [pytest.mark.asyncio(loop_scope="session"), pytest.mark.integration]

_COUNT_ENTRIES_SQL = "SELECT count(*) FROM entries WHERE user_id = $1"
_LINK_SQL = "SELECT source_conversation_id FROM entries WHERE id = $1"

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


async def _act_as(conn: asyncpg.Connection, user_id: UUID) -> None:
    await conn.execute("SELECT set_config('app.current_user_id', $1, true)", str(user_id))


async def _second_conversation(admin_pool: asyncpg.Pool, seed: TenantSeed, slug: str) -> int:
    """Clone the seed's conversation under a new slug for the same owner."""
    async with admin_pool.acquire() as conn:
        conversation_id = await conn.fetchval(
            "INSERT INTO conversations"
            " (topic_id, user_id, title_encrypted, title_nonce, slug,"
            "  summary_encrypted, summary_nonce)"
            " SELECT topic_id, user_id, title_encrypted, title_nonce, $2,"
            "  summary_encrypted, summary_nonce FROM conversations WHERE id = $1"
            " RETURNING id",
            seed.conversation_id,
            slug,
        )
    return int(conversation_id)


@pytest.mark.parametrize(
    ("reference", "error"),
    [
        pytest.param("duplicate_of_entry_id", EntryNotFoundError, id="other-users-entry"),
        pytest.param(
            "source_conversation_id", ConversationNotFoundError, id="other-users-conversation"
        ),
    ],
)
async def test_append_refuses_a_reference_to_another_users_row(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    seeded_a: TenantSeed,
    seeded_b: TenantSeed,
    reference: str,
    error: type[LookupError],
) -> None:
    foreign_ids = {
        "duplicate_of_entry_id": seeded_b.entry_ids[0],
        "source_conversation_id": seeded_b.conversation_id,
    }
    async with admin_pool.acquire() as conn:
        before = await conn.fetchval(_COUNT_ENTRIES_SQL, seeded_a.user_id)

    async with user_scoped_connection(app_pool, user_id=seeded_a.user_id) as conn:
        with pytest.raises(error):
            await entry_repo.append(
                conn,
                cipher,
                topic=seeded_a.topic_path,
                content="Points across owners",
                **{reference: foreign_ids[reference]},
            )

    async with admin_pool.acquire() as conn:
        after = await conn.fetchval(_COUNT_ENTRIES_SQL, seeded_a.user_id)
    assert after == before


@pytest.mark.parametrize(
    ("reference", "error"),
    [
        pytest.param("duplicate_of_entry_id", EntryNotFoundError, id="other-users-entry"),
        pytest.param(
            "source_conversation_id", ConversationNotFoundError, id="other-users-conversation"
        ),
    ],
)
async def test_append_refuses_another_users_row_even_without_rls(
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    seeded_a: TenantSeed,
    seeded_b: TenantSeed,
    reference: str,
    error: type[LookupError],
) -> None:
    """The ownership check holds on a connection that bypasses RLS and sees every row."""
    foreign_ids = {
        "duplicate_of_entry_id": seeded_b.entry_ids[0],
        "source_conversation_id": seeded_b.conversation_id,
    }
    async with admin_pool.acquire() as conn:
        before = await conn.fetchval(_COUNT_ENTRIES_SQL, seeded_a.user_id)
        async with conn.transaction():
            await _act_as(conn, seeded_a.user_id)
            with pytest.raises(error):
                await entry_repo.append(
                    conn,
                    cipher,
                    topic=seeded_a.topic_path,
                    content="Points across owners",
                    **{reference: foreign_ids[reference]},
                )
        after = await conn.fetchval(_COUNT_ENTRIES_SQL, seeded_a.user_id)

    assert after == before


async def test_append_accepts_a_soft_deleted_entry_of_the_same_user_as_duplicate_target(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    seeded_a: TenantSeed,
) -> None:
    deleted_id = seeded_a.entry_ids[0]
    async with admin_pool.acquire() as conn:
        await conn.execute("UPDATE entries SET deleted_at = now() WHERE id = $1", deleted_id)

    async with user_scoped_connection(app_pool, user_id=seeded_a.user_id) as conn:
        entry_id = await entry_repo.append(
            conn,
            cipher,
            topic=seeded_a.topic_path,
            content="Near match of a deleted entry",
            duplicate_of_entry_id=deleted_id,
        )
        stored = await conn.fetchval(
            "SELECT duplicate_of_entry_id FROM entries WHERE id = $1", entry_id
        )

    assert stored == deleted_id


async def test_link_to_conversation_refuses_another_users_conversation(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    seeded_a: TenantSeed,
    seeded_b: TenantSeed,
) -> None:
    assert seeded_b.conversation_id is not None
    entry_id = seeded_a.entry_ids[0]

    async with user_scoped_connection(app_pool, user_id=seeded_a.user_id) as conn:
        linked = await entry_repo.link_to_conversation(conn, entry_id, seeded_b.conversation_id)
    async with admin_pool.acquire() as conn:
        stored = await conn.fetchval(_LINK_SQL, entry_id)

    assert linked is False
    assert stored is None


async def test_link_to_conversation_refuses_another_users_conversation_even_without_rls(
    admin_pool: asyncpg.Pool,
    seeded_a: TenantSeed,
    seeded_b: TenantSeed,
) -> None:
    """The ownership check holds on a connection that bypasses RLS and sees every row."""
    assert seeded_b.conversation_id is not None
    entry_id = seeded_a.entry_ids[0]

    async with admin_pool.acquire() as conn:
        linked = await entry_repo.link_to_conversation(conn, entry_id, seeded_b.conversation_id)
        stored = await conn.fetchval(_LINK_SQL, entry_id)

    assert linked is False
    assert stored is None


async def test_concurrent_links_to_one_entry_leave_exactly_the_winner(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    seeded_a: TenantSeed,
) -> None:
    assert seeded_a.conversation_id is not None
    entry_id = seeded_a.entry_ids[0]
    conversation_ids = (
        seeded_a.conversation_id,
        await _second_conversation(admin_pool, seeded_a, "racing-conv"),
    )
    both_ready = asyncio.Barrier(2)

    async def link(conversation_id: int) -> bool:
        async with user_scoped_connection(app_pool, user_id=seeded_a.user_id) as conn:
            await both_ready.wait()
            return await entry_repo.link_to_conversation(conn, entry_id, conversation_id)

    results = await asyncio.gather(*(link(cid) for cid in conversation_ids))

    async with admin_pool.acquire() as conn:
        stored = await conn.fetchval(_LINK_SQL, entry_id)
    assert sorted(results) == [False, True]
    assert stored == conversation_ids[results.index(True)]
