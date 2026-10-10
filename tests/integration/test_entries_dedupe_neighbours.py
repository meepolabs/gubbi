"""``entries.list_dedupe_neighbours`` against a real RLS-enforced Postgres.

Every exclusion is asserted next to a positive control: a row identical to the
excluded one except for the single property under test is returned, so an
empty result cannot pass for a working filter.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from uuid import UUID

import asyncpg
import pytest
import pytest_asyncio
from gubbi_common.db.user_scoped import user_scoped_connection

from gubbi.crypto.cipher import ContentCipher
from gubbi.storage.repositories import entries as entry_repo
from tests.fixtures.tenants import TenantSeed, seed_for

pytestmark = pytest.mark.asyncio(loop_scope="session")

DAY = date(2026, 6, 1)
OTHER_DAY = date(2026, 6, 2)
UNQUERIED_DAY = date(2026, 6, 3)
CLIENT = "claude"
EMBEDDING = [1.0] + [0.0] * 383


@dataclass(frozen=True)
class _Planted:
    seed: TenantSeed
    ids: dict[str, int]


async def _plant_entry(
    conn: asyncpg.Connection,
    cipher: ContentCipher,
    seed: TenantSeed,
    *,
    content: str,
    day: date = DAY,
    client: str | None = CLIENT,
    conversation_id: int | None = None,
    deleted: bool = False,
    embedded: bool = True,
) -> int:
    content_ct, content_nonce = cipher.encrypt(content)
    now = datetime.now(UTC)
    entry_id = int(
        await conn.fetchval(
            """
            INSERT INTO entries
                (topic_id, user_id, date, content_encrypted, content_nonce,
                 client, conversation_id, deleted_at, tags, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, '{}', $9, $9)
            RETURNING id
            """,
            seed.topic_id,
            seed.user_id,
            day,
            content_ct,
            content_nonce,
            client,
            conversation_id,
            now if deleted else None,
            now,
        )
    )
    if embedded:
        await conn.execute(
            "INSERT INTO entry_embeddings (entry_id, user_id, embedding, indexed_at)"
            " VALUES ($1, $2, $3, $4)",
            entry_id,
            seed.user_id,
            EMBEDDING,
            now,
        )
    return entry_id


async def _second_conversation(conn: asyncpg.Connection, seed: TenantSeed) -> int:
    """Clone the seeded conversation row so a non-excluded summary entry can exist."""
    return int(
        await conn.fetchval(
            """
            INSERT INTO conversations
                (topic_id, user_id, title_encrypted, title_nonce, slug, source,
                 summary_encrypted, summary_nonce, tags, participants, message_count,
                 created_at, updated_at, json_path, search_vector)
            SELECT topic_id, user_id, title_encrypted, title_nonce, slug || '-other', source,
                   summary_encrypted, summary_nonce, tags, participants, message_count,
                   created_at, updated_at, json_path || '.other', search_vector
            FROM conversations WHERE id = $1
            RETURNING id
            """,
            seed.conversation_id,
        )
    )


@pytest_asyncio.fixture
async def planted(
    admin_pool: asyncpg.Pool,
    tenant_a: UUID,
    tenant_b: UUID,
    cipher: ContentCipher,
) -> _Planted:
    seed_a = await seed_for(admin_pool, tenant_a, topic_path="dedupe-a/notes", n_entries=0)
    seed_b = await seed_for(admin_pool, tenant_b, topic_path="dedupe-b/notes", n_entries=0)
    assert seed_a.conversation_id is not None
    async with admin_pool.acquire() as conn, conn.transaction():
        other_conversation = await _second_conversation(conn, seed_a)
        ids = {
            "same_client": await _plant_entry(conn, cipher, seed_a, content="same client"),
            "null_client": await _plant_entry(conn, cipher, seed_a, content="legacy", client=None),
            "other_client": await _plant_entry(
                conn, cipher, seed_a, content="other client", client="chatgpt"
            ),
            "other_day": await _plant_entry(conn, cipher, seed_a, content="day two", day=OTHER_DAY),
            "unqueried_day": await _plant_entry(
                conn, cipher, seed_a, content="day three", day=UNQUERIED_DAY
            ),
            "deleted": await _plant_entry(conn, cipher, seed_a, content="gone", deleted=True),
            "own_summary": await _plant_entry(
                conn, cipher, seed_a, content="own summary", conversation_id=seed_a.conversation_id
            ),
            "other_summary": await _plant_entry(
                conn, cipher, seed_a, content="other summary", conversation_id=other_conversation
            ),
            "unembedded": await _plant_entry(
                conn, cipher, seed_a, content="not indexed", embedded=False
            ),
            "other_tenant": await _plant_entry(conn, cipher, seed_b, content="tenant b"),
        }
    return _Planted(seed=seed_a, ids=ids)


async def _neighbours(
    app_pool: asyncpg.Pool,
    cipher: ContentCipher,
    planted: _Planted,
    *,
    dates: list[date],
    client: str | None = CLIENT,
) -> dict[int, entry_repo.DedupeNeighbour]:
    assert planted.seed.conversation_id is not None
    async with user_scoped_connection(app_pool, user_id=planted.seed.user_id) as conn:
        rows = await entry_repo.list_dedupe_neighbours(
            conn, cipher, dates, client, planted.seed.conversation_id
        )
    return {row.id: row for row in rows}


async def test_returns_exactly_the_eligible_rows_for_the_queried_dates(
    app_pool: asyncpg.Pool, cipher: ContentCipher, planted: _Planted
) -> None:
    found = await _neighbours(app_pool, cipher, planted, dates=[DAY, OTHER_DAY])

    ids = planted.ids
    assert set(found) == {
        ids["same_client"],
        ids["null_client"],
        ids["other_day"],
        ids["other_summary"],
        ids["unembedded"],
    }


async def test_excludes_other_tenant_rows_under_rls(
    app_pool: asyncpg.Pool, cipher: ContentCipher, planted: _Planted, tenant_b: UUID
) -> None:
    found = await _neighbours(app_pool, cipher, planted, dates=[DAY])
    async with user_scoped_connection(app_pool, user_id=tenant_b) as conn:
        as_b = await entry_repo.list_dedupe_neighbours(conn, cipher, [DAY], CLIENT, 0)

    assert planted.ids["other_tenant"] not in found
    assert [row.id for row in as_b] == [planted.ids["other_tenant"]]


async def test_honours_the_date_set(
    app_pool: asyncpg.Pool, cipher: ContentCipher, planted: _Planted
) -> None:
    found = await _neighbours(app_pool, cipher, planted, dates=[OTHER_DAY])

    assert set(found) == {planted.ids["other_day"]}


@pytest.mark.parametrize(
    ("client", "included", "excluded"),
    [
        pytest.param(CLIENT, {"same_client", "null_client"}, {"other_client"}, id="claude"),
        pytest.param("chatgpt", {"other_client", "null_client"}, {"same_client"}, id="chatgpt"),
        pytest.param(None, {"null_client"}, {"same_client", "other_client"}, id="none"),
    ],
)
async def test_applies_the_client_or_null_rule(
    app_pool: asyncpg.Pool,
    cipher: ContentCipher,
    planted: _Planted,
    client: str | None,
    included: set[str],
    excluded: set[str],
) -> None:
    found = await _neighbours(app_pool, cipher, planted, dates=[DAY], client=client)

    assert {planted.ids[name] for name in included} <= set(found)
    assert not {planted.ids[name] for name in excluded} & set(found)


async def test_excludes_own_summary_but_not_another_conversations_summary(
    app_pool: asyncpg.Pool, cipher: ContentCipher, planted: _Planted
) -> None:
    found = await _neighbours(app_pool, cipher, planted, dates=[DAY])

    assert planted.ids["own_summary"] not in found
    assert planted.ids["other_summary"] in found


async def test_excludes_deleted_rows(
    app_pool: asyncpg.Pool, cipher: ContentCipher, planted: _Planted
) -> None:
    found = await _neighbours(app_pool, cipher, planted, dates=[DAY])

    assert planted.ids["deleted"] not in found
    assert planted.ids["same_client"] in found


async def test_decrypts_content_and_returns_embedding_or_none(
    app_pool: asyncpg.Pool, cipher: ContentCipher, planted: _Planted
) -> None:
    found = await _neighbours(app_pool, cipher, planted, dates=[DAY])

    embedded = found[planted.ids["same_client"]]
    unembedded = found[planted.ids["unembedded"]]
    assert (embedded.content, embedded.date, embedded.client) == ("same client", DAY, CLIENT)
    assert embedded.embedding == pytest.approx(tuple(EMBEDDING))
    assert unembedded.content == "not indexed"
    assert unembedded.embedding is None


async def test_empty_date_set_returns_nothing(
    app_pool: asyncpg.Pool, cipher: ContentCipher, planted: _Planted
) -> None:
    assert await _neighbours(app_pool, cipher, planted, dates=[]) == {}
