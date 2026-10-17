"""``extraction_slice`` repository functions against a real RLS-enforced Postgres."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from uuid import UUID

import asyncpg
import pytest
from gubbi_common.db.user_scoped import user_scoped_connection

from gubbi.constants import EXTRACTION_TAIL_MESSAGES
from gubbi.crypto.cipher import ContentCipher, DecryptionError
from gubbi.storage.exceptions import ConversationNotFoundError
from gubbi.storage.repositories import extraction_slice
from tests.fixtures.tenants import TenantSeed, seed_for

pytestmark = [pytest.mark.asyncio(loop_scope="session"), pytest.mark.integration]

MESSAGE_COUNT = 10
FIRST_DAY = date(2026, 1, 1)


async def _thread(admin_pool: asyncpg.Pool, user_id: UUID, topic_path: str) -> TenantSeed:
    seed = await seed_for(
        admin_pool, user_id, topic_path=topic_path, n_entries=0, n_messages=MESSAGE_COUNT
    )
    assert seed.conversation_id is not None
    return seed


async def _set_watermark(
    admin_pool: asyncpg.Pool, conversation_id: int, watermark: int | None
) -> None:
    async with admin_pool.acquire() as conn:
        await conn.execute(
            "UPDATE conversations SET extracted_through_position = $2 WHERE id = $1",
            conversation_id,
            watermark,
        )


async def _watermark(admin_pool: asyncpg.Pool, conversation_id: int) -> int | None:
    async with admin_pool.acquire() as conn:
        return await conn.fetchval(  # type: ignore[no-any-return]
            "SELECT extracted_through_position FROM conversations WHERE id = $1",
            conversation_id,
        )


async def _load(
    app_pool: asyncpg.Pool, cipher: ContentCipher, seed: TenantSeed
) -> extraction_slice.ExtractionSlice:
    assert seed.conversation_id is not None
    async with user_scoped_connection(app_pool, user_id=seed.user_id) as conn:
        return await extraction_slice.load_extraction_slice(conn, cipher, seed.conversation_id)


def _positions(messages: tuple[extraction_slice.SliceMessage, ...]) -> list[int]:
    return [m.position for m in messages]


# -- load_extraction_slice ------------------------------------------------------


async def test_null_watermark_loads_every_message_with_an_empty_tail(
    app_pool: asyncpg.Pool, admin_pool: asyncpg.Pool, cipher: ContentCipher, tenant_a: UUID
) -> None:
    seed = await _thread(admin_pool, tenant_a, "slice/null")

    loaded = await _load(app_pool, cipher, seed)

    assert loaded.base is None
    assert loaded.tail == ()
    assert _positions(loaded.new) == list(range(MESSAGE_COUNT))
    assert loaded.context_count == 0
    assert loaded.target == MESSAGE_COUNT - 1


@pytest.mark.parametrize(
    ("watermark", "tail", "new"),
    [
        pytest.param(7, [2, 3, 4, 5, 6, 7], [8, 9], id="full-tail"),
        pytest.param(2, [0, 1, 2], list(range(3, MESSAGE_COUNT)), id="short-tail"),
        pytest.param(0, [0], list(range(1, MESSAGE_COUNT)), id="first-position"),
        pytest.param(MESSAGE_COUNT - 1, [4, 5, 6, 7, 8, 9], [], id="at-max"),
    ],
)
async def test_watermark_splits_tail_and_new_slices(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    tenant_a: UUID,
    watermark: int,
    tail: list[int],
    new: list[int],
) -> None:
    seed = await _thread(admin_pool, tenant_a, "slice/split")
    assert seed.conversation_id is not None
    await _set_watermark(admin_pool, seed.conversation_id, watermark)

    loaded = await _load(app_pool, cipher, seed)

    assert loaded.base == watermark
    assert _positions(loaded.tail) == tail
    assert _positions(loaded.new) == new
    assert _positions(loaded.messages) == tail + new
    assert loaded.context_count == len(tail) <= EXTRACTION_TAIL_MESSAGES
    assert loaded.target == (new[-1] if new else None)


async def test_slice_messages_are_decrypted_with_role_and_timestamp(
    app_pool: asyncpg.Pool, admin_pool: asyncpg.Pool, cipher: ContentCipher, tenant_a: UUID
) -> None:
    seed = await _thread(admin_pool, tenant_a, "slice/decrypt")
    assert seed.conversation_id is not None
    sent = datetime(2026, 3, 4, 5, 6, 7, tzinfo=UTC)
    async with admin_pool.acquire() as conn:
        await conn.execute(
            "UPDATE messages SET timestamp = $2 WHERE conversation_id = $1 AND position = 9",
            seed.conversation_id,
            sent,
        )
    await _set_watermark(admin_pool, seed.conversation_id, 7)

    loaded = await _load(app_pool, cipher, seed)

    assert loaded.new == (
        extraction_slice.SliceMessage(
            position=8, role="user", content="Message 8 content", timestamp=None
        ),
        extraction_slice.SliceMessage(
            position=9, role="assistant", content="Message 9 content", timestamp=sent.isoformat()
        ),
    )


async def test_target_counts_messages_a_role_filter_would_drop(
    app_pool: asyncpg.Pool, admin_pool: asyncpg.Pool, cipher: ContentCipher, tenant_a: UUID
) -> None:
    seed = await _thread(admin_pool, tenant_a, "slice/roles")
    assert seed.conversation_id is not None
    async with admin_pool.acquire() as conn:
        await conn.execute(
            "UPDATE messages SET role = 'tool' WHERE conversation_id = $1 AND position = 9",
            seed.conversation_id,
        )

    loaded = await _load(app_pool, cipher, seed)

    assert loaded.new[-1].role == "tool"
    assert loaded.target == MESSAGE_COUNT - 1


async def test_another_users_conversation_is_not_found(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    tenant_a: UUID,
    tenant_b: UUID,
) -> None:
    seed = await _thread(admin_pool, tenant_a, "slice/owner")
    assert seed.conversation_id is not None

    async with user_scoped_connection(app_pool, user_id=tenant_b) as conn:
        with pytest.raises(ConversationNotFoundError):
            await extraction_slice.load_extraction_slice(conn, cipher, seed.conversation_id)


async def test_undecryptable_message_fails_the_load(
    app_pool: asyncpg.Pool, admin_pool: asyncpg.Pool, cipher: ContentCipher, tenant_a: UUID
) -> None:
    seed = await _thread(admin_pool, tenant_a, "slice/corrupt")
    assert seed.conversation_id is not None
    async with admin_pool.acquire() as conn:
        await conn.execute(
            "UPDATE messages SET content_encrypted = $2"
            " WHERE conversation_id = $1 AND position = 9",
            seed.conversation_id,
            b"not a ciphertext of this key",
        )

    with pytest.raises(DecryptionError):
        await _load(app_pool, cipher, seed)


# -- list_thread_entries --------------------------------------------------------


async def _plant_entry(
    conn: asyncpg.Connection,
    cipher: ContentCipher,
    seed: TenantSeed,
    *,
    content: str,
    day: date,
    source_conversation_id: int | None,
    conversation_id: int | None = None,
    deleted: bool = False,
) -> int:
    content_ct, content_nonce = cipher.encrypt(content)
    now = datetime.now(UTC)
    return int(
        await conn.fetchval(
            """
            INSERT INTO entries
                (topic_id, user_id, date, content_encrypted, content_nonce,
                 conversation_id, source_conversation_id, deleted_at, tags,
                 created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, '{}', $9, $9)
            RETURNING id
            """,
            seed.topic_id,
            seed.user_id,
            day,
            content_ct,
            content_nonce,
            conversation_id,
            source_conversation_id,
            now if deleted else None,
            now,
        )
    )


async def _entries_as(
    app_pool: asyncpg.Pool, cipher: ContentCipher, user_id: UUID, conversation_id: int
) -> list[extraction_slice.ThreadEntry]:
    async with user_scoped_connection(app_pool, user_id=user_id) as conn:
        return await extraction_slice.list_thread_entries(conn, cipher, conversation_id)


async def test_thread_entries_keep_the_newest_hundred_oldest_first(
    app_pool: asyncpg.Pool, admin_pool: asyncpg.Pool, cipher: ContentCipher, tenant_a: UUID
) -> None:
    seed = await _thread(admin_pool, tenant_a, "slice/hundred")
    conv = seed.conversation_id
    assert conv is not None
    async with admin_pool.acquire() as conn, conn.transaction():
        for i in range(101):
            await _plant_entry(
                conn,
                cipher,
                seed,
                content=f"thread entry {i}",
                day=FIRST_DAY + timedelta(days=i),
                source_conversation_id=conv,
            )

    found = await _entries_as(app_pool, cipher, tenant_a, conv)

    assert [e.content for e in found] == [f"thread entry {i}" for i in range(1, 101)]
    assert [e.entry_date for e in found] == [FIRST_DAY + timedelta(days=i) for i in range(1, 101)]


async def test_thread_entries_order_by_event_day_then_id(
    app_pool: asyncpg.Pool, admin_pool: asyncpg.Pool, cipher: ContentCipher, tenant_a: UUID
) -> None:
    seed = await _thread(admin_pool, tenant_a, "slice/order")
    conv = seed.conversation_id
    assert conv is not None
    later = FIRST_DAY + timedelta(days=5)
    async with admin_pool.acquire() as conn, conn.transaction():
        for content, day in (("late", later), ("early", FIRST_DAY), ("late again", later)):
            await _plant_entry(
                conn, cipher, seed, content=content, day=day, source_conversation_id=conv
            )

    found = await _entries_as(app_pool, cipher, tenant_a, conv)

    assert [e.content for e in found] == ["early", "late", "late again"]


async def test_thread_entries_exclude_deleted_summary_unlinked_and_other_users_rows(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    tenant_a: UUID,
    tenant_b: UUID,
) -> None:
    seed_a = await _thread(admin_pool, tenant_a, "slice/excl-a")
    seed_b = await seed_for(admin_pool, tenant_b, topic_path="slice/excl-b", n_entries=0)
    conv = seed_a.conversation_id
    assert conv is not None
    async with admin_pool.acquire() as conn, conn.transaction():
        for seed, content, deleted, summary_of in (
            (seed_a, "kept", False, None),
            (seed_a, "deleted", True, None),
            (seed_a, "summary", False, conv),
            (seed_b, "other user", False, None),
        ):
            await _plant_entry(
                conn,
                cipher,
                seed,
                content=content,
                day=FIRST_DAY,
                source_conversation_id=conv,
                conversation_id=summary_of,
                deleted=deleted,
            )
        await _plant_entry(
            conn, cipher, seed_a, content="unlinked", day=FIRST_DAY, source_conversation_id=None
        )

    as_a = await _entries_as(app_pool, cipher, tenant_a, conv)
    as_b = await _entries_as(app_pool, cipher, tenant_b, conv)

    assert [e.content for e in as_a] == ["kept"]
    assert [e.content for e in as_b] == ["other user"]


async def test_thread_entries_skip_a_row_that_fails_to_decrypt(
    app_pool: asyncpg.Pool, admin_pool: asyncpg.Pool, cipher: ContentCipher, tenant_a: UUID
) -> None:
    seed = await _thread(admin_pool, tenant_a, "slice/corrupt-entry")
    conv = seed.conversation_id
    assert conv is not None
    async with admin_pool.acquire() as conn, conn.transaction():
        await _plant_entry(
            conn, cipher, seed, content="readable", day=FIRST_DAY, source_conversation_id=conv
        )
        corrupt = await _plant_entry(
            conn, cipher, seed, content="corrupt", day=FIRST_DAY, source_conversation_id=conv
        )
        await conn.execute(
            "UPDATE entries SET content_encrypted = $2 WHERE id = $1",
            corrupt,
            b"not a ciphertext of this key",
        )

    found = await _entries_as(app_pool, cipher, tenant_a, conv)

    assert [e.content for e in found] == ["readable"]


# -- advance_watermark ----------------------------------------------------------


async def _advance(app_pool: asyncpg.Pool, seed: TenantSeed, base: int | None, target: int) -> bool:
    assert seed.conversation_id is not None
    async with user_scoped_connection(app_pool, user_id=seed.user_id) as conn:
        return await extraction_slice.advance_watermark(conn, seed.conversation_id, base, target)


@pytest.mark.parametrize(
    ("stored", "base", "target"),
    [
        pytest.param(None, None, 0, id="null-to-first"),
        pytest.param(None, None, 9, id="null-to-max"),
        pytest.param(7, 7, 9, id="forward"),
    ],
)
async def test_advance_moves_the_watermark_from_a_current_base(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    tenant_a: UUID,
    stored: int | None,
    base: int | None,
    target: int,
) -> None:
    seed = await _thread(admin_pool, tenant_a, "slice/advance")
    assert seed.conversation_id is not None
    await _set_watermark(admin_pool, seed.conversation_id, stored)

    advanced = await _advance(app_pool, seed, base, target)

    assert advanced is True
    assert await _watermark(admin_pool, seed.conversation_id) == target


@pytest.mark.parametrize(
    ("stored", "base", "target"),
    [
        pytest.param(7, 5, 9, id="stale-base"),
        pytest.param(7, None, 9, id="null-base-over-set-watermark"),
        pytest.param(None, 7, 9, id="set-base-over-null-watermark"),
        pytest.param(7, 7, 7, id="target-equal"),
        pytest.param(7, 7, 3, id="target-below"),
    ],
)
async def test_advance_refuses_and_writes_nothing(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    tenant_a: UUID,
    stored: int | None,
    base: int | None,
    target: int,
) -> None:
    seed = await _thread(admin_pool, tenant_a, "slice/refuse")
    assert seed.conversation_id is not None
    await _set_watermark(admin_pool, seed.conversation_id, stored)

    advanced = await _advance(app_pool, seed, base, target)

    assert advanced is False
    assert await _watermark(admin_pool, seed.conversation_id) == stored


async def test_advance_cannot_touch_another_users_conversation(
    app_pool: asyncpg.Pool, admin_pool: asyncpg.Pool, tenant_a: UUID, tenant_b: UUID
) -> None:
    seed = await _thread(admin_pool, tenant_a, "slice/advance-owner")
    assert seed.conversation_id is not None

    async with user_scoped_connection(app_pool, user_id=tenant_b) as conn:
        advanced = await extraction_slice.advance_watermark(conn, seed.conversation_id, None, 9)

    assert advanced is False
    assert await _watermark(admin_pool, seed.conversation_id) is None
