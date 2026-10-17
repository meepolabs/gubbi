"""Merging re-sent synced conversations into stored history.

Every merge runs through the RLS-enforced ``app_pool``; stored state is read
back through ``admin_pool`` so assertions see the raw columns.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import asyncpg
import pytest
from gubbi_common.db.user_scoped import user_scoped_connection

from gubbi.constants import MAX_MSG_CHARS, MSG_TRUNCATION_MARKER
from gubbi.crypto.cipher import ContentCipher
from gubbi.storage.repositories import topics as topic_repo
from gubbi.storage.repositories.synced_conversations import (
    MergeResult,
    SyncedConversation,
    SyncedMessage,
    merge_synced_conversation,
)

pytestmark = pytest.mark.asyncio(loop_scope="session")

INBOX = "inbox"
PLATFORM = "claude"
CREATED = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)


def _msg(text: str, message_id: str | None = None, role: str = "user") -> SyncedMessage:
    return SyncedMessage(role=role, content=text, message_id=message_id)


def _conv(messages: list[SyncedMessage], platform_id: str = "thread-1") -> SyncedConversation:
    return SyncedConversation(
        platform=PLATFORM,
        platform_id=platform_id,
        title="Synced thread",
        topic=INBOX,
        created_at=CREATED,
        messages=tuple(messages),
    )


async def _merge(
    app_pool: asyncpg.Pool, cipher: ContentCipher, user: UUID, conv: SyncedConversation
) -> MergeResult:
    async with user_scoped_connection(app_pool, user_id=user) as conn:
        return await merge_synced_conversation(conn, cipher, conv)


async def _make_inbox(app_pool: asyncpg.Pool, user: UUID) -> None:
    async with user_scoped_connection(app_pool, user_id=user) as conn:
        await topic_repo.create(conn, topic=INBOX, title="Inbox")


async def _messages(admin_pool: asyncpg.Pool, conv_id: int) -> list[asyncpg.Record]:
    async with admin_pool.acquire() as conn:
        return list(
            await conn.fetch(
                "SELECT id, position, source_message_id, content_encrypted, content_nonce,"
                " role FROM messages WHERE conversation_id = $1 ORDER BY position",
                conv_id,
            )
        )


async def _conversation(admin_pool: asyncpg.Pool, conv_id: int) -> asyncpg.Record:
    async with admin_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT updated_at, processed_at, message_count FROM conversations WHERE id = $1",
            conv_id,
        )
    assert row is not None
    return row


async def _mark_processed(admin_pool: asyncpg.Pool, conv_id: int) -> None:
    async with admin_pool.acquire() as conn:
        await conn.execute(
            "UPDATE conversations SET processed_at = now(),"
            " updated_at = now() - interval '1 day' WHERE id = $1",
            conv_id,
        )


def _texts(cipher: ContentCipher, rows: list[asyncpg.Record]) -> list[str]:
    return [cipher.decrypt(bytes(r["content_encrypted"]), bytes(r["content_nonce"])) for r in rows]


def _ids(rows: list[asyncpg.Record]) -> list[Any]:
    return [r["source_message_id"] for r in rows]


@pytest.fixture
async def seeded(
    app_pool: asyncpg.Pool, admin_pool: asyncpg.Pool, cipher: ContentCipher, tenant_a: UUID
) -> int:
    """Tenant A's inbox holding thread-1 = [k1, k2], already extracted."""
    await _make_inbox(app_pool, tenant_a)
    first = await _merge(
        app_pool, cipher, tenant_a, _conv([_msg("hello one", "k1"), _msg("hello two", "k2")])
    )
    assert first.created
    await _mark_processed(admin_pool, first.conversation_id)
    return first.conversation_id


async def test_identical_replay_writes_nothing(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    tenant_a: UUID,
    seeded: int,
) -> None:
    before_rows = await _messages(admin_pool, seeded)
    before_conv = await _conversation(admin_pool, seeded)

    result = await _merge(
        app_pool, cipher, tenant_a, _conv([_msg("hello one", "k1"), _msg("hello two", "k2")])
    )

    assert (result.created, result.appended, result.replaced) == (False, 0, 0)
    assert not result.has_new_messages
    after_rows = await _messages(admin_pool, seeded)
    assert [bytes(r["content_encrypted"]) for r in after_rows] == [
        bytes(r["content_encrypted"]) for r in before_rows
    ]
    after_conv = await _conversation(admin_pool, seeded)
    assert after_conv["updated_at"] == before_conv["updated_at"]
    assert after_conv["processed_at"] == before_conv["processed_at"] is not None


async def test_unseen_ids_append_in_payload_order(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    tenant_a: UUID,
    seeded: int,
) -> None:
    payload = [_msg("hello one", "k1"), _msg("new one", "n1"), _msg("hello two", "k2")]
    payload.append(_msg("new two", "n2"))

    result = await _merge(app_pool, cipher, tenant_a, _conv(payload))

    assert (result.appended, result.replaced) == (2, 0)
    rows = await _messages(admin_pool, seeded)
    assert [(r["position"], r["source_message_id"]) for r in rows] == [
        (0, "k1"),
        (1, "k2"),
        (2, "n1"),
        (3, "n2"),
    ]
    assert _texts(cipher, rows)[2:] == ["new one", "new two"]


async def test_changed_text_is_replaced_in_place_without_reextraction(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    tenant_a: UUID,
    seeded: int,
) -> None:
    before_conv = await _conversation(admin_pool, seeded)

    result = await _merge(
        app_pool,
        cipher,
        tenant_a,
        _conv([_msg("hello one", "k1"), _msg("zebra crossing", "k2")]),
    )

    assert (result.appended, result.replaced) == (0, 1)
    rows = await _messages(admin_pool, seeded)
    assert _texts(cipher, rows) == ["hello one", "zebra crossing"]
    async with admin_pool.acquire() as conn:
        hits = await conn.fetch(
            "SELECT source_message_id FROM messages WHERE conversation_id = $1"
            " AND search_vector @@ plainto_tsquery('english', $2)",
            seeded,
            "zebra",
        )
        stale = await conn.fetchval(
            "SELECT count(*) FROM messages WHERE conversation_id = $1"
            " AND search_vector @@ plainto_tsquery('english', 'two')",
            seeded,
        )
    assert [h["source_message_id"] for h in hits] == ["k2"]
    assert stale == 0
    after_conv = await _conversation(admin_pool, seeded)
    assert after_conv["updated_at"] == before_conv["updated_at"]
    assert after_conv["message_count"] == before_conv["message_count"] == 2
    assert after_conv["processed_at"] == before_conv["processed_at"] is not None


async def test_stored_messages_missing_from_payload_are_kept(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    tenant_a: UUID,
    seeded: int,
) -> None:
    result = await _merge(
        app_pool, cipher, tenant_a, _conv([_msg("hello one", "k1"), _msg("other branch", "b2")])
    )

    assert (result.appended, result.replaced) == (1, 0)
    rows = await _messages(admin_pool, seeded)
    assert _ids(rows) == ["k1", "k2", "b2"]
    assert _texts(cipher, rows) == ["hello one", "hello two", "other branch"]


async def test_duplicate_ids_in_one_payload_keep_the_first(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    tenant_a: UUID,
    seeded: int,
) -> None:
    payload = [_msg("hello one", "k1"), _msg("first copy", "n1"), _msg("second copy", "n1")]

    result = await _merge(app_pool, cipher, tenant_a, _conv(payload))

    assert (result.appended, result.duplicate_ids) == (1, 1)
    rows = await _messages(admin_pool, seeded)
    assert _texts(cipher, rows)[-1] == "first copy"
    assert _ids(rows).count("n1") == 1


@pytest.fixture
async def seeded_idless(
    app_pool: asyncpg.Pool, admin_pool: asyncpg.Pool, cipher: ContentCipher, tenant_a: UUID
) -> int:
    """Tenant A's inbox holding thread-1 = [a, b, c] stored without source ids."""
    await _make_inbox(app_pool, tenant_a)
    first = await _merge(
        app_pool, cipher, tenant_a, _conv([_msg("alpha"), _msg("bravo"), _msg("charlie")])
    )
    await _mark_processed(admin_pool, first.conversation_id)
    return first.conversation_id


async def test_new_idless_conversation_is_stored_with_null_ids(
    admin_pool: asyncpg.Pool, cipher: ContentCipher, seeded_idless: int
) -> None:
    rows = await _messages(admin_pool, seeded_idless)

    assert _ids(rows) == [None, None, None]
    assert [r["position"] for r in rows] == [0, 1, 2]
    assert _texts(cipher, rows) == ["alpha", "bravo", "charlie"]


async def test_idless_payload_extending_the_stored_prefix_appends_the_suffix(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    tenant_a: UUID,
    seeded_idless: int,
) -> None:
    payload = [_msg("alpha"), _msg("bravo"), _msg("charlie"), _msg("delta"), _msg("echo")]

    result = await _merge(app_pool, cipher, tenant_a, _conv(payload))

    assert (result.appended, result.legacy_identity_conflict) == (2, False)
    rows = await _messages(admin_pool, seeded_idless)
    assert _texts(cipher, rows) == ["alpha", "bravo", "charlie", "delta", "echo"]
    assert [r["position"] for r in rows] == [0, 1, 2, 3, 4]


async def test_idless_payload_shorter_than_stored_prefix_is_a_noop(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    tenant_a: UUID,
    seeded_idless: int,
) -> None:
    before_conv = await _conversation(admin_pool, seeded_idless)

    result = await _merge(app_pool, cipher, tenant_a, _conv([_msg("alpha"), _msg("bravo")]))

    assert (result.appended, result.replaced, result.legacy_identity_conflict) == (0, 0, False)
    assert len(await _messages(admin_pool, seeded_idless)) == 3
    assert (await _conversation(admin_pool, seeded_idless))["updated_at"] == before_conv[
        "updated_at"
    ]


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(["ALPHA", "bravo", "charlie", "delta"], id="diverges-at-first"),
        pytest.param(["alpha", "bravo", "charlie!", "delta"], id="diverges-at-last-stored"),
        pytest.param(["alpha", "bravx"], id="shorter-and-diverging"),
    ],
)
async def test_idless_payload_diverging_from_stored_text_writes_nothing(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    tenant_a: UUID,
    seeded_idless: int,
    payload: list[str],
) -> None:
    before = await _messages(admin_pool, seeded_idless)

    result = await _merge(app_pool, cipher, tenant_a, _conv([_msg(t) for t in payload]))

    assert result.legacy_identity_conflict
    assert (result.appended, result.replaced) == (0, 0)
    after = await _messages(admin_pool, seeded_idless)
    assert len(after) == len(before) == 3
    assert _texts(cipher, after) == ["alpha", "bravo", "charlie"]


@pytest.mark.parametrize(
    ("thread", "payload"),
    [
        pytest.param(
            "ids",
            [_msg("hello one", "k1"), _msg("hello two", "k2"), _msg("no id here")],
            id="mixed-payload-on-id-thread",
        ),
        pytest.param(
            "idless",
            [_msg("alpha"), _msg("bravo"), _msg("charlie"), _msg("delta", "d1")],
            id="mixed-payload-on-idless-thread",
        ),
        pytest.param(
            "idless",
            [_msg("alpha", "a1"), _msg("bravo", "b1"), _msg("charlie", "c1"), _msg("d", "d1")],
            id="id-payload-on-idless-thread",
        ),
        pytest.param(
            "ids",
            [_msg("hello one"), _msg("hello two"), _msg("no ids at all")],
            id="idless-payload-on-id-thread",
        ),
    ],
)
async def test_payload_whose_identity_cannot_be_matched_writes_nothing(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    tenant_a: UUID,
    thread: str,
    payload: list[SyncedMessage],
) -> None:
    await _make_inbox(app_pool, tenant_a)
    stored = (
        [_msg("hello one", "k1"), _msg("hello two", "k2")]
        if thread == "ids"
        else [_msg("alpha"), _msg("bravo"), _msg("charlie")]
    )
    conv_id = (await _merge(app_pool, cipher, tenant_a, _conv(stored))).conversation_id

    result = await _merge(app_pool, cipher, tenant_a, _conv(payload))

    assert result.legacy_identity_conflict
    assert (result.appended, result.replaced) == (0, 0)
    assert len(await _messages(admin_pool, conv_id)) == len(stored)


async def test_append_marks_the_thread_for_extraction(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    tenant_a: UUID,
    seeded: int,
) -> None:
    before = await _conversation(admin_pool, seeded)

    result = await _merge(
        app_pool,
        cipher,
        tenant_a,
        _conv([_msg("hello one", "k1"), _msg("reply", "n1", role="assistant")]),
    )

    assert result.has_new_messages
    after = await _conversation(admin_pool, seeded)
    assert after["updated_at"] > before["updated_at"]
    assert after["message_count"] == 3
    assert after["processed_at"] is None
    async with admin_pool.acquire() as conn:
        participants = await conn.fetchval(
            "SELECT participants FROM conversations WHERE id = $1", seeded
        )
    assert sorted(participants) == ["assistant", "user"]


async def test_over_limit_message_is_stored_cut_and_its_replay_is_a_noop(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    tenant_a: UUID,
    seeded: int,
) -> None:
    payload = [_msg("hello one", "k1"), _msg("x" * (MAX_MSG_CHARS + 50), "long")]

    first = await _merge(app_pool, cipher, tenant_a, _conv(payload))
    replay = await _merge(app_pool, cipher, tenant_a, _conv(payload))

    assert (first.appended, first.truncated) == (1, 1)
    assert (replay.appended, replay.replaced, replay.truncated) == (0, 0, 0)
    stored = _texts(cipher, await _messages(admin_pool, seeded))[-1]
    assert len(stored) == MAX_MSG_CHARS
    assert stored.endswith(MSG_TRUNCATION_MARKER)


async def test_linked_entry_is_created_on_first_insert_only(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    tenant_a: UUID,
    seeded: int,
) -> None:
    async with admin_pool.acquire() as conn:
        before = await conn.fetch(
            "SELECT id, updated_at FROM entries WHERE conversation_id = $1", seeded
        )

    await _merge(app_pool, cipher, tenant_a, _conv([_msg("hello one", "k1"), _msg("x", "n1")]))

    async with admin_pool.acquire() as conn:
        after = await conn.fetch(
            "SELECT id, updated_at FROM entries WHERE conversation_id = $1", seeded
        )
    assert len(before) == 1
    assert [tuple(r) for r in after] == [tuple(r) for r in before]


async def test_another_users_conversation_with_the_same_source_id_is_untouched(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    tenant_a: UUID,
    tenant_b: UUID,
) -> None:
    await _make_inbox(app_pool, tenant_a)
    await _make_inbox(app_pool, tenant_b)
    theirs = await _merge(app_pool, cipher, tenant_b, _conv([_msg("private", "k1")]))
    before = await _messages(admin_pool, theirs.conversation_id)

    mine = await _merge(
        app_pool, cipher, tenant_a, _conv([_msg("changed", "k1"), _msg("more", "n1")])
    )

    assert mine.created
    assert mine.conversation_id != theirs.conversation_id
    assert (mine.appended, mine.replaced) == (2, 0)
    after = await _messages(admin_pool, theirs.conversation_id)
    assert [tuple(r) for r in after] == [tuple(r) for r in before]
    assert (await _conversation(admin_pool, theirs.conversation_id))["message_count"] == 1


async def test_concurrent_merges_store_each_source_id_once(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    tenant_a: UUID,
) -> None:
    await _make_inbox(app_pool, tenant_a)
    payload = [_msg(f"turn {i}", f"m{i}") for i in range(4)]

    for round_no in range(20):
        conv = _conv(payload, platform_id=f"race-{round_no}")
        results = await asyncio.gather(
            _merge(app_pool, cipher, tenant_a, conv),
            _merge(app_pool, cipher, tenant_a, conv),
        )

        assert sorted(r.created for r in results) == [False, True], f"round {round_no}"
        async with admin_pool.acquire() as conn:
            conv_ids = await conn.fetch(
                "SELECT id FROM conversations WHERE platform_id = $1", conv.platform_id
            )
        assert len(conv_ids) == 1, f"round {round_no}"
        rows = await _messages(admin_pool, int(conv_ids[0]["id"]))
        assert _ids(rows) == ["m0", "m1", "m2", "m3"], f"round {round_no}"
        assert [r["position"] for r in rows] == [0, 1, 2, 3], f"round {round_no}"


async def test_concurrent_appends_to_a_stored_thread_keep_positions_contiguous(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    tenant_a: UUID,
) -> None:
    await _make_inbox(app_pool, tenant_a)

    for round_no in range(20):
        platform_id = f"grow-{round_no}"
        stored = await _merge(app_pool, cipher, tenant_a, _conv([_msg("root", "r")], platform_id))
        await asyncio.gather(
            _merge(
                app_pool, cipher, tenant_a, _conv([_msg("root", "r"), _msg("x", "a")], platform_id)
            ),
            _merge(
                app_pool, cipher, tenant_a, _conv([_msg("root", "r"), _msg("y", "b")], platform_id)
            ),
        )

        rows = await _messages(admin_pool, stored.conversation_id)
        assert sorted(_ids(rows)) == ["a", "b", "r"], f"round {round_no}"
        assert [r["position"] for r in rows] == [0, 1, 2], f"round {round_no}"
        conv = await _conversation(admin_pool, stored.conversation_id)
        assert conv["message_count"] == 3, f"round {round_no}"
