"""Schema, backfill and title-identity contract for the continued-threads revision.

The round trip and the downgrade refusal run on throwaway databases on the
working cluster, so a downgrade never disturbs the shared ``journal_test``.
Index behaviour and the title-keyed save run on the shared RLS database, the
first through the admin pool and the second through the RLS-enforced app pool.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import asyncpg
import pytest
import pytest_asyncio
from gubbi_common.db.user_scoped import user_scoped_connection

from gubbi.crypto.cipher import ContentCipher
from gubbi.models.conversation import Message
from gubbi.storage.exceptions import ConversationNotFoundError
from gubbi.storage.repositories import conversations as conv_repo
from gubbi.storage.repositories import topics as topic_repo
from tests.integration.test_extraction_queue_columns_migration import (
    _alembic_off_the_loop,
    _run_alembic,
    _scratch_database,
)

pytestmark = [pytest.mark.integration]

_REVISION = "0007_continued_threads"
_PREVIOUS_REVISION = "0006_llm_wallet"
_NONCE = b"\x00" * 12

EXPECTED_COLUMNS: dict[tuple[str, str], tuple[str, str, str | None]] = {
    ("conversations", "extracted_through_position"): ("integer", "YES", None),
    ("messages", "source_message_id"): ("text", "YES", None),
}

_TITLE_CONSTRAINT_INDEX = (
    "CREATE UNIQUE INDEX conversations_topic_id_slug_key"
    " ON public.conversations USING btree (topic_id, slug)"
)
_TITLE_PARTIAL_INDEX = (
    "CREATE UNIQUE INDEX conversations_topic_id_slug_mcp_key"
    " ON public.conversations USING btree (topic_id, slug) WHERE (platform_id IS NULL)"
)
_SOURCE_ID_INDEX = (
    "CREATE UNIQUE INDEX idx_messages_conversation_source_id"
    " ON public.messages USING btree (conversation_id, source_message_id)"
    " WHERE (source_message_id IS NOT NULL)"
)
_INDEX_NAMES = (
    "conversations_topic_id_slug_key",
    "conversations_topic_id_slug_mcp_key",
    "idx_messages_conversation_source_id",
)


@dataclass(frozen=True)
class SchemaSnapshot:
    columns: dict[tuple[str, str], tuple[str, str, str | None]]
    indexes: dict[str, str]
    constraints: frozenset[str]


_AT_HEAD = SchemaSnapshot(
    columns=EXPECTED_COLUMNS,
    indexes={
        "conversations_topic_id_slug_mcp_key": _TITLE_PARTIAL_INDEX,
        "idx_messages_conversation_source_id": _SOURCE_ID_INDEX,
    },
    constraints=frozenset(),
)
_AT_PREVIOUS = SchemaSnapshot(
    columns={},
    indexes={"conversations_topic_id_slug_key": _TITLE_CONSTRAINT_INDEX},
    constraints=frozenset({"conversations_topic_id_slug_key"}),
)


@dataclass(frozen=True)
class BackfillCase:
    """A conversation planted before the upgrade and the watermark it must get."""

    conversation_id: int
    expected: int | None


@dataclass(frozen=True)
class RoundTrip:
    after_first_upgrade: SchemaSnapshot
    after_downgrade: SchemaSnapshot
    after_second_upgrade: SchemaSnapshot
    planted: dict[str, BackfillCase]
    watermarks: dict[int, int | None]


async def _snapshot(dsn: str) -> SchemaSnapshot:
    conn = await asyncpg.connect(dsn, timeout=5)
    try:
        column_rows = await conn.fetch(
            """
            SELECT table_name, column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND (table_name, column_name) IN (SELECT * FROM unnest($1::text[], $2::text[]))
            """,
            [table for table, _ in EXPECTED_COLUMNS],
            [column for _, column in EXPECTED_COLUMNS],
        )
        index_rows = await conn.fetch(
            "SELECT indexname, indexdef FROM pg_indexes"
            " WHERE schemaname = 'public' AND indexname = ANY($1::text[])",
            list(_INDEX_NAMES),
        )
        constraint_rows = await conn.fetch(
            "SELECT conname FROM pg_constraint"
            " WHERE conrelid = 'public.conversations'::regclass AND conname = ANY($1::text[])",
            list(_INDEX_NAMES),
        )
    finally:
        await conn.close()
    return SchemaSnapshot(
        columns={
            (r["table_name"], r["column_name"]): (
                r["data_type"],
                r["is_nullable"],
                r["column_default"],
            )
            for r in column_rows
        },
        indexes={r["indexname"]: r["indexdef"] for r in index_rows},
        constraints=frozenset(r["conname"] for r in constraint_rows),
    )


async def _insert_user_topic(conn: asyncpg.Connection, label: str) -> tuple[UUID, int]:
    user_id = await conn.fetchval(
        "INSERT INTO users (id, email) VALUES (gen_random_uuid(), $1) RETURNING id",
        f"{label}-{uuid.uuid4().hex}@test.local",
    )
    topic_id = await conn.fetchval(
        "INSERT INTO topics (path, title, user_id) VALUES ('threads/topic', 'Threads', $1)"
        " RETURNING id",
        user_id,
    )
    return user_id, int(topic_id)


async def _insert_conversation(
    conn: asyncpg.Connection,
    user_id: UUID,
    topic_id: int,
    *,
    slug: str,
    processed: bool = False,
    platform_id: str | None = None,
) -> int:
    conversation_id = await conn.fetchval(
        """
        INSERT INTO conversations
            (topic_id, user_id, slug, title_encrypted, title_nonce,
             summary_encrypted, summary_nonce, platform, platform_id, processed_at)
        VALUES ($1, $2, $3, '\\x00', $4, '\\x00', $4, $5, $6,
                CASE WHEN $7 THEN now() END)
        RETURNING id
        """,
        topic_id,
        user_id,
        slug,
        _NONCE,
        "claude" if platform_id is not None else None,
        platform_id,
        processed,
    )
    return int(conversation_id)


async def _insert_messages(
    conn: asyncpg.Connection, user_id: UUID, conversation_id: int, positions: list[int]
) -> None:
    for position in positions:
        await conn.execute(
            """
            INSERT INTO messages
                (conversation_id, user_id, role, content_encrypted, content_nonce, position)
            VALUES ($1, $2, 'user', '\\x00', $3, $4)
            """,
            conversation_id,
            user_id,
            _NONCE,
            position,
        )


async def _plant_backfill_rows(dsn: str) -> dict[str, BackfillCase]:
    """Per user: a processed and an unprocessed thread with messages, and a processed empty one.

    Positions are planted out of order and with a gap so ``max`` is the only
    aggregate that yields the expected watermark.
    """
    planted: dict[str, BackfillCase] = {}
    conn = await asyncpg.connect(dsn, timeout=5)
    try:
        async with conn.transaction():
            for label, positions in (("first", [2, 0, 5, 1]), ("second", [0, 3])):
                user_id, topic_id = await _insert_user_topic(conn, label)
                processed = await _insert_conversation(
                    conn, user_id, topic_id, slug="processed", processed=True
                )
                await _insert_messages(conn, user_id, processed, positions)
                unprocessed = await _insert_conversation(conn, user_id, topic_id, slug="pending")
                await _insert_messages(conn, user_id, unprocessed, positions)
                empty = await _insert_conversation(
                    conn, user_id, topic_id, slug="empty", processed=True
                )
                planted[f"{label}-processed"] = BackfillCase(processed, max(positions))
                planted[f"{label}-unprocessed"] = BackfillCase(unprocessed, None)
                planted[f"{label}-processed-empty"] = BackfillCase(empty, None)
    finally:
        await conn.close()
    return planted


async def _read_watermarks(dsn: str, ids: list[int]) -> dict[int, int | None]:
    conn = await asyncpg.connect(dsn, timeout=5)
    try:
        rows = await conn.fetch(
            "SELECT id, extracted_through_position FROM conversations WHERE id = ANY($1::int[])",
            ids,
        )
    finally:
        await conn.close()
    return {int(r["id"]): r["extracted_through_position"] for r in rows}


@pytest_asyncio.fixture(scope="module", loop_scope="session")
async def round_trip() -> AsyncIterator[RoundTrip]:
    """Upgrade to head, step back one revision, plant rows, and upgrade again."""
    async with _scratch_database() as dsn:
        await _alembic_off_the_loop(dsn, "upgrade", "head")
        after_first_upgrade = await _snapshot(dsn)
        await _alembic_off_the_loop(dsn, "downgrade", _PREVIOUS_REVISION)
        after_downgrade = await _snapshot(dsn)
        planted = await _plant_backfill_rows(dsn)
        await _alembic_off_the_loop(dsn, "upgrade", "head")
        after_second_upgrade = await _snapshot(dsn)
        watermarks = await _read_watermarks(dsn, [c.conversation_id for c in planted.values()])
        yield RoundTrip(
            after_first_upgrade=after_first_upgrade,
            after_downgrade=after_downgrade,
            after_second_upgrade=after_second_upgrade,
            planted=planted,
            watermarks=watermarks,
        )


@pytest.mark.parametrize("stage", ["after_first_upgrade", "after_second_upgrade"])
def test_upgrade_replaces_the_title_constraint_and_adds_columns_and_index(
    round_trip: RoundTrip, stage: str
) -> None:
    assert getattr(round_trip, stage) == _AT_HEAD


def test_downgrade_restores_the_title_constraint_and_drops_the_additions(
    round_trip: RoundTrip,
) -> None:
    assert round_trip.after_downgrade == _AT_PREVIOUS


@pytest.mark.parametrize(
    "case",
    [
        "first-processed",
        "first-unprocessed",
        "first-processed-empty",
        "second-processed",
        "second-unprocessed",
        "second-processed-empty",
    ],
)
def test_backfill_sets_the_watermark_only_on_processed_threads_with_messages(
    round_trip: RoundTrip, case: str
) -> None:
    planted = round_trip.planted[case]

    assert round_trip.watermarks[planted.conversation_id] == planted.expected


@pytest.mark.asyncio(loop_scope="session")
async def test_downgrade_refuses_two_same_title_synced_threads_in_one_topic() -> None:
    async with _scratch_database() as dsn:
        await _alembic_off_the_loop(dsn, "upgrade", "head")
        conn = await asyncpg.connect(dsn, timeout=5)
        try:
            user_id, topic_id = await _insert_user_topic(conn, "downgrade")
            for platform_id in ("thread-a", "thread-b"):
                await _insert_conversation(
                    conn, user_id, topic_id, slug="same-title", platform_id=platform_id
                )
        finally:
            await conn.close()

        result = await asyncio.to_thread(_run_alembic, dsn, "downgrade", _PREVIOUS_REVISION)

        conn = await asyncpg.connect(dsn, timeout=5)
        try:
            version = await conn.fetchval("SELECT version_num FROM alembic_version")
        finally:
            await conn.close()
        schema = await _snapshot(dsn)

    assert result.returncode != 0, "downgrade must refuse rows the restored constraint rejects"
    assert "conversations_topic_id_slug_key" in result.stderr
    assert version == _REVISION
    assert schema == _AT_HEAD, "a refused downgrade must roll back every step"


async def _admin_user_topic(admin_pool: asyncpg.Pool, user_id: UUID, path: str) -> int:
    async with admin_pool.acquire() as conn:
        topic_id = await conn.fetchval(
            "INSERT INTO topics (path, title, user_id) VALUES ($1, 'Threads', $2) RETURNING id",
            path,
            user_id,
        )
    return int(topic_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_same_title_synced_threads_insert_as_separate_rows(
    admin_pool: asyncpg.Pool, tenant_a: UUID
) -> None:
    topic_id = await _admin_user_topic(admin_pool, tenant_a, "threads/synced")

    async with admin_pool.acquire() as conn:
        for platform_id in ("thread-a", "thread-b"):
            await _insert_conversation(
                conn, tenant_a, topic_id, slug="same-title", platform_id=platform_id
            )
        count = await conn.fetchval(
            "SELECT count(*) FROM conversations WHERE topic_id = $1 AND slug = 'same-title'",
            topic_id,
        )

    assert count == 2


@pytest.mark.asyncio(loop_scope="session")
async def test_same_title_unsynced_conversations_still_collide(
    admin_pool: asyncpg.Pool, tenant_a: UUID
) -> None:
    topic_id = await _admin_user_topic(admin_pool, tenant_a, "threads/titled")

    async with admin_pool.acquire() as conn:
        await _insert_conversation(conn, tenant_a, topic_id, slug="same-title")
        with pytest.raises(
            asyncpg.UniqueViolationError, match="conversations_topic_id_slug_mcp_key"
        ):
            await _insert_conversation(conn, tenant_a, topic_id, slug="same-title")


async def _insert_sourced_message(
    conn: asyncpg.Connection,
    user_id: UUID,
    conversation_id: int,
    position: int,
    source_message_id: str | None,
) -> None:
    await conn.execute(
        """
        INSERT INTO messages
            (conversation_id, user_id, role, content_encrypted, content_nonce, position,
             source_message_id)
        VALUES ($1, $2, 'user', '\\x00', $3, $4, $5)
        """,
        conversation_id,
        user_id,
        _NONCE,
        position,
        source_message_id,
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_a_source_message_id_is_stored_once_per_conversation(
    admin_pool: asyncpg.Pool, tenant_a: UUID
) -> None:
    topic_id = await _admin_user_topic(admin_pool, tenant_a, "threads/sourced")

    async with admin_pool.acquire() as conn:
        first = await _insert_conversation(
            conn, tenant_a, topic_id, slug="one", platform_id="thread-1"
        )
        second = await _insert_conversation(
            conn, tenant_a, topic_id, slug="two", platform_id="thread-2"
        )
        await _insert_sourced_message(conn, tenant_a, first, 0, "msg-1")
        await _insert_sourced_message(conn, tenant_a, second, 0, "msg-1")
        await _insert_sourced_message(conn, tenant_a, first, 1, None)
        await _insert_sourced_message(conn, tenant_a, first, 2, None)
        with pytest.raises(
            asyncpg.UniqueViolationError, match="idx_messages_conversation_source_id"
        ):
            await _insert_sourced_message(conn, tenant_a, first, 3, "msg-1")
        stored = await conn.fetchval(
            "SELECT count(*) FROM messages WHERE conversation_id = ANY($1::int[])",
            [first, second],
        )

    assert stored == 4


_MCP_TOPIC = "threads/mcp"
_MCP_TITLE = "Shared Title"
_MCP_SLUG = "shared-title"


async def _plant_synced_thread(
    admin_pool: asyncpg.Pool, user_id: UUID, cipher: ContentCipher
) -> int:
    """A synced conversation titled like the MCP save, created long before it."""
    title_ct, title_nonce = cipher.encrypt(_MCP_TITLE)
    summary_ct, summary_nonce = cipher.encrypt("")
    content_ct, content_nonce = cipher.encrypt("synced message")
    async with admin_pool.acquire() as conn, conn.transaction():
        topic_id = await conn.fetchval(
            "SELECT id FROM topics WHERE path = $1 AND user_id = $2", _MCP_TOPIC, user_id
        )
        conversation_id = await conn.fetchval(
            """
            INSERT INTO conversations
                (topic_id, user_id, slug, title_encrypted, title_nonce,
                 summary_encrypted, summary_nonce, platform, platform_id, message_count,
                 created_at, json_path)
            VALUES ($1, $2, $3, $4, $5, $6, $7, 'claude', 'synced-thread', 1, $8,
                    'conversations_json/synced.json')
            RETURNING id
            """,
            topic_id,
            user_id,
            _MCP_SLUG,
            title_ct,
            title_nonce,
            summary_ct,
            summary_nonce,
            datetime(2001, 1, 1, tzinfo=UTC),
        )
        await conn.execute(
            """
            INSERT INTO messages
                (conversation_id, user_id, role, content_encrypted, content_nonce, position)
            VALUES ($1, $2, 'user', $3, $4, 0)
            """,
            conversation_id,
            user_id,
            content_ct,
            content_nonce,
        )
    return int(conversation_id)


async def _mcp_save(
    app_pool: asyncpg.Pool,
    user_id: UUID,
    cipher: ContentCipher,
    json_dir: Path,
    contents: list[str],
) -> conv_repo.SaveConversationResult:
    async with user_scoped_connection(app_pool, user_id=user_id) as conn:
        return await conv_repo.save_conversation(
            conn,
            cipher,
            conversations_json_dir=json_dir,
            topic=_MCP_TOPIC,
            title=_MCP_TITLE,
            messages=[Message(role="user", content=c, timestamp=None) for c in contents],
            summary="mcp summary",
            date="2026-10-17",
        )


async def _create_mcp_topic(app_pool: asyncpg.Pool, user_id: UUID) -> None:
    async with user_scoped_connection(app_pool, user_id=user_id) as conn:
        await topic_repo.create(conn, topic=_MCP_TOPIC, title="Threads")


async def _message_texts(
    admin_pool: asyncpg.Pool, cipher: ContentCipher, conversation_id: int
) -> list[str]:
    async with admin_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT content_encrypted, content_nonce FROM messages"
            " WHERE conversation_id = $1 ORDER BY position",
            conversation_id,
        )
    return [cipher.decrypt(r["content_encrypted"], r["content_nonce"]) for r in rows]


@pytest.mark.asyncio(loop_scope="session")
async def test_a_second_mcp_save_of_a_title_updates_the_same_row(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    tmp_path: Path,
    tenant_a: UUID,
) -> None:
    await _create_mcp_topic(app_pool, tenant_a)

    first = await _mcp_save(app_pool, tenant_a, cipher, tmp_path, ["one", "two"])
    second = await _mcp_save(app_pool, tenant_a, cipher, tmp_path, ["three"])

    async with admin_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, platform_id FROM conversations WHERE slug = $1", _MCP_SLUG
        )
    assert first.is_update is False
    assert second.is_update is True
    assert second.conversation_id == first.conversation_id
    assert [(r["id"], r["platform_id"]) for r in rows] == [(first.conversation_id, None)]
    assert await _message_texts(admin_pool, cipher, first.conversation_id) == ["three"]


@pytest.mark.asyncio(loop_scope="session")
async def test_an_mcp_save_never_overwrites_a_synced_thread_with_its_title(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    tmp_path: Path,
    tenant_a: UUID,
) -> None:
    await _create_mcp_topic(app_pool, tenant_a)
    synced_id = await _plant_synced_thread(admin_pool, tenant_a, cipher)

    result = await _mcp_save(app_pool, tenant_a, cipher, tmp_path, ["mcp message"])

    assert result.is_update is False
    assert result.superseded_json_path is None
    assert result.conversation_id != synced_id
    assert await _message_texts(admin_pool, cipher, synced_id) == ["synced message"]
    assert await _message_texts(admin_pool, cipher, result.conversation_id) == ["mcp message"]


@pytest.mark.asyncio(loop_scope="session")
async def test_an_mcp_save_takes_its_created_date_from_its_own_row_not_a_synced_one(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    tmp_path: Path,
    tenant_a: UUID,
) -> None:
    await _create_mcp_topic(app_pool, tenant_a)
    await _plant_synced_thread(admin_pool, tenant_a, cipher)

    result = await _mcp_save(app_pool, tenant_a, cipher, tmp_path, ["mcp message"])

    async with admin_pool.acquire() as conn:
        json_path = await conn.fetchval(
            "SELECT json_path FROM conversations WHERE id = $1", result.conversation_id
        )
    archive = json.loads((tmp_path / Path(str(json_path)).name).read_text(encoding="utf-8"))
    assert archive["meta"]["created"] == "2026-10-17"


@pytest.mark.asyncio(loop_scope="session")
async def test_reading_by_title_ignores_synced_threads(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    tmp_path: Path,
    tenant_a: UUID,
) -> None:
    await _create_mcp_topic(app_pool, tenant_a)
    await _plant_synced_thread(admin_pool, tenant_a, cipher)

    async with user_scoped_connection(app_pool, user_id=tenant_a) as conn:
        with pytest.raises(ConversationNotFoundError):
            await conv_repo.get_conversation(conn, cipher, _MCP_TOPIC, _MCP_TITLE)

    await _mcp_save(app_pool, tenant_a, cipher, tmp_path, ["mcp message"])

    async with user_scoped_connection(app_pool, user_id=tenant_a) as conn:
        _meta, messages = await conv_repo.get_conversation(conn, cipher, _MCP_TOPIC, _MCP_TITLE)
    assert [m.content for m in messages] == ["mcp message"]
