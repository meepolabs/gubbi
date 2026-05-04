"""Integration tests for conversation upsert correctness.

Covers ``is_update`` tracking via the ``xmax`` RETURNING trick,
deferred cleanup of the superseded JSON archive on re-save, and
rollback safety of the previous archive file.
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest
from gubbi_common.db.user_scoped import user_scoped_connection

from gubbi.core.crypto import ContentCipher
from gubbi.models.conversation import Message
from gubbi.storage.repositories import conversations as conv_repo
from gubbi.storage.repositories import topics as topic_repo

pytestmark = pytest.mark.asyncio(loop_scope="session")


async def test_upsert_is_update_tracking(
    app_pool,  #: asyncpg.Pool (RLS-enforced)
    cipher: ContentCipher,
    tmp_path: Path,
    tenant_a: UUID,
) -> None:
    """``_upsert_conversation_record`` must report ``is_update`` correctly
    for both insert and update paths via the single-round-trip xmax
    derivation."""
    async with user_scoped_connection(app_pool, user_id=tenant_a) as conn:
        await topic_repo.create(conn, topic="upsert-test", title="Upsert Test")

        messages = [
            Message(role="user", content="Hello", timestamp=None),
            Message(role="assistant", content="Hi there", timestamp=None),
        ]

        first = await conv_repo.save_conversation(
            conn,
            cipher,
            conversations_json_dir=tmp_path,
            topic="upsert-test",
            title="Same Chat",
            messages=messages,
            summary="First summary",
        )
        assert first.is_update is False, "first save should report is_update=False"

        messages_v2 = messages + [
            Message(role="user", content="Follow up", timestamp=None),
            Message(role="assistant", content="Sure thing", timestamp=None),
        ]
        second = await conv_repo.save_conversation(
            conn,
            cipher,
            conversations_json_dir=tmp_path,
            topic="upsert-test",
            title="Same Chat",
            messages=messages_v2,
            summary="Updated summary",
        )
        assert second.is_update is True, "re-save should report is_update=True"


async def test_superseded_json_is_deleted_on_resave(
    app_pool,  #: asyncpg.Pool (RLS-enforced)
    admin_pool,  #: asyncpg.Pool (BYPASSRLS)
    cipher: ContentCipher,
    tmp_path: Path,
    tenant_a: UUID,
) -> None:
    """Re-saving a conversation supersedes the previous JSON archive;
    the caller deletes it after the transaction commits."""
    async with user_scoped_connection(app_pool, user_id=tenant_a) as conn:
        await topic_repo.create(conn, topic="superseded-test", title="Superseded Test")

        messages = [
            Message(role="user", content="Version 1", timestamp=None),
        ]

        await conv_repo.save_conversation(
            conn,
            cipher,
            conversations_json_dir=tmp_path,
            topic="superseded-test",
            title="Same Title",
            messages=messages,
            summary="V1",
        )

    async with admin_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT json_path FROM conversations WHERE slug = $1",
            "same-title",
        )
    assert row is not None
    old_json_path = str(row["json_path"])
    old_file = tmp_path / Path(old_json_path).name
    assert old_file.exists(), f"first-save archive should exist: {old_file}"

    messages_v2 = [
        Message(role="user", content="Version 2", timestamp=None),
        Message(role="assistant", content="Response", timestamp=None),
    ]
    async with user_scoped_connection(app_pool, user_id=tenant_a) as conn:
        result = await conv_repo.save_conversation(
            conn,
            cipher,
            conversations_json_dir=tmp_path,
            topic="superseded-test",
            title="Same Title",
            messages=messages_v2,
            summary="V2",
        )

    assert (
        result.superseded_json_path is not None
    ), "re-save should return a superseded json_path to clean up"
    assert old_file.exists(), (
        "save_conversation must NOT delete the previous archive itself -- "
        "rollback would leave the row pointing at a deleted file"
    )
    conv_repo.delete_superseded_json_archive(tmp_path, result.superseded_json_path)

    assert not old_file.exists(), f"superseded archive should be deleted: {old_file}"

    async with admin_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT json_path FROM conversations WHERE slug = $1",
            "same-title",
        )
    assert row is not None
    new_json_path = str(row["json_path"])
    new_file = tmp_path / Path(new_json_path).name
    assert new_file.exists(), f"new archive should exist: {new_file}"
    assert old_json_path != new_json_path, "json_path should have changed"


class _Rollback(Exception):
    """Sentinel to abort a transaction after probing intra-transaction state."""

    def __init__(self, superseded: str | None, file_present: bool) -> None:
        super().__init__()
        self.superseded = superseded
        self.file_present = file_present


async def test_old_archive_survives_transaction_rollback(
    app_pool,  #: asyncpg.Pool (RLS-enforced)
    admin_pool,  #: asyncpg.Pool (BYPASSRLS)
    cipher: ContentCipher,
    tmp_path: Path,
    tenant_a: UUID,
) -> None:
    """If the transaction wrapping a re-save rolls back, the previous
    archive file must still exist on disk -- otherwise the conversations
    row would revert to a ``json_path`` that points at a deleted file.

    Asserts that ``save_conversation`` does not delete the old archive
    itself; deletion is the caller's job and must happen only after the
    transaction commits.
    """
    async with user_scoped_connection(app_pool, user_id=tenant_a) as conn:
        await topic_repo.create(conn, topic="rollback-test", title="Rollback Test")

        first = await conv_repo.save_conversation(
            conn,
            cipher,
            conversations_json_dir=tmp_path,
            topic="rollback-test",
            title="Rollback Title",
            messages=[Message(role="user", content="V1", timestamp=None)],
            summary="V1",
        )
    assert first.superseded_json_path is None, "first save has nothing to clean up"

    async with admin_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT json_path FROM conversations WHERE slug = $1",
            "rollback-title",
        )
    assert row is not None
    persisted_json_path = str(row["json_path"])
    persisted_file = tmp_path / Path(persisted_json_path).name
    assert persisted_file.exists()

    async def _resave_then_rollback() -> None:
        async with user_scoped_connection(app_pool, user_id=tenant_a) as conn:
            result = await conv_repo.save_conversation(
                conn,
                cipher,
                conversations_json_dir=tmp_path,
                topic="rollback-test",
                title="Rollback Title",
                messages=[Message(role="user", content="V2", timestamp=None)],
                summary="V2",
            )
            file_still_present = persisted_file.exists()
            raise _Rollback(result.superseded_json_path, file_still_present)

    with pytest.raises(_Rollback) as excinfo:
        await _resave_then_rollback()
    assert excinfo.value.superseded == persisted_json_path
    assert excinfo.value.file_present, "old file must still exist while txn is open"

    async with admin_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT json_path FROM conversations WHERE slug = $1",
            "rollback-title",
        )
    assert row is not None
    assert str(row["json_path"]) == persisted_json_path, "rollback should revert json_path"
    assert persisted_file.exists(), (
        "old archive file MUST survive a rolled-back re-save -- "
        "the row points at it after rollback"
    )
