"""Integration tests for conversation upsert TOCTOU fix (9.2) and orphan cleanup (9.3).

Fix 9.2: ``_upsert_conversation_record`` no longer runs a pre-check SELECT.
Instead it derives ``is_update`` from ``(xmax != 0) AS was_update`` in the
RETURNING clause — race-free single round-trip.

Fix 9.3: On re-save the old JSON archive file is deleted after the DB upsert
succeeds, preventing orphan file accumulation on disk.
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest
from gubbi_common.db.user_scoped import user_scoped_connection

from journalctl.core.crypto import ContentCipher
from journalctl.models.conversation import Message
from journalctl.storage.repositories import conversations as conv_repo
from journalctl.storage.repositories import topics as topic_repo

pytestmark = pytest.mark.asyncio(loop_scope="session")


async def test_upsert_is_update_tracking(
    app_pool,  #: asyncpg.Pool (RLS-enforced)
    cipher: ContentCipher,
    tmp_path: Path,
    tenant_a: UUID,
) -> None:
    """Verify that ``_upsert_conversation_record`` returns ``is_update``
    correctly for both insert and update paths using the ``xmax`` trick."""
    async with user_scoped_connection(app_pool, user_id=tenant_a) as conn:
        await topic_repo.create(conn, topic="upsert-test", title="Upsert Test")

        messages = [
            Message(role="user", content="Hello", timestamp=None),
            Message(role="assistant", content="Hi there", timestamp=None),
        ]

        # First save — should be an insert (is_update=False).
        _conv_id, _summary, is_update, _linked_id = await conv_repo.save_conversation(
            conn,
            cipher,
            conversations_json_dir=tmp_path,
            topic="upsert-test",
            title="Same Chat",
            messages=messages,
            summary="First summary",
        )
        assert is_update is False, "first save should report is_update=False"

        # Second save with same topic+title — should be an update (is_update=True).
        messages_v2 = messages + [
            Message(role="user", content="Follow up", timestamp=None),
            Message(role="assistant", content="Sure thing", timestamp=None),
        ]
        _conv_id, _summary, is_update, _linked_id = await conv_repo.save_conversation(
            conn,
            cipher,
            conversations_json_dir=tmp_path,
            topic="upsert-test",
            title="Same Chat",
            messages=messages_v2,
            summary="Updated summary",
        )
        assert is_update is True, "re-save should report is_update=True"


async def test_orphan_json_is_deleted_on_resave(
    app_pool,  #: asyncpg.Pool (RLS-enforced)
    admin_pool,  #: asyncpg.Pool (BYPASSRLS)
    cipher: ContentCipher,
    tmp_path: Path,
    tenant_a: UUID,
) -> None:
    """Verify that re-saving a conversation deletes the old JSON archive file."""
    async with user_scoped_connection(app_pool, user_id=tenant_a) as conn:
        await topic_repo.create(conn, topic="orphan-test", title="Orphan Test")

        messages = [
            Message(role="user", content="Version 1", timestamp=None),
        ]

        # First save.
        _conv_id, _summary, _is_update, _linked_id = await conv_repo.save_conversation(
            conn,
            cipher,
            conversations_json_dir=tmp_path,
            topic="orphan-test",
            title="Same Title",
            messages=messages,
            summary="V1",
        )

    # Read back the json_path from admin pool.
    async with admin_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT json_path FROM conversations WHERE slug = $1",
            "same-title",  # slugified title
        )
    assert row is not None
    old_json_path = str(row["json_path"])
    old_file = tmp_path / Path(old_json_path).name
    assert old_file.exists(), f"old JSON archive should exist at first save: {old_file}"

    # Second save — same topic+title triggers update.
    messages_v2 = [
        Message(role="user", content="Version 2", timestamp=None),
        Message(role="assistant", content="Response", timestamp=None),
    ]
    async with user_scoped_connection(app_pool, user_id=tenant_a) as conn:
        _conv_id, _summary, _is_update, _linked_id = await conv_repo.save_conversation(
            conn,
            cipher,
            conversations_json_dir=tmp_path,
            topic="orphan-test",
            title="Same Title",
            messages=messages_v2,
            summary="V2",
        )

    # Old file should be gone; new file should exist.
    assert not old_file.exists(), f"old JSON archive should be deleted after resave: {old_file}"

    # The new file should exist.
    async with admin_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT json_path FROM conversations WHERE slug = $1",
            "same-title",
        )
    assert row is not None
    new_json_path = str(row["json_path"])
    new_file = tmp_path / Path(new_json_path).name
    assert new_file.exists(), f"new JSON archive should exist after resave: {new_file}"
    assert old_json_path != new_json_path, "json_path should have changed after resave"
