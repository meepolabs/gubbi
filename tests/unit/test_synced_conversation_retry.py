"""Retry policy of ``merge_synced_conversation`` on a unique violation."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, create_autospec, patch

import asyncpg
import pytest

from gubbi.crypto.cipher import ContentCipher
from gubbi.storage.repositories import synced_conversations as module
from gubbi.storage.repositories.synced_conversations import (
    MergeResult,
    SyncedConversation,
    SyncedMessage,
    merge_synced_conversation,
)

_CONV = SyncedConversation(
    platform="claude",
    platform_id="thread-1",
    title="Thread",
    topic="inbox",
    created_at=datetime(2026, 10, 1, tzinfo=UTC),
    messages=(SyncedMessage(role="user", content="hi", message_id="k1"),),
)
_OK = MergeResult(1, True, 1, 0, 0, 0, legacy_identity_conflict=False)


def _conn() -> MagicMock:
    conn = create_autospec(asyncpg.Connection, instance=True)
    conn.is_in_transaction.return_value = True
    savepoint = MagicMock()
    savepoint.__aenter__ = AsyncMock(return_value=None)
    savepoint.__aexit__ = AsyncMock(return_value=False)
    conn.transaction.return_value = savepoint
    return conn


async def test_one_unique_violation_is_retried_in_a_fresh_savepoint() -> None:
    conn = _conn()
    attempt = AsyncMock(side_effect=[asyncpg.UniqueViolationError("duplicate key"), _OK])

    with patch.object(module, "_merge_once", attempt):
        result = await merge_synced_conversation(conn, MagicMock(spec=ContentCipher), _CONV)

    assert result is _OK
    assert attempt.await_count == 2
    assert conn.transaction.call_count == 2


async def test_a_second_unique_violation_propagates() -> None:
    conn = _conn()
    attempt = AsyncMock(
        side_effect=[
            asyncpg.UniqueViolationError("first"),
            asyncpg.UniqueViolationError("second"),
        ]
    )

    with (
        patch.object(module, "_merge_once", attempt),
        pytest.raises(asyncpg.UniqueViolationError, match="second"),
    ):
        await merge_synced_conversation(conn, MagicMock(spec=ContentCipher), _CONV)

    assert attempt.await_count == 2


async def test_a_call_outside_a_transaction_is_refused_before_any_work() -> None:
    conn = _conn()
    conn.is_in_transaction.return_value = False
    attempt = AsyncMock(return_value=_OK)

    with (
        patch.object(module, "_merge_once", attempt),
        pytest.raises(RuntimeError, match="transaction"),
    ):
        await merge_synced_conversation(conn, MagicMock(spec=ContentCipher), _CONV)

    assert attempt.await_count == 0
