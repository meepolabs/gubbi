"""Test that read() isolates DecryptionError into [decryption-failed] sentinel.

Item 2 from quick-fixes plan: a single corrupt row in read() (the
get_by_date_range path) must not poison the entire range. Mirror the
get_texts sentinel pattern at the per-entry level.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gubbi.crypto.cipher import DecryptionError
from gubbi.models.journal import TopicMeta
from gubbi.storage.repositories import entries as entry_repo


@pytest.mark.unit
@pytest.mark.asyncio
async def test_read_returns_sentinel_on_per_entry_decryption_failure() -> None:
    """A row whose decrypt raises returns Entry with [decryption-failed] content.

    Two rows: row 1 decrypts cleanly, row 2 raises DecryptionError. Both
    must come back; row 2 must carry the sentinel content and a None
    reasoning, not bubble the error up.
    """
    topic_meta = TopicMeta(
        id=7, topic="work/acme", title="Acme", created="2026-05-01", updated="2026-05-01"
    )

    rows = [
        {
            "id": 1,
            "date": "2026-05-01",
            "content_encrypted": b"ok",
            "content_nonce": b"n1",
            "reasoning_encrypted": None,
            "reasoning_nonce": None,
            "conversation_id": None,
            "tags": ["fine"],
            "total_count": 2,
        },
        {
            "id": 2,
            "date": "2026-05-02",
            "content_encrypted": b"corrupt",
            "content_nonce": b"n2",
            "reasoning_encrypted": None,
            "reasoning_nonce": None,
            "conversation_id": None,
            "tags": ["broken"],
            "total_count": 2,
        },
    ]

    conn = MagicMock()
    conn.is_in_transaction = MagicMock(return_value=True)
    conn.fetch = AsyncMock(return_value=rows)
    cipher = MagicMock()

    def fake_decrypt(_cipher: Any, row: Any, *_args: str) -> str | None:
        if row["id"] == 2:
            raise DecryptionError("simulated cipher failure")
        return "ok-content"

    with (
        patch.object(entry_repo, "get_topic", new=AsyncMock(return_value=topic_meta)),
        patch.object(entry_repo, "_decrypt_content_field", side_effect=fake_decrypt),
    ):
        meta, entries, total = await entry_repo.read(conn, cipher, "work/acme", limit=10)

    assert meta.topic == "work/acme"
    assert total == 2
    assert len(entries) == 2
    by_id = {e.id: e for e in entries}
    assert by_id[1].content == "ok-content"
    assert by_id[2].content == "[decryption-failed]"
    assert by_id[2].reasoning is None
