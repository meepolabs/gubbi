"""Unit tests for ``entries.update`` dynamic SET clause (A4 / S6 H4 + MEDIUM).

Pre-fix shape: ``UPDATE entries`` always wrote
``date, tags, updated_at, indexed_at=NULL, content_encrypted, content_nonce,
reasoning_encrypted, reasoning_nonce, search_vector``.  Side effects:

* ``indexed_at=NULL`` fired on every update, including date- or tag-only
  edits whose embedding is still valid -- the reindex worker would re-encode
  the same text on every harmless edit.
* ``cipher.encrypt(new_content)`` ran on every update, burning AES + WAL
  amplification even when content was unchanged.

Post-fix shape: SET clause built dynamically based on which inputs are
non-None.  date-/tags-only updates do NOT touch ciphertext columns and do
NOT null ``indexed_at``.  Content / reasoning changes still re-encrypt and
null ``indexed_at`` so the reindex worker picks them up.

These are unit tests that inspect the SQL string + parameters passed to
``conn.execute`` via an AsyncMock; the conftest ``pool`` fixture is not
required.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gubbi.storage.repositories import entries as entry_repo

pytestmark = pytest.mark.unit


def _make_conn_and_row() -> tuple[MagicMock, dict[str, Any]]:
    """Build a connection mock that returns a plausible existing-row dict.

    ``fetchrow`` returns a mapping that ``_decrypt_content_field`` can iterate
    safely (we never decrypt because none of these tests use ``mode=append``).
    ``execute`` is an AsyncMock so we can inspect the SQL + params it
    receives.
    """
    row = {
        "id": 1,
        "content_encrypted": b"\x00" * 16,
        "content_nonce": b"\x00" * 12,
        "reasoning_encrypted": None,
        "reasoning_nonce": None,
        "topic_id": 99,
        "date": date(2026, 1, 1),
        "tags": [],
    }
    conn = MagicMock()
    conn.is_in_transaction = MagicMock(return_value=True)
    conn.fetchrow = AsyncMock(return_value=row)
    conn.execute = AsyncMock()
    return conn, row


def _make_cipher() -> MagicMock:
    cipher = MagicMock()
    cipher.encrypt = MagicMock(return_value=(b"new-ct", b"new-nonce"))
    return cipher


# ---------------------------------------------------------------------------
# S6 H4 -- indexed_at = NULL only when content or reasoning changes
# ---------------------------------------------------------------------------


async def test_update_date_only_preserves_indexed_at() -> None:
    """date-only update must NOT issue ``indexed_at = NULL``."""
    conn, _row = _make_conn_and_row()
    cipher = _make_cipher()

    await entry_repo.update(conn, cipher, entry_id=1, date="2026-06-01")

    conn.execute.assert_awaited_once()
    sql = conn.execute.await_args.args[0]
    assert "indexed_at" not in sql, "date-only update must not touch indexed_at; SQL was: " + sql


async def test_update_tags_only_preserves_indexed_at() -> None:
    """tags-only update must NOT issue ``indexed_at = NULL``."""
    conn, _row = _make_conn_and_row()
    cipher = _make_cipher()

    await entry_repo.update(conn, cipher, entry_id=1, tags=["new"])

    conn.execute.assert_awaited_once()
    sql = conn.execute.await_args.args[0]
    assert "indexed_at" not in sql, "tags-only update must not touch indexed_at; SQL was: " + sql


async def test_update_content_nulls_indexed_at() -> None:
    """content change MUST issue ``indexed_at = NULL`` so the reindex worker
    picks the row up.  Regression guard for the original behaviour."""
    conn, _row = _make_conn_and_row()
    cipher = _make_cipher()

    await entry_repo.update(conn, cipher, entry_id=1, content="new text")

    # First execute is the UPDATE entries CTE; the second is the DELETE
    # FROM entry_embeddings (added when content/reasoning change to keep
    # the semantic-search side from surfacing stale-content matches).
    sql = conn.execute.await_args_list[0].args[0]
    assert "indexed_at = NULL" in sql, "content change must null indexed_at; SQL was: " + sql


async def test_update_reasoning_nulls_indexed_at() -> None:
    """reasoning change MUST issue ``indexed_at = NULL`` for the same reason."""
    conn, _row = _make_conn_and_row()
    cipher = _make_cipher()

    await entry_repo.update(conn, cipher, entry_id=1, reasoning="new reasoning")

    sql = conn.execute.await_args_list[0].args[0]
    assert "indexed_at = NULL" in sql, "reasoning change must null indexed_at; SQL was: " + sql


# ---------------------------------------------------------------------------
# S6 MEDIUM -- skip re-encrypt on tag/date-only updates
# ---------------------------------------------------------------------------


async def test_update_date_only_does_not_re_encrypt() -> None:
    """date-only update must NOT call cipher.encrypt -- nothing changed in
    the text, so re-encrypting just burns AES + WAL cycles."""
    conn, _row = _make_conn_and_row()
    cipher = _make_cipher()

    await entry_repo.update(conn, cipher, entry_id=1, date="2026-06-01")

    cipher.encrypt.assert_not_called()


async def test_update_tags_only_does_not_re_encrypt() -> None:
    """tags-only update must NOT call cipher.encrypt for the same reason."""
    conn, _row = _make_conn_and_row()
    cipher = _make_cipher()

    await entry_repo.update(conn, cipher, entry_id=1, tags=["new"])

    cipher.encrypt.assert_not_called()


async def test_update_content_re_encrypts() -> None:
    """Content change MUST re-encrypt and write ciphertext columns."""
    conn, _row = _make_conn_and_row()
    cipher = _make_cipher()

    await entry_repo.update(conn, cipher, entry_id=1, content="new text")

    cipher.encrypt.assert_called_once_with("new text")
    sql = conn.execute.await_args_list[0].args[0]
    assert "content_encrypted" in sql
    assert "content_nonce" in sql


async def test_update_reasoning_re_encrypts() -> None:
    """Reasoning change MUST re-encrypt and write the reasoning ciphertext columns."""
    conn, _row = _make_conn_and_row()
    cipher = _make_cipher()

    await entry_repo.update(conn, cipher, entry_id=1, reasoning="new reasoning")

    cipher.encrypt.assert_called_once_with("new reasoning")
    sql = conn.execute.await_args_list[0].args[0]
    assert "reasoning_encrypted" in sql
    assert "reasoning_nonce" in sql


# ---------------------------------------------------------------------------
# Dynamic SET clause -- only-the-changed-column writes
# ---------------------------------------------------------------------------


async def test_update_date_only_does_not_write_tags_column() -> None:
    """date-only update SET clause must not include the ``tags`` column."""
    conn, _row = _make_conn_and_row()
    cipher = _make_cipher()

    await entry_repo.update(conn, cipher, entry_id=1, date="2026-06-01")

    sql = conn.execute.await_args.args[0]
    # ``tags = `` would appear in the SET list if tags were being written.
    assert "tags =" not in sql, "date-only update must not write tags column; SQL was: " + sql
    assert "date =" in sql


async def test_update_tags_only_does_not_write_date_column() -> None:
    """tags-only update SET clause must not include the ``date`` column."""
    conn, _row = _make_conn_and_row()
    cipher = _make_cipher()

    await entry_repo.update(conn, cipher, entry_id=1, tags=["new"])

    sql = conn.execute.await_args.args[0]
    # ``date = `` would appear in the SET list if date were being written.
    # We allow ``date`` to appear elsewhere (e.g. in ``updated_at``) but the
    # ``date = `` assignment fragment must not be present.
    assert "date =" not in sql, "tags-only update must not write date column; SQL was: " + sql
    assert "tags =" in sql


async def test_update_always_writes_updated_at() -> None:
    """updated_at must always be written (mirrors topics.updated_at bump)."""
    conn, _row = _make_conn_and_row()
    cipher = _make_cipher()

    await entry_repo.update(conn, cipher, entry_id=1, tags=["new"])

    sql = conn.execute.await_args.args[0]
    assert "updated_at = $1" in sql
    # And the value bound to $1 must be a datetime.
    bound = conn.execute.await_args.args[1]
    assert isinstance(bound, datetime)
    assert bound.tzinfo == UTC


async def test_update_writes_topic_updated_at() -> None:
    """The CTE must propagate updated_at to the parent topic row."""
    conn, _row = _make_conn_and_row()
    cipher = _make_cipher()

    await entry_repo.update(conn, cipher, entry_id=1, tags=["new"])

    sql = conn.execute.await_args.args[0]
    assert "UPDATE topics SET updated_at = $1" in sql


# ---------------------------------------------------------------------------
# Append mode still decrypts existing content (needs concat)
# ---------------------------------------------------------------------------


async def test_update_content_append_mode_concats_old_and_new() -> None:
    """mode='append' must read the existing plaintext and concatenate.

    We supply a deterministic decrypt result via the cipher mock so the
    concatenation is observable in the value bound to the search_vector
    parameter.
    """
    conn, row = _make_conn_and_row()
    # cipher.encrypt should still produce some ciphertext.
    cipher = _make_cipher()
    # cipher.decrypt is invoked indirectly through ``decrypt_or_raise``;
    # the helper used in the repo path is ``_decrypt_content_field`` which
    # calls ``decrypt_or_raise(cipher, ct, nonce)``.  Patch ``decrypt_or_raise``
    # at module scope so we control the plaintext.
    with patch(
        "gubbi.storage.repositories.entries.decrypt_or_raise",
        return_value="old content",
    ):
        await entry_repo.update(conn, cipher, entry_id=1, content="addendum", mode="append")

    # The bound search_vector parameter (the plaintext passed to to_tsvector)
    # should be the concatenated text.
    sql = conn.execute.await_args_list[0].args[0]
    assert "to_tsvector" in sql
    bound_args = conn.execute.await_args_list[0].args[1:]
    assert any(
        isinstance(a, str) and "old content" in a and "addendum" in a for a in bound_args
    ), f"expected concatenated plaintext bound to to_tsvector; got args: {bound_args!r}"


# ---------------------------------------------------------------------------
# Invalid mode rejected
# ---------------------------------------------------------------------------


async def test_update_invalid_mode_raises_value_error() -> None:
    """Invalid mode for a content change must raise ValueError (unchanged
    contract, but the dynamic-SET refactor preserves it)."""
    conn, _row = _make_conn_and_row()
    cipher = _make_cipher()

    with pytest.raises(ValueError, match="Invalid mode"):
        await entry_repo.update(conn, cipher, entry_id=1, content="new text", mode="bogus")


# ---------------------------------------------------------------------------
# Stale embedding cleanup -- content/reasoning change must DELETE the existing
# entry_embeddings row inside the same transaction. Without this, journal_search's
# semantic side surfaces the row by old-content keywords because the inline
# re-embed in tools/entries.py runs in a separate transaction (best-effort) and
# can silently fail.
# ---------------------------------------------------------------------------


async def test_update_content_change_deletes_entry_embedding() -> None:
    """Content change MUST DELETE FROM entry_embeddings inside the same txn."""
    conn, _row = _make_conn_and_row()
    cipher = _make_cipher()

    await entry_repo.update(conn, cipher, entry_id=1, content="new text")

    # Two executes: the UPDATE entries CTE and the DELETE FROM entry_embeddings.
    assert (
        conn.execute.await_count == 2
    ), f"expected 2 executes (UPDATE + DELETE), got {conn.execute.await_count}"
    delete_sql = conn.execute.await_args_list[1].args[0]
    delete_args = conn.execute.await_args_list[1].args[1:]
    assert (
        "DELETE FROM entry_embeddings" in delete_sql
    ), f"second execute must DELETE the stale embedding row; SQL was: {delete_sql}"
    assert delete_args == (1,), f"DELETE must bind entry_id=1 only; got args: {delete_args!r}"


async def test_update_reasoning_change_deletes_entry_embedding() -> None:
    """Reasoning change MUST DELETE FROM entry_embeddings inside the same txn."""
    conn, _row = _make_conn_and_row()
    cipher = _make_cipher()

    await entry_repo.update(conn, cipher, entry_id=1, reasoning="new reasoning")

    assert (
        conn.execute.await_count == 2
    ), f"expected 2 executes (UPDATE + DELETE), got {conn.execute.await_count}"
    delete_sql = conn.execute.await_args_list[1].args[0]
    assert (
        "DELETE FROM entry_embeddings" in delete_sql
    ), f"second execute must DELETE the stale embedding row; SQL was: {delete_sql}"


async def test_update_date_only_does_not_delete_entry_embedding() -> None:
    """Date-only update must NOT touch entry_embeddings -- the stored vector
    still reflects the current text."""
    conn, _row = _make_conn_and_row()
    cipher = _make_cipher()

    await entry_repo.update(conn, cipher, entry_id=1, date="2026-06-01")

    assert conn.execute.await_count == 1, (
        f"date-only update must run a single execute (UPDATE); "
        f"got {conn.execute.await_count} -- did the embedding-delete fire on a "
        f"non-text edit?"
    )
    sql = conn.execute.await_args_list[0].args[0]
    assert "DELETE FROM entry_embeddings" not in sql


async def test_update_tags_only_does_not_delete_entry_embedding() -> None:
    """Tags-only update must NOT touch entry_embeddings -- same reason as date-only."""
    conn, _row = _make_conn_and_row()
    cipher = _make_cipher()

    await entry_repo.update(conn, cipher, entry_id=1, tags=["new"])

    assert conn.execute.await_count == 1, (
        f"tags-only update must run a single execute (UPDATE); " f"got {conn.execute.await_count}"
    )
    sql = conn.execute.await_args_list[0].args[0]
    assert "DELETE FROM entry_embeddings" not in sql
