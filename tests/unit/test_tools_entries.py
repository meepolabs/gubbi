"""Unit tests for `gubbi.tools.entries` tool-layer guards (A4 bundle).

Covers:
    - S6 H1: content + reasoning caps applied PRE-sanitization so the error
      reports the real input size, not the post-strip size.
    - S6 H2: all-None no-op update returns `validation_error("No fields to
      update")` and signals failure via `success=False`, which prevents the
      @audited decorator from writing a ghost audit row.
    - Append-mode empty content stays a distinct error (locked by A4 Q4).

Tests call `_journal_append_entry` and `_journal_update_entry` directly, not
through the registered MCP tool wrapper.  That bypasses the `@require_scope`
+ `@audited` decorator stack but exercises the tool-layer validation
boundaries we are trying to harden.  Audit-decorator skip behaviour for
`success=False` is already covered by ``test_audit_decorator.py``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

from gubbi.app_context import AppContext
from gubbi.audit import ACTION_ENTRY_UPDATED, audited
from gubbi.tools.constants import MAX_ENTRY_CONTENT_CHARS, MAX_ENTRY_REASONING_CHARS
from gubbi.tools.entries import _journal_append_entry, _journal_update_entry

pytestmark = pytest.mark.unit


def _make_app_ctx() -> MagicMock:
    """Build a minimal AppContext mock.

    The cap + no-op guards short-circuit before any DB call, so we do not
    need a real pool / cipher / embedding service.  We do need ``settings``
    on the mock chain because ``_journal_append_entry`` may resolve
    ``app_ctx.settings.timezone`` for the date default -- but only on the
    success path, which these tests do not exercise.
    """
    ctx = MagicMock()
    ctx.settings.timezone = "UTC"
    return ctx


# ---------------------------------------------------------------------------
# S6 H1 -- entry content + reasoning caps (append path)
# ---------------------------------------------------------------------------


async def test_append_entry_rejects_oversized_content() -> None:
    """content longer than MAX_ENTRY_CONTENT_CHARS is rejected pre-sanitization."""
    app_ctx = _make_app_ctx()
    oversized = "x" * (MAX_ENTRY_CONTENT_CHARS + 1)

    result = await _journal_append_entry(app_ctx, topic="work", content=oversized)

    assert result["error_code"] == "VALIDATION_ERROR"
    assert result["success"] is False
    assert str(MAX_ENTRY_CONTENT_CHARS) in result["error"]
    assert "content exceeds" in result["error"]


async def test_append_entry_accepts_max_length_content() -> None:
    """content of exactly MAX_ENTRY_CONTENT_CHARS is accepted (boundary).

    The cap is a *strict* greater-than check, so the exact-cap input must
    not be rejected by the cap guard.  We intercept the DB call so the
    test does not require Postgres.
    """
    app_ctx = _make_app_ctx()
    at_cap = "x" * MAX_ENTRY_CONTENT_CHARS

    # If the cap rejected the input, we would never reach the connection
    # context -- the test asserts via the path taken.  Either
    # MissingUserIdError or our marker exception means the cap did NOT
    # short-circuit -- which is the behaviour we want to confirm.  Both
    # indicate the cap guard let the request flow past validation.
    with (
        patch("gubbi.tools.entries.user_scoped_connection") as mock_conn_ctx,
        patch("gubbi.tools.entries.current_user_id") as mock_user_id,
        patch("gubbi.tools.entries.require_cipher"),
    ):
        mock_conn_ctx.side_effect = RuntimeError("reached DB layer")
        mock_user_id.get.return_value = None  # tool will raise MissingUserIdError
        with pytest.raises(Exception) as exc_info:  # noqa: BLE001 - marker check below
            await _journal_append_entry(app_ctx, topic="work", content=at_cap)
    # Sanity: this MUST not be a validation_error -- if the cap guard fired,
    # we would have gotten a dict back, not an exception.
    assert "exceeds" not in str(exc_info.value)


async def test_append_entry_rejects_oversized_reasoning() -> None:
    """reasoning longer than MAX_ENTRY_REASONING_CHARS is rejected pre-sanitization."""
    app_ctx = _make_app_ctx()
    oversized = "r" * (MAX_ENTRY_REASONING_CHARS + 1)

    result = await _journal_append_entry(app_ctx, topic="work", content="ok", reasoning=oversized)

    assert result["error_code"] == "VALIDATION_ERROR"
    assert result["success"] is False
    assert str(MAX_ENTRY_REASONING_CHARS) in result["error"]
    assert "reasoning exceeds" in result["error"]


async def test_append_entry_cap_uses_pre_sanitization_length() -> None:
    """Reported size is real input size, not post-strip size.

    Control characters get stripped by ``sanitize_freetext``.  If the cap
    were applied post-sanitization, an attacker could send a giant string
    of NUL bytes that shrinks to zero after stripping and bypass the cap
    entirely.  Pre-sanitization length check closes that gap.
    """
    app_ctx = _make_app_ctx()
    # All NULs get stripped; post-sanitization length is 0.
    # Pre-sanitization length is MAX_ENTRY_CONTENT_CHARS + 1.
    oversized = "\x00" * (MAX_ENTRY_CONTENT_CHARS + 1)

    result = await _journal_append_entry(app_ctx, topic="work", content=oversized)

    assert result["error_code"] == "VALIDATION_ERROR"
    assert "content exceeds" in result["error"]


# ---------------------------------------------------------------------------
# S6 H1 -- entry content + reasoning caps (update path)
# ---------------------------------------------------------------------------


async def test_update_entry_rejects_oversized_content() -> None:
    """content longer than MAX_ENTRY_CONTENT_CHARS is rejected pre-sanitization."""
    app_ctx = _make_app_ctx()
    oversized = "x" * (MAX_ENTRY_CONTENT_CHARS + 1)

    result = await _journal_update_entry(app_ctx, entry_id=1, content=oversized)

    assert result["error_code"] == "VALIDATION_ERROR"
    assert result["success"] is False
    assert str(MAX_ENTRY_CONTENT_CHARS) in result["error"]
    assert "content exceeds" in result["error"]


async def test_update_entry_rejects_oversized_reasoning() -> None:
    """reasoning longer than MAX_ENTRY_REASONING_CHARS is rejected pre-sanitization."""
    app_ctx = _make_app_ctx()
    oversized = "r" * (MAX_ENTRY_REASONING_CHARS + 1)

    result = await _journal_update_entry(app_ctx, entry_id=1, reasoning=oversized)

    assert result["error_code"] == "VALIDATION_ERROR"
    assert result["success"] is False
    assert str(MAX_ENTRY_REASONING_CHARS) in result["error"]
    assert "reasoning exceeds" in result["error"]


# ---------------------------------------------------------------------------
# S6 H2 -- no-op update early-exit (prevents ghost audit row)
# ---------------------------------------------------------------------------


async def test_update_entry_noop_returns_validation_error() -> None:
    """All-None update returns the locked house-style error message."""
    app_ctx = _make_app_ctx()

    result = await _journal_update_entry(app_ctx, entry_id=1)

    assert result["error_code"] == "VALIDATION_ERROR"
    assert result["error"] == "No fields to update"
    assert result["success"] is False


async def test_update_entry_noop_does_not_touch_db() -> None:
    """No-op guard must short-circuit before opening any connection.

    Asserts via patch that the ``user_scoped_connection`` helper is never
    invoked -- the tool returns the validation_error envelope without
    issuing any UPDATE.  This together with @audited's
    ``success=False`` short-circuit gives the "no ghost audit row"
    contract claimed by A4 Q2.
    """
    app_ctx = _make_app_ctx()

    with patch("gubbi.tools.entries.user_scoped_connection") as mock_conn_ctx:
        result = await _journal_update_entry(app_ctx, entry_id=1)

    assert result["error_code"] == "VALIDATION_ERROR"
    assert result["error"] == "No fields to update"
    mock_conn_ctx.assert_not_called()


async def test_update_entry_noop_success_false_suppresses_audit() -> None:
    """validation_error must carry success=False so @audited skips the write.

    The audit decorator's ``_result_is_success`` checks
    ``result.get("success", True)`` -- if validation_error omits the key,
    the no-op result is treated as a success and a ghost audit row gets
    written with target_id=None.  This test guards the contract that ties
    the no-op guard (tool layer) to the audit suppression (decorator
    layer).
    """
    from gubbi.audit.decorator import _result_is_success

    app_ctx = _make_app_ctx()
    result = await _journal_update_entry(app_ctx, entry_id=1)

    assert _result_is_success(result) is False


# ---------------------------------------------------------------------------
# A4 Q4 -- append-mode empty content stays a distinct error
# ---------------------------------------------------------------------------


async def test_update_entry_empty_content_is_distinct_from_noop() -> None:
    """mode='append', content='' returns the original empty-content error,
    not the new no-op message.  Locked by A4 Q4."""
    app_ctx = _make_app_ctx()

    result = await _journal_update_entry(app_ctx, entry_id=1, content="", mode="append")

    assert result["error_code"] == "VALIDATION_ERROR"
    assert result["error"] == "content cannot be empty"
    assert result["success"] is False


# ---------------------------------------------------------------------------
# A4 R2 M-1 -- empty-after-sanitize reasoning is normalized to noop
# ---------------------------------------------------------------------------
#
# Pre-fix bug: ``reasoning=""`` (or ``"\x00"`` or whitespace-only) passed the
# all-None no-op guard because the value is not None.  Sanitization reduced
# it to ``""`` and ``entry_repo.update`` treated ``reasoning is not None``
# as "reasoning changed" -- encrypting the empty string, including
# ``reasoning_encrypted`` + ``reasoning_nonce`` in the SET clause, setting
# ``indexed_at = NULL`` (triggering reindex), and writing an audit row.
# Two-stage validation closes this: stage (1) sanitizes + normalizes empty
# reasoning to None; stage (2) re-runs the no-op guard so the call returns
# ``validation_error("No fields to update")`` before any DB / audit work.


async def test_update_entry_empty_reasoning_normalized_to_noop() -> None:
    """reasoning='' is empty after sanitization -> caught by no-op guard."""
    app_ctx = _make_app_ctx()

    with patch("gubbi.tools.entries.user_scoped_connection") as mock_conn_ctx:
        result = await _journal_update_entry(app_ctx, entry_id=1, reasoning="")

    assert result["error_code"] == "VALIDATION_ERROR"
    assert result["error"] == "No fields to update"
    assert result["success"] is False
    # Critical: no DB connection opened, so no reindex, no audit row.
    mock_conn_ctx.assert_not_called()


async def test_update_entry_whitespace_only_reasoning_normalized_to_noop() -> None:
    """reasoning='   ' (whitespace) survives sanitize but fails .strip() ->
    normalized to None -> caught by no-op guard."""
    app_ctx = _make_app_ctx()

    with patch("gubbi.tools.entries.user_scoped_connection") as mock_conn_ctx:
        result = await _journal_update_entry(app_ctx, entry_id=1, reasoning="   ")

    assert result["error_code"] == "VALIDATION_ERROR"
    assert result["error"] == "No fields to update"
    assert result["success"] is False
    mock_conn_ctx.assert_not_called()


async def test_update_entry_null_byte_reasoning_normalized_to_noop() -> None:
    """reasoning='\\x00' sanitizes to empty -> normalized to None -> noop.

    ``sanitize_freetext`` strips control chars including NUL; the post-sanitize
    string is empty.  The bug was that ``reasoning != None`` flowed through to
    ``entry_repo.update`` anyway -- this test pins the fix at the tool boundary.
    """
    app_ctx = _make_app_ctx()

    with patch("gubbi.tools.entries.user_scoped_connection") as mock_conn_ctx:
        result = await _journal_update_entry(app_ctx, entry_id=1, reasoning="\x00")

    assert result["error_code"] == "VALIDATION_ERROR"
    assert result["error"] == "No fields to update"
    assert result["success"] is False
    mock_conn_ctx.assert_not_called()


async def test_update_entry_real_reasoning_still_passes_noop_guard() -> None:
    """Sanity: a non-empty reasoning is NOT normalized to None.

    Without this, an over-eager normalization would silently drop legitimate
    reasoning updates.  The path past the no-op guard reaches the DB layer;
    we intercept the connection and assert by exception type / no validation
    error.
    """
    app_ctx = _make_app_ctx()

    with (
        patch("gubbi.tools.entries.user_scoped_connection") as mock_conn_ctx,
        patch("gubbi.tools.entries.current_user_id") as mock_user_id,
        patch("gubbi.tools.entries.require_cipher"),
    ):
        mock_conn_ctx.side_effect = RuntimeError("reached DB layer")
        mock_user_id.get.return_value = None
        try:
            result: Any = await _journal_update_entry(
                app_ctx, entry_id=1, reasoning="real reasoning text"
            )
        except Exception:  # noqa: BLE001 - any exception means the guard let it through
            result = None
    if isinstance(result, dict):
        assert result.get("error") != "No fields to update"


# ---------------------------------------------------------------------------
# Sanity: a regular partial update is NOT caught by the no-op guard
# ---------------------------------------------------------------------------


async def test_update_entry_date_only_passes_noop_guard() -> None:
    """date-only update must NOT be treated as a no-op.

    The guard only short-circuits the all-None case.  date-only is a real
    partial update that proceeds to the DB layer (which we don't exercise
    here -- we just confirm the guard lets it through by checking the
    error is NOT the no-op message).
    """
    app_ctx = _make_app_ctx()

    # The path past the no-op guard either raises MissingUserIdError
    # (current_user_id is None) or our marker error.  Either way, the no-op
    # guard let it through.  We never want a validation_error("No fields to
    # update") dict here.
    with (
        patch("gubbi.tools.entries.user_scoped_connection") as mock_conn_ctx,
        patch("gubbi.tools.entries.current_user_id") as mock_user_id,
        patch("gubbi.tools.entries.require_cipher"),
    ):
        mock_conn_ctx.side_effect = RuntimeError("reached DB layer")
        mock_user_id.get.return_value = None
        try:
            result: Any = await _journal_update_entry(app_ctx, entry_id=1, date="2026-01-01")
        except Exception:  # noqa: BLE001 - any exception means the guard let it through
            result = None
    if isinstance(result, dict):
        assert result.get("error") != "No fields to update"


# ---------------------------------------------------------------------------
# H-1 integration: no-op + @audited decorator -> no ghost audit row
# ---------------------------------------------------------------------------


_AUDIT_USER_ID = UUID("11111111-2222-3333-4444-555555555555")


def _make_audit_app_ctx() -> AppContext:
    """AppContext stub for the @audited decorator integration test."""
    ctx = MagicMock(spec=AppContext)
    ctx.pool = MagicMock()
    return ctx


@patch("gubbi.audit.decorator.current_user_id")
@patch("gubbi.audit.decorator.user_scoped_connection")
async def test_update_entry_noop_does_not_write_audit_row(
    mock_user_scoped_conn: MagicMock,
    mock_current_user_id: MagicMock,
) -> None:
    """End-to-end: all-None update wrapped by @audited writes NO audit row.

    Stacks the real ``@audited`` decorator around ``_journal_update_entry``
    (the same way the registered MCP tool stacks them in ``register()``).
    Calls with all-None inputs.  The tool returns a ``validation_error``
    envelope with ``success=False``, and the decorator's success heuristic
    (``_result_is_success``) must short-circuit the audit-write path so
    ``record_audit`` is never invoked.  This pins the contract between the
    tool-layer no-op guard and the decorator-layer skip behaviour.
    """
    # Arrange: authenticated user + a mock DB conn that we expect to never use.
    mock_current_user_id.get.return_value = _AUDIT_USER_ID
    mock_conn = AsyncMock()
    mock_user_scoped_conn.return_value.__aenter__.return_value = mock_conn

    tool_app_ctx = _make_app_ctx()
    audit_app_ctx = _make_audit_app_ctx()

    # Wrap the tool function with the real @audited decorator, mirroring the
    # production stack in ``register()``.  Bind ``app_ctx`` for the tool call.
    async def _bound_update(**kwargs: Any) -> dict[str, Any]:
        return await _journal_update_entry(tool_app_ctx, **kwargs)

    decorated = audited(
        ACTION_ENTRY_UPDATED,
        target_type="entry",
        target_kind="entry",
        app_ctx=audit_app_ctx,
    )(_bound_update)

    # Act: call with no fields to update.
    with patch("gubbi.audit.decorator.record_audit", new=AsyncMock()) as mock_record_audit:
        result = await decorated(entry_id=1)

    # Assert: tool returned the locked no-op envelope ...
    assert result["error_code"] == "VALIDATION_ERROR"
    assert result["error"] == "No fields to update"
    assert result["success"] is False
    # ... and the audit-write path was never reached.
    mock_record_audit.assert_not_called()
    mock_record_audit.assert_not_awaited()


# ---------------------------------------------------------------------------
# M-1 integration: date-only update keeps the entry out of get_unindexed
# ---------------------------------------------------------------------------


async def test_update_date_only_does_not_appear_in_get_unindexed() -> None:
    """Date-only update SQL must NOT set ``indexed_at = NULL``.

    ``get_unindexed`` selects rows where ``indexed_at IS NULL``.  If the
    update path nulled ``indexed_at`` on a date-only edit, the entry would
    re-enter the reindex queue even though its embedding is still valid.
    The dynamic-SET fix ensures date-only updates don't flip the column;
    this test pins the contract at the SQL boundary by asserting the
    column is absent from the SET clause AND that the same column is the
    filter used by ``get_unindexed`` -- so a date-only update cannot make
    the row visible to the reindex worker.
    """
    from datetime import date as _date

    from gubbi.storage.repositories import entries as entry_repo

    # Build a connection stub: fetchrow returns an existing row (so update
    # proceeds past the read), and execute records the SQL it was called with.
    row = {
        "id": 1,
        "content_encrypted": b"\x00" * 16,
        "content_nonce": b"\x00" * 12,
        "reasoning_encrypted": None,
        "reasoning_nonce": None,
        "topic_id": 99,
        "date": _date(2026, 1, 1),
        "tags": [],
    }
    conn = MagicMock()
    conn.is_in_transaction = MagicMock(return_value=True)
    conn.fetchrow = AsyncMock(return_value=row)
    conn.execute = AsyncMock()

    cipher = MagicMock()
    cipher.encrypt = MagicMock(return_value=(b"new-ct", b"new-nonce"))

    # Apply a date-only update.
    await entry_repo.update(conn, cipher, entry_id=1, date="2026-06-01")

    update_sql = conn.execute.await_args.args[0]

    # The update SQL must NOT touch indexed_at ...
    assert "indexed_at" not in update_sql, (
        "date-only update must not touch indexed_at; SQL was: " + update_sql
    )

    # ... and ``get_unindexed`` filters on ``indexed_at IS NULL``, so a row
    # whose indexed_at was NOT nulled cannot be selected by it.  We pin this
    # by inspecting the get_unindexed SQL source to confirm the filter shape.
    import inspect

    src = inspect.getsource(entry_repo.get_unindexed)
    assert "indexed_at IS NULL" in src, (
        "get_unindexed must filter on indexed_at IS NULL; otherwise the "
        "date-only contract is meaningless"
    )
