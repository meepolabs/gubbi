"""Verify _atomic does not shadow the body exception when rollback itself fails.

Regression guard for the m-h12 LOW gap: when an exception inside the _atomic
context manager body triggers a rollback that ALSO fails, the original body
exception (the one the caller actually wants to see) must be the one that
propagates -- not the rollback failure. The rollback failure is logged as a
warning but otherwise dropped.

This shape is easy to get subtly wrong: an unguarded `await conn.rollback()`
inside the `except` block would shadow the body's exception with the rollback
exception when both fire; a bare `raise rollback_err` would do the same. The
implementation guards the rollback in its own try/except and re-raises the
outer (body) exception with a bare `raise`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from gubbi.oauth.storage import OAuthStorage


class _FailingRollbackConn:
    """Stand-in aiosqlite.Connection where rollback() raises.

    Only the methods _atomic touches need to exist; everything else can stay
    undefined to keep the test focused.
    """

    def __init__(self) -> None:
        self.executed: list[str] = []
        self.committed = False
        self.rollback_attempts = 0

    async def execute(self, sql: str, *args: Any, **kwargs: Any) -> Any:
        self.executed.append(sql)

    async def commit(self) -> None:
        self.committed = True

    async def rollback(self) -> None:
        self.rollback_attempts += 1
        raise RuntimeError("rollback-failed")


class _BodyError(Exception):
    """Distinct marker class so the assertion is unambiguous."""


@pytest.mark.asyncio
async def test_atomic_does_not_shadow_body_error_when_rollback_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Body exception must propagate; rollback failure must be logged, not shadowing."""
    storage = OAuthStorage(tmp_path / "shadow.db")
    fake_conn = _FailingRollbackConn()

    async def _fake_get_conn(self: OAuthStorage) -> Any:
        # Bypass real aiosqlite open + the per-instance lock the real
        # _get_conn uses; _atomic still acquires self._lock separately.
        return fake_conn

    monkeypatch.setattr(OAuthStorage, "_get_conn", _fake_get_conn)

    with (
        caplog.at_level(logging.WARNING, logger="gubbi.oauth.storage"),
        pytest.raises(_BodyError, match="body-failed"),
    ):
        async with storage._atomic():
            raise _BodyError("body-failed")

    # Rollback was attempted (and failed), but the BodyError is what propagated.
    assert fake_conn.rollback_attempts == 1, "rollback path must run on body error"
    assert not fake_conn.committed, "commit must not run on body error"

    # The rollback failure was recorded -- not silently dropped.
    rollback_records = [
        rec for rec in caplog.records if "rollback failed" in rec.getMessage().lower()
    ]
    assert rollback_records, "rollback failure must be logged at WARNING"
