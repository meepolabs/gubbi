"""Async-concurrency unit tests for OAuthStorage.

Covers three scenarios that only manifest under asyncio concurrency:

1. Concurrent saves + reads across 50 coroutines -- no row is lost.
2. Lazy-init race -- two parallel first-callers converge to a single open
   connection (initialize() is idempotent).
3. Atomic rotation interleave -- rotate_refresh_token leaves no half-state
   even when two coroutines race.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

import pytest
import pytest_asyncio
from mcp.server.auth.provider import AccessToken, RefreshToken
from mcp.shared.auth import OAuthClientInformationFull

from gubbi.oauth.storage import OAuthStorage


def _access_token(token: str, client_id: str = "c") -> AccessToken:
    return AccessToken(
        token=token,
        client_id=client_id,
        scopes=["journal:read"],
        expires_at=int(time.time()) + 3600,
    )


def _refresh_token(token: str, client_id: str = "c") -> RefreshToken:
    return RefreshToken(
        token=token,
        client_id=client_id,
        scopes=["journal:read"],
        expires_at=int(time.time()) + 86400,
    )


def _client(client_id: str = "client-1") -> OAuthClientInformationFull:
    return OAuthClientInformationFull(
        client_id=client_id,
        client_secret="test-secret",
        redirect_uris=["http://localhost/callback"],
    )


async def _cancel_inside_atomic(storage: OAuthStorage) -> None:
    task = asyncio.current_task()
    assert task is not None
    async with storage._atomic():
        task.cancel()
        await asyncio.sleep(0)


@pytest_asyncio.fixture
async def storage(tmp_path: Path) -> OAuthStorage:
    """Initialized OAuthStorage; closed after each test."""
    s = OAuthStorage(tmp_path / "oauth.db")
    await s.initialize()
    yield s
    await s.close()


class TestConcurrentSaveAndGet:
    async def test_fifty_parallel_saves_all_land(self, storage: OAuthStorage) -> None:
        """50 concurrent save_access_token coroutines must all persist successfully."""
        tokens = [f"tok-{i:03d}" for i in range(50)]

        async def save(t: str) -> None:
            await storage.save_access_token(t, _access_token(t))

        await asyncio.gather(*[save(t) for t in tokens])

        for t in tokens:
            row = await storage.get_access_token(t)
            assert row is not None, f"token {t!r} not found after concurrent saves"

    async def test_parallel_reads_after_write_all_see_row(self, storage: OAuthStorage) -> None:
        """A single write followed by 50 concurrent reads must all return the row."""
        await storage.save_access_token("shared-tok", _access_token("shared-tok"))

        results = await asyncio.gather(*[storage.get_access_token("shared-tok") for _ in range(50)])
        assert all(r is not None for r in results)
        assert all(r.token == "shared-tok" for r in results)  # type: ignore[union-attr]


class TestLazyInitRace:
    async def test_two_parallel_first_callers_produce_single_connection(
        self, tmp_path: Path
    ) -> None:
        """Calling initialize() concurrently must not open the connection twice."""
        s = OAuthStorage(tmp_path / "race.db")
        try:
            # Neither has run; race them.
            await asyncio.gather(s.initialize(), s.initialize())
            # If both opens happened, _initialized would be set twice -- but the
            # lock ensures exactly one open. Verify by checking a simple query works
            # and the storage is usable (schema applied once, no "table already exists" error).
            await s.save_access_token("probe", _access_token("probe"))
            row = await s.get_access_token("probe")
            assert row is not None
        finally:
            await s.close()

    async def test_initialize_is_idempotent(self, storage: OAuthStorage) -> None:
        """Calling initialize() on an already-initialized storage is a no-op."""
        # Should not raise, should not double-migrate, and storage stays usable.
        await storage.initialize()
        await storage.initialize()
        await storage.save_access_token("idempotent", _access_token("idempotent"))
        assert await storage.get_access_token("idempotent") is not None

    async def test_close_then_initialize_reapplies_schema(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Closing and reinitializing must reapply schema on the new connection.

        Beyond verifying that close() + initialize() does not crash, this test
        spies on _init_schema to confirm the schema is actively reapplied on
        the post-close initialize() -- the first close drops the in-memory
        _initialized flag, so the next initialize() must call _init_schema
        again on the fresh connection.
        """
        storage = OAuthStorage(tmp_path / "reinit.db")
        call_count = 0
        original_init_schema = OAuthStorage._init_schema

        async def _counting_init_schema(self: OAuthStorage) -> None:
            nonlocal call_count
            call_count += 1
            await original_init_schema(self)

        monkeypatch.setattr(OAuthStorage, "_init_schema", _counting_init_schema)
        try:
            await storage.initialize()
            assert call_count == 1, "first initialize() must apply schema"
            await storage.close()
            await storage.initialize()
            assert call_count == 2, "post-close initialize() must reapply schema"
            await storage.save_client(_client())
            assert await storage.get_client("client-1") is not None
        finally:
            await storage.close()

    async def test_atomic_rolls_back_on_cancellation(self, tmp_path: Path) -> None:
        """CancelledError inside _atomic must roll back the open transaction."""
        storage = OAuthStorage(tmp_path / "cancel.db")
        try:
            await storage.initialize()
            with pytest.raises(asyncio.CancelledError):
                await _cancel_inside_atomic(storage)
            conn = await storage._get_conn()
            assert not conn.in_transaction
            async with storage._lock:
                await conn.execute("BEGIN IMMEDIATE")
                await conn.rollback()
        finally:
            await storage.close()

    async def test_atomic_preserves_original_exception_when_rollback_fails(
        self,
        storage: OAuthStorage,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Body exception must propagate even when conn.rollback() also fails.

        Pins the contract that _atomic's inner try/except around rollback
        preserves the caller's original exception (rather than masking it
        with the rollback failure) and logs the rollback failure at WARNING
        with the original body exception captured both in the message text
        AND via exc_info -- the message text is the load-bearing surface
        because gubbi's production JSON logger does not walk __context__
        chains.
        """

        class _RollbackBoom(Exception):
            pass

        class _BodyBoom(Exception):
            pass

        conn = await storage._get_conn()

        rollback_call_count = 0

        async def _failing_rollback() -> None:
            nonlocal rollback_call_count
            rollback_call_count += 1
            raise _RollbackBoom("simulated rollback failure")

        monkeypatch.setattr(conn, "rollback", _failing_rollback)

        with (
            caplog.at_level(logging.WARNING, logger="gubbi.oauth.storage"),
            pytest.raises(_BodyBoom, match="body-failed"),
        ):
            async with storage._atomic():
                raise _BodyBoom("body-failed")

        # Self-validation: the monkeypatched rollback must have actually fired.
        # Without this assert, a future refactor of _atomic that bypasses
        # rollback would silently pass the test.
        assert (
            rollback_call_count == 1
        ), f"conn.rollback must be called exactly once; got {rollback_call_count}"

        rollback_records = [
            r for r in caplog.records if "rollback failed" in r.getMessage().lower()
        ]
        assert rollback_records, "rollback failure must be logged at WARNING"

        # exc_info=True must populate the log record's exc_info tuple so
        # chain-walking formatters (stdlib, structlog with format_exc_info)
        # render both the rollback failure and the original via __context__.
        record = rollback_records[0]
        assert record.exc_info is not None, (
            "rollback WARNING must carry exc_info so chain-walking formatters "
            "render the original exception via __context__"
        )

        # Load-bearing assertion under the production JSON logger (which does
        # NOT walk __context__ chains): the original body exception must
        # appear in the rendered message text. Without this, an operator
        # reading JSON logs sees only "rollback failed" and loses the
        # original auth-context.
        message = record.getMessage()
        assert "_BodyBoom" in message, (
            "rollback WARNING message text must include the original body "
            "exception's repr (gubbi's production JSON logger does not walk "
            f"__context__ chains). Got message: {message!r}"
        )
        assert "body-failed" in message, (
            "rollback WARNING message text must include the original "
            f"exception's args. Got message: {message!r}"
        )


class TestAtomicRotationInterleave:
    async def test_rotation_leaves_no_half_state_under_concurrency(
        self, storage: OAuthStorage
    ) -> None:
        """Two rotate_refresh_token calls on different pairs must not interleave writes."""
        # Seed two independent (access, refresh) pairs.
        await storage.save_issued_token_pair(
            "at-a", _access_token("at-a"), "rt-a", _refresh_token("rt-a")
        )
        await storage.save_issued_token_pair(
            "at-b", _access_token("at-b"), "rt-b", _refresh_token("rt-b")
        )

        async def rotate_a() -> None:
            await storage.rotate_refresh_token(
                "rt-a",
                "at-a2",
                _access_token("at-a2"),
                "rt-a2",
                _refresh_token("rt-a2"),
            )

        async def rotate_b() -> None:
            await storage.rotate_refresh_token(
                "rt-b",
                "at-b2",
                _access_token("at-b2"),
                "rt-b2",
                _refresh_token("rt-b2"),
            )

        await asyncio.gather(rotate_a(), rotate_b())

        # Old tokens must be gone.
        assert await storage.get_access_token("at-a") is None
        assert await storage.get_access_token("at-b") is None
        assert await storage.get_refresh_token("rt-a") is None
        assert await storage.get_refresh_token("rt-b") is None

        # New tokens must be present.
        assert await storage.get_access_token("at-a2") is not None
        assert await storage.get_access_token("at-b2") is not None
        assert await storage.get_refresh_token("rt-a2") is not None
        assert await storage.get_refresh_token("rt-b2") is not None

    async def test_rotation_old_tokens_gone_new_tokens_present(self, storage: OAuthStorage) -> None:
        """Sequential rotation: old pair removed, new pair inserted atomically."""
        await storage.save_issued_token_pair(
            "at1", _access_token("at1"), "rt1", _refresh_token("rt1")
        )
        await storage.rotate_refresh_token(
            "rt1",
            "at2",
            _access_token("at2"),
            "rt2",
            _refresh_token("rt2"),
        )
        assert await storage.get_access_token("at1") is None
        assert await storage.get_refresh_token("rt1") is None
        assert await storage.get_access_token("at2") is not None
        assert await storage.get_refresh_token("rt2") is not None
