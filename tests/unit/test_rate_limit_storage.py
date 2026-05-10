"""Unit tests for RateLimitStorage.

The schema for `rate_limit_events` is owned by `OAuthStorage._init_schema`,
so each test seeds the schema by touching `OAuthStorage(db_path).initialize()` once,
then exercises `RateLimitStorage` directly against the same db file.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest_asyncio

from gubbi.oauth._rate_limit import RateLimitStorage
from gubbi.oauth.storage import OAuthStorage


@pytest_asyncio.fixture
async def db_path(tmp_path: Path) -> Path:
    """Initialize the OAuth db schema and return the db path.

    `OAuthStorage` creates the `rate_limit_events` table via its
    `_init_schema`. We then close it so `RateLimitStorage` can open its own
    connection against the same file -- mirroring how the two classes
    coexist in production.
    """
    p = tmp_path / "oauth.db"
    schema_owner = OAuthStorage(p)
    await schema_owner.initialize()  # force schema init
    await schema_owner.close()
    return p


@pytest_asyncio.fixture
async def rl(db_path: Path) -> RateLimitStorage:
    """Return a fresh RateLimitStorage for each test; close on teardown."""
    storage = RateLimitStorage(db_path)
    yield storage
    await storage.aclose()


class TestRecordEvent:
    async def test_record_event_increments_count(self, rl: RateLimitStorage) -> None:
        # Arrange / Act
        for _ in range(3):
            await rl.record_event("login_failure:1.2.3.4")

        # Assert
        assert await rl.count_events("login_failure:1.2.3.4", 60) == 3

    async def test_record_event_isolates_keys(self, rl: RateLimitStorage) -> None:
        await rl.record_event("k1")
        await rl.record_event("k1")
        await rl.record_event("k2")

        assert await rl.count_events("k1", 60) == 2
        assert await rl.count_events("k2", 60) == 1


class TestCountEventsWindow:
    async def test_count_excludes_events_outside_window(
        self, rl: RateLimitStorage, db_path: Path
    ) -> None:
        # Insert one stale event directly via raw SQL, then a fresh one.
        # Using a separate connection mirrors the cross-worker scenario.
        import sqlite3

        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(
                "INSERT INTO rate_limit_events (event_key, occurred_at) VALUES (?, ?)",
                ("k", int(time.time()) - 3600),
            )
            conn.commit()

        await rl.record_event("k")  # recent

        # Window of 60s only catches the recent one; window of 7200s catches both.
        assert await rl.count_events("k", 60) == 1
        assert await rl.count_events("k", 7200) == 2

    async def test_count_returns_zero_for_unknown_key(self, rl: RateLimitStorage) -> None:
        assert await rl.count_events("nope", 60) == 0


class TestPruneRetention:
    async def test_prune_deletes_events_older_than_retention(
        self, rl: RateLimitStorage, db_path: Path
    ) -> None:
        import sqlite3

        # Seed an old row + a fresh row.
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(
                "INSERT INTO rate_limit_events (event_key, occurred_at) VALUES (?, ?)",
                ("k", int(time.time()) - 10_000),
            )
            conn.commit()
        await rl.record_event("k")

        deleted = await rl.prune(3600)

        assert deleted == 1
        assert await rl.count_events("k", 60) == 1

    async def test_prune_returns_zero_when_nothing_old(self, rl: RateLimitStorage) -> None:
        await rl.record_event("k")
        assert await rl.prune(3600) == 0
        assert await rl.count_events("k", 60) == 1


class TestConcurrentAccess:
    async def test_concurrent_record_events_do_not_lose_writes(self, rl: RateLimitStorage) -> None:
        """All writes from concurrent coroutines must land; lock prevents lost updates."""
        import asyncio

        # Arrange
        coroutine_count = 8
        per_coroutine = 25
        expected = coroutine_count * per_coroutine

        async def worker() -> None:
            for _ in range(per_coroutine):
                await rl.record_event("contended_key")

        # Act
        await asyncio.gather(*[worker() for _ in range(coroutine_count)])

        # Assert
        assert await rl.count_events("contended_key", 600) == expected

    async def test_lock_is_independent_of_oauth_storage_lock(self, db_path: Path) -> None:
        """RateLimitStorage owns its own lock; sharing is forbidden."""
        oauth = OAuthStorage(db_path)
        try:
            assert oauth._rl._lock is not oauth._lock
        finally:
            await oauth.close()


class TestMultiInstanceContention:
    async def test_multi_instance_concurrent_writes_share_storage(self, db_path: Path) -> None:
        """Two RateLimitStorage instances on the same db_path must serialize via
        SQLite WAL + busy_timeout, not corrupt counts.

        The single-instance contention test above exercises the in-process
        asyncio.Lock. This test exercises the cross-instance / cross-
        connection path: each instance has its own lock + connection, so
        only SQLite-level locking (busy_timeout retries under WAL) keeps the
        writes consistent.
        """
        import asyncio

        # Arrange: two independent storages against the same SQLite file.
        rl_a = RateLimitStorage(db_path)
        rl_b = RateLimitStorage(db_path)

        coroutines_per_instance = 4
        per_coroutine = 25
        total_coroutines = coroutines_per_instance * 2
        expected = total_coroutines * per_coroutine  # 8 * 25 = 200

        async def worker(storage: RateLimitStorage) -> None:
            for _ in range(per_coroutine):
                await storage.record_event("multi_instance_key")

        tasks = [
            asyncio.create_task(worker(s))
            for s in ([rl_a] * coroutines_per_instance + [rl_b] * coroutines_per_instance)
        ]

        # Act
        try:
            await asyncio.gather(*tasks)

            # Assert: every write landed; SQLite WAL + busy_timeout serialized
            # the writers without losing any rows.
            assert await rl_a.count_events("multi_instance_key", 600) == expected
            # rl_b reads its own connection -- confirms the result is visible
            # across both connections, not cached on the writer.
            assert await rl_b.count_events("multi_instance_key", 600) == expected
        finally:
            await rl_a.aclose()
            await rl_b.aclose()
