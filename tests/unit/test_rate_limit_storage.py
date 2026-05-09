"""Unit tests for RateLimitStorage.

The schema for `rate_limit_events` is owned by `OAuthStorage._init_schema`,
so each test seeds the schema by touching `OAuthStorage(db_path).conn` once,
then exercises `RateLimitStorage` directly against the same db file.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from gubbi.oauth._rate_limit import RateLimitStorage
from gubbi.oauth.storage import OAuthStorage


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Initialize the OAuth db schema and return the db path.

    `OAuthStorage` creates the `rate_limit_events` table via its
    `_init_schema`. We then close it so `RateLimitStorage` can open its own
    connection against the same file -- mirroring how the two classes
    coexist in production.
    """
    p = tmp_path / "oauth.db"
    schema_owner = OAuthStorage(p)
    _ = schema_owner.conn  # force schema init
    schema_owner.close()
    return p


@pytest.fixture
def rl(db_path: Path) -> Iterator[RateLimitStorage]:
    storage = RateLimitStorage(db_path)
    yield storage
    storage.close()


class TestRecordEvent:
    def test_record_event_increments_count(self, rl: RateLimitStorage) -> None:
        # Arrange / Act
        for _ in range(3):
            rl.record_event("login_failure:1.2.3.4")

        # Assert
        assert rl.count_events("login_failure:1.2.3.4", 60) == 3

    def test_record_event_isolates_keys(self, rl: RateLimitStorage) -> None:
        rl.record_event("k1")
        rl.record_event("k1")
        rl.record_event("k2")

        assert rl.count_events("k1", 60) == 2
        assert rl.count_events("k2", 60) == 1


class TestCountEventsWindow:
    def test_count_excludes_events_outside_window(
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

        rl.record_event("k")  # recent

        # Window of 60s only catches the recent one; window of 7200s catches both.
        assert rl.count_events("k", 60) == 1
        assert rl.count_events("k", 7200) == 2

    def test_count_returns_zero_for_unknown_key(self, rl: RateLimitStorage) -> None:
        assert rl.count_events("nope", 60) == 0


class TestPruneRetention:
    def test_prune_deletes_events_older_than_retention(
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
        rl.record_event("k")

        deleted = rl.prune(3600)

        assert deleted == 1
        assert rl.count_events("k", 60) == 1

    def test_prune_returns_zero_when_nothing_old(self, rl: RateLimitStorage) -> None:
        rl.record_event("k")
        assert rl.prune(3600) == 0
        assert rl.count_events("k", 60) == 1


class TestThreadingLock:
    def test_concurrent_record_events_do_not_lose_writes(self, rl: RateLimitStorage) -> None:
        """All writes from concurrent threads must land; lock prevents lost updates."""
        # Arrange
        thread_count = 8
        per_thread = 25
        expected = thread_count * per_thread

        def worker() -> None:
            for _ in range(per_thread):
                rl.record_event("contended_key")

        threads = [threading.Thread(target=worker) for _ in range(thread_count)]

        # Act
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Assert
        assert rl.count_events("contended_key", 600) == expected

    def test_lock_is_independent_of_oauth_storage_lock(self, db_path: Path) -> None:
        """RateLimitStorage owns its own lock; sharing is forbidden."""
        oauth = OAuthStorage(db_path)
        try:
            assert oauth._rl._lock is not oauth._lock
        finally:
            oauth.close()


class TestMultiInstanceContention:
    def test_multi_instance_concurrent_writes_share_storage(self, db_path: Path) -> None:
        """Two RateLimitStorage instances on the same db_path must serialize via
        SQLite WAL + busy_timeout, not corrupt counts.

        The single-instance contention test above exercises the in-process
        `threading.Lock`. This test exercises the cross-instance / cross-
        connection path: each instance has its own lock + connection, so
        only SQLite-level locking (busy_timeout retries under WAL) keeps the
        writes consistent.
        """
        # Arrange: two independent storages against the same SQLite file.
        rl_a = RateLimitStorage(db_path)
        rl_b = RateLimitStorage(db_path)

        thread_count_per_instance = 4
        per_thread = 25
        total_threads = thread_count_per_instance * 2
        expected = total_threads * per_thread  # 8 * 25 = 200

        barrier = threading.Barrier(total_threads)
        errors: list[BaseException] = []
        errors_lock = threading.Lock()

        def worker(storage: RateLimitStorage) -> None:
            try:
                # Synchronize the start so all threads contend at once.
                barrier.wait(timeout=10)
                for _ in range(per_thread):
                    storage.record_event("multi_instance_key")
            except BaseException as e:  # noqa: BLE001 - re-raised after join
                with errors_lock:
                    errors.append(e)

        threads: list[threading.Thread] = []
        for storage in (rl_a, rl_b):
            for _ in range(thread_count_per_instance):
                threads.append(threading.Thread(target=worker, args=(storage,)))

        # Act
        try:
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # Surface any worker exception (e.g. SQLite "database is locked"
            # would mean busy_timeout is too short for this contention level).
            assert not errors, f"worker errors: {errors!r}"

            # Assert: every write landed; SQLite WAL + busy_timeout serialized
            # the writers without losing any rows.
            assert rl_a.count_events("multi_instance_key", 600) == expected
            # rl_b reads its own connection — confirms the result is visible
            # across both connections, not cached on the writer.
            assert rl_b.count_events("multi_instance_key", 600) == expected
        finally:
            rl_a.close()
            rl_b.close()
