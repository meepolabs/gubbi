"""Shared test fixtures.

PostgreSQL tests require a running PostgreSQL instance.
Set TEST_DATABASE_URL to point at it, or run:

    docker run -d --name journalctl-test-pg \\
        -e POSTGRES_DB=journal_test \\
        -e POSTGRES_USER=journal \\
        -e POSTGRES_PASSWORD=testpass \\
        -p 5433:5432 \\
        pgvector/pgvector:pg17

Then run: pytest tests/
"""

import os
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import asyncpg
import bcrypt
import pytest
import pytest_asyncio

from journalctl.config import get_settings
from journalctl.oauth.storage import OAuthStorage
from journalctl.storage.pg_setup import _init_connection, setup_schema

TEST_PASSWORD = "test-password"
TEST_PASSWORD_HASH = bcrypt.hashpw(TEST_PASSWORD.encode(), bcrypt.gensalt()).decode()

_DEFAULT_TEST_DB = "postgresql://journal:testpass@localhost:5433/journal_test"
TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL", _DEFAULT_TEST_DB)


@pytest.fixture
def tmp_journal(tmp_path: Path) -> Path:
    """Create a temporary journal directory structure."""
    (tmp_path / "knowledge").mkdir()
    (tmp_path / "conversations_json").mkdir()
    return tmp_path


_TEST_ENV: dict[str, str] = {
    "JOURNAL_API_KEY": "test-api-key-for-unit-tests-only",  # must be >= 32 chars
    "JOURNAL_TRANSPORT": "stdio",
    "JOURNAL_SERVER_URL": "http://localhost:8100",
    "JOURNAL_OAUTH_ACCESS_TOKEN_TTL": "3600",
    "JOURNAL_OAUTH_REFRESH_TOKEN_TTL": "2592000",
    "JOURNAL_OAUTH_AUTH_CODE_TTL": "300",
    "JOURNAL_DATABASE_URL": TEST_DATABASE_URL,
}


@pytest.fixture(autouse=True)
def _set_env(tmp_journal: Path, tmp_path: Path) -> Iterator[None]:
    """Set environment variables for tests and restore them on teardown."""
    env = {
        **_TEST_ENV,
        "JOURNAL_JOURNAL_ROOT": str(tmp_journal),
        "JOURNAL_OWNER_PASSWORD_HASH": TEST_PASSWORD_HASH,
        "JOURNAL_OAUTH_DB_PATH": str(tmp_path / "oauth.db"),
    }
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    get_settings.cache_clear()
    yield
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    get_settings.cache_clear()


@pytest_asyncio.fixture(scope="session")
async def pool() -> AsyncIterator[asyncpg.Pool]:
    """asyncpg Pool connected to the test PostgreSQL database.

    Session-scoped: one pool for the entire test run.
    Skips the test session if PostgreSQL is not reachable.

    Start a test database with:
        docker run -d --name journalctl-test-pg \\
            -e POSTGRES_DB=journal_test -e POSTGRES_USER=journal \\
            -e POSTGRES_PASSWORD=testpass -p 5433:5432 \\
            pgvector/pgvector:pg17
    """
    try:
        _pool: asyncpg.Pool = await asyncpg.create_pool(
            TEST_DATABASE_URL,
            statement_cache_size=0,
            min_size=1,
            max_size=5,
            timeout=5,  # fast fail if PG not reachable
            init=_init_connection,
        )
    except (
        asyncpg.InvalidCatalogNameError,
        OSError,
        asyncpg.CannotConnectNowError,
        ConnectionRefusedError,
        TimeoutError,
        Exception,
    ) as exc:
        pytest.skip(f"PostgreSQL not reachable at {TEST_DATABASE_URL}: {exc}")
        return

    async with _pool.acquire() as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    await setup_schema(_pool)
    yield _pool
    await _pool.close()


@pytest_asyncio.fixture
async def clean_pool(pool: asyncpg.Pool) -> AsyncIterator[asyncpg.Pool]:
    """Yield the shared pool; TRUNCATE all tables before and after each test for isolation."""
    async with pool.acquire() as conn:
        await conn.execute(
            "TRUNCATE topics, entries, conversations, messages, entry_embeddings"
            " RESTART IDENTITY CASCADE"
        )
    yield pool
    async with pool.acquire() as conn:
        await conn.execute(
            "TRUNCATE topics, entries, conversations, messages, entry_embeddings"
            " RESTART IDENTITY CASCADE"
        )


@pytest.fixture
def oauth_storage(tmp_path: Path) -> Iterator[OAuthStorage]:
    """OAuthStorage with a temp database."""
    db = OAuthStorage(tmp_path / "oauth.db")
    _ = db.conn  # Force schema init
    yield db
    db.close()
