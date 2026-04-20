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
import subprocess
import sys
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import asyncpg
import bcrypt
import pytest
import pytest_asyncio

from journalctl.config import get_settings
from journalctl.core.crypto import ContentCipher
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


@pytest.fixture
def cipher() -> ContentCipher:
    """Deterministic test cipher with a fixed key so test data is reproducible.

    NEVER use this key outside tests -- it is intentionally weak (all 0x01
    bytes) and must not end up in any .env, Doppler secret, or production
    config. Production keys come from ``JOURNAL_ENCRYPTION_MASTER_KEY_V*``.
    """
    return ContentCipher({1: bytes([1]) * 32})


# ---------------------------------------------------------------------------
# RLS test-database fixtures (TASK-02.15)
# ---------------------------------------------------------------------------
#
# The legacy `pool` fixture above speaks to a schema-only test database
# (`journal_test`) and is shared by every pre-multi-tenant test. RLS tests
# need a SEPARATE database where migrations 0001..0005 are applied, because
# migration 0005 uses FORCE ROW LEVEL SECURITY — that would cause existing
# tests to see zero rows under the legacy pool, breaking the whole suite.
#
# Layout (all three DSNs share host+port, differ by user+password+database):
#   TEST_DATABASE_URL_RLS         -> journal:testpass  (privileged bootstrap)
#   TEST_DATABASE_URL_RLS_APP     -> journal_app:...   (runtime, RLS enforced)
#   TEST_DATABASE_URL_RLS_ADMIN   -> journal_admin:... (BYPASSRLS, seed only)
#
# The `_rls_provisioned` session fixture creates the DB if missing, runs
# alembic upgrade head against it (as the privileged bootstrap user), and
# sets test-only passwords on the journal_app / journal_admin roles so the
# derived pool fixtures can authenticate.
# ---------------------------------------------------------------------------

_DEFAULT_RLS_DB = "postgresql://journal:testpass@localhost:5433/journal_rls_test"
RLS_BOOTSTRAP_URL = os.environ.get("TEST_DATABASE_URL_RLS", _DEFAULT_RLS_DB)

_RLS_APP_PASSWORD = "testpass_app"  # noqa: S105 — test-only credential
_RLS_ADMIN_PASSWORD = "testpass_admin"  # noqa: S105 — test-only credential
_RLS_FOUNDER_EMAIL = "founder@test.local"


def _rewrite_dsn(dsn: str, *, user: str, password: str) -> str:
    """Return ``dsn`` with its userinfo replaced. Preserves scheme/host/port/db."""
    parsed = urlparse(dsn)
    # ``urlparse`` puts "user:pass@host:port" into ``parsed.netloc``. Rebuild it.
    host = parsed.hostname or "localhost"
    port_part = f":{parsed.port}" if parsed.port else ""
    new_netloc = f"{user}:{password}@{host}{port_part}"
    return urlunparse(parsed._replace(netloc=new_netloc))


def _postgres_db_dsn(dsn: str) -> str:
    """Return ``dsn`` pointing at the ``postgres`` maintenance DB for CREATE DATABASE."""
    parsed = urlparse(dsn)
    return urlunparse(parsed._replace(path="/postgres"))


def _db_name(dsn: str) -> str:
    return urlparse(dsn).path.lstrip("/") or "postgres"


async def _ensure_database_exists(dsn: str) -> None:
    """CREATE DATABASE <name> if it does not already exist. Caller must have CREATEDB."""
    maintenance_dsn = _postgres_db_dsn(dsn)
    target_db = _db_name(dsn)
    # Escape embedded double-quotes in the identifier — mirrors the pattern
    # used by migration 0002's _quoted_db_name(). Defense-in-depth even though
    # target_db is sourced from an env-configured DSN rather than user input.
    safe_name = target_db.replace('"', '""')
    conn = await asyncpg.connect(maintenance_dsn, timeout=5)
    try:
        exists = await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", target_db)
        if not exists:
            # CREATE DATABASE cannot run in a transaction and does not accept params.
            await conn.execute(f'CREATE DATABASE "{safe_name}"')
    finally:
        await conn.close()


def _run_alembic_upgrade(bootstrap_dsn: str) -> None:
    """Run ``alembic upgrade head`` against ``bootstrap_dsn`` as a subprocess.

    Subprocess keeps alembic's global state out of the pytest session's process
    and isolates JOURNAL_* env overrides to just the migration run.
    """
    project_root = Path(__file__).resolve().parents[1]
    env = {
        **os.environ,
        "JOURNAL_DATABASE_URL": bootstrap_dsn,
        "JOURNAL_FOUNDER_EMAIL": _RLS_FOUNDER_EMAIL,
    }
    result = subprocess.run(  # noqa: S603 — args are a hard-coded list, no shell
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(
            "alembic upgrade head failed for RLS test DB:\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


async def _set_role_passwords(bootstrap_dsn: str) -> None:
    """Assign test-only passwords to journal_app and journal_admin so derived pools can log in.

    ``ALTER ROLE ... PASSWORD`` is a PostgreSQL utility statement and cannot be
    parameterized. f-string interpolation is safe here ONLY because both passwords
    are hard-coded module-level constants containing no single-quotes — the
    guard below locks that invariant so a future contributor who switches to
    env-sourced passwords is forced to introduce proper escaping. Using ``raise``
    (not ``assert``) so ``python -O`` cannot strip the check.
    """
    if "'" in _RLS_APP_PASSWORD or "'" in _RLS_ADMIN_PASSWORD:
        raise ValueError(
            "RLS test passwords must not contain single-quotes — "
            "see _set_role_passwords docstring for the interpolation-safety contract"
        )
    conn = await asyncpg.connect(bootstrap_dsn, timeout=5)
    try:
        await conn.execute(f"ALTER ROLE journal_app WITH PASSWORD '{_RLS_APP_PASSWORD}'")
        await conn.execute(f"ALTER ROLE journal_admin WITH PASSWORD '{_RLS_ADMIN_PASSWORD}'")
    finally:
        await conn.close()


@pytest_asyncio.fixture(scope="session")
async def _rls_provisioned() -> None:
    """One-shot session setup: ensure DB exists, migrations applied, roles have passwords.

    Demanded purely for its side effects — consumers depend on it so the pool
    fixtures see a migrated DB with logins ready. On any failure the fixture
    calls ``pytest.skip``, which aborts the demanding test (and any other tests
    that depend on this fixture transitively) with a clear reason.

    Recovery note: alembic upgrade is idempotent, so a partial provisioning in
    one session (e.g. upgrade succeeded but _set_role_passwords failed) recovers
    automatically on the next session — the upgrade no-ops and password setting
    runs fresh. Do NOT add an early-exit before _set_role_passwords.
    """
    try:
        await _ensure_database_exists(RLS_BOOTSTRAP_URL)
    except (OSError, asyncpg.PostgresError, TimeoutError) as exc:
        pytest.skip(f"Cannot provision RLS test DB at {RLS_BOOTSTRAP_URL}: {exc}")

    # Ensure pgvector is installed in the target DB before migrations reference vectors.
    conn = await asyncpg.connect(RLS_BOOTSTRAP_URL, timeout=5)
    try:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    finally:
        await conn.close()

    _run_alembic_upgrade(RLS_BOOTSTRAP_URL)
    await _set_role_passwords(RLS_BOOTSTRAP_URL)
    await _reassign_ownership_to_admin(RLS_BOOTSTRAP_URL)


async def _reassign_ownership_to_admin(bootstrap_dsn: str) -> None:
    """Transfer ownership of every public-schema object to journal_admin.

    In production, alembic runs under journal_admin, so tables + sequences
    are created owned by that role. In this test harness, alembic bootstraps
    as the docker default (``journal``), leaving ownership with ``journal``.
    TRUNCATE RESTART IDENTITY inside ``clean_rls_db`` needs sequence
    ownership (not just GRANT), so we re-point ownership to journal_admin
    after migrations land.

    ``REASSIGN OWNED BY journal`` is rejected because the bootstrap role
    also owns the database itself. A targeted DO-block loop over
    ``pg_tables`` + ``pg_sequences`` + ``pg_views`` scoped to the public
    schema sidesteps that.
    """
    conn = await asyncpg.connect(bootstrap_dsn, timeout=5)
    try:
        await conn.execute(
            """
            DO $$
            DECLARE
                r record;
            BEGIN
                FOR r IN SELECT tablename AS name FROM pg_tables WHERE schemaname = 'public'
                LOOP
                    EXECUTE format('ALTER TABLE public.%I OWNER TO journal_admin', r.name);
                END LOOP;
                FOR r IN SELECT sequence_name AS name
                         FROM information_schema.sequences WHERE sequence_schema = 'public'
                LOOP
                    EXECUTE format('ALTER SEQUENCE public.%I OWNER TO journal_admin', r.name);
                END LOOP;
                FOR r IN SELECT viewname AS name FROM pg_views WHERE schemaname = 'public'
                LOOP
                    EXECUTE format('ALTER VIEW public.%I OWNER TO journal_admin', r.name);
                END LOOP;
            END $$;
            """
        )
    finally:
        await conn.close()


def _rls_derived_dsn(role: str, password: str) -> str:
    return _rewrite_dsn(RLS_BOOTSTRAP_URL, user=role, password=password)


RLS_APP_URL = _rls_derived_dsn("journal_app", _RLS_APP_PASSWORD)
RLS_ADMIN_URL = _rls_derived_dsn("journal_admin", _RLS_ADMIN_PASSWORD)


@pytest_asyncio.fixture(scope="session")
async def app_pool(_rls_provisioned: None) -> AsyncIterator[asyncpg.Pool]:
    """asyncpg pool authenticated as ``journal_app`` (RLS-enforced, NO BYPASSRLS).

    Use this pool with ``core.db_context.user_scoped_connection`` for RLS
    assertions. Direct ``pool.acquire()`` without the helper will see zero
    rows — that's intentional default-deny.
    """
    _pool = await asyncpg.create_pool(
        RLS_APP_URL,
        statement_cache_size=0,
        min_size=1,
        max_size=5,
        timeout=5,
        init=_init_connection,
    )
    try:
        yield _pool
    finally:
        await _pool.close()


@pytest_asyncio.fixture(scope="session")
async def admin_pool(_rls_provisioned: None) -> AsyncIterator[asyncpg.Pool]:
    """asyncpg pool authenticated as ``journal_admin`` (BYPASSRLS).

    Used only to seed cross-tenant test data and to tear down between tests.
    Must NOT be used for RLS assertions — BYPASSRLS defeats the policy check.
    """
    _pool = await asyncpg.create_pool(
        RLS_ADMIN_URL,
        statement_cache_size=0,
        min_size=1,
        max_size=5,
        timeout=5,
        init=_init_connection,
    )
    try:
        yield _pool
    finally:
        await _pool.close()


@pytest_asyncio.fixture
async def clean_rls_db(admin_pool: asyncpg.Pool) -> AsyncIterator[asyncpg.Pool]:
    """Yield admin_pool; TRUNCATE tenant tables + users before and after each test.

    RESTART IDENTITY resets serial PKs so assertions about entry_id stay stable.
    CASCADE walks FKs to clean up dependent rows.
    """
    truncate_sql = (
        "TRUNCATE topics, entries, conversations, messages, entry_embeddings, users "
        "RESTART IDENTITY CASCADE"
    )
    async with admin_pool.acquire() as conn:
        await conn.execute(truncate_sql)
    yield admin_pool
    async with admin_pool.acquire() as conn:
        await conn.execute(truncate_sql)
