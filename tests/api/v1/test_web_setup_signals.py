"""Tests for the user setup-signals endpoint.

``GET /api/v1/user/setup-signals`` -- onboarding booleans derived from the
user's journal data: ``{has_entries, has_synced_conversations}``.

Covers:
- Auth: missing token -> 401.
- Both-false on an empty journal.
- has_entries true after a non-deleted entry exists; soft-deleted entries do
  not count.
- has_synced_conversations true only for an extension/zip-sourced (ingested)
  conversation; a hand-saved conversation does not count.
- RLS isolation: user B's signals are unaffected by user A's data.

The DB-backed tests require the RLS test database. They auto-skip when the
database is unreachable (the shared pool fixtures call ``pytest.skip``), so
``pytest tests/api/v1/test_web_setup_signals.py`` exits PASS in environments
without a running Postgres. The 401 test needs no DB.
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

import asyncpg
import pytest
import pytest_asyncio
import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from httpx import ASGITransport, AsyncClient

from gubbi.api.v1.user import router as user_router
from gubbi.app_context import AppContext
from gubbi.auth.strategies import TrustGatewayStrategy
from gubbi.config import Settings
from gubbi.crypto.cipher import ContentCipher
from gubbi.storage.embedding_service import EmbeddingService

pytestmark = pytest.mark.asyncio(loop_scope="session")

API_PREFIX = "/api/v1"
ENDPOINT = f"{API_PREFIX}/user/setup-signals"

_USER_A = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
_USER_B = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")

# Minimal bytea payloads for NOT NULL encrypted columns. Nonces must be exactly
# 12 bytes to satisfy the ``*_nonce_len`` CHECK constraints; the ciphertext is
# never decrypted by this endpoint, so any non-empty bytes will do.
_CT = b"\x00" * 16
_NONCE = b"\x00" * 12


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app(pool: asyncpg.Pool, *, auth_user_id: UUID | None = None) -> FastAPI:
    """Minimal FastAPI with an AppContext backed by the given pool.

    ``auth_user_id`` sets operator_user_id so the X-Auth-User-Id gateway header
    authenticates as that user (matches the web/topics test harness).
    """
    settings = Settings(
        db={"app_url": ""},
        auth={
            "api_key": "test-api-key-for-unit-tests-only",
            "operator_email": "setup-signals-test@test.local",
            "trust_gateway": True,
        },
        server={"url": "http://localhost:8100"},
        data_dir=str(Path(__file__).parent),
    )
    cipher = ContentCipher({1: bytes([1]) * 32})
    app_ctx = AppContext(
        pool=pool,
        embedding_service=EmbeddingService(),
        settings=settings,
        logger=structlog.get_logger("test"),
        admin_pool=None,
        operator_user_id=auth_user_id,
        cipher=cipher,
    )
    app = FastAPI()
    app.state.app_ctx = app_ctx
    app.state.auth_strategies = [
        TrustGatewayStrategy(gateway_secret=None, gateway_require_signature=False),
    ]
    app.include_router(user_router, prefix=API_PREFIX)

    @app.exception_handler(Exception)
    async def _handler(request: Request, exc: Exception) -> JSONResponse:
        raise exc

    return app


async def _seed_user(conn: asyncpg.Connection, user_id: UUID, email: str) -> None:
    """INSERT user; ON CONFLICT DO NOTHING for idempotency."""
    await conn.execute(
        """
        INSERT INTO users (id, email, timezone, created_at, updated_at)
        VALUES ($1, $2, 'UTC', now(), now())
        ON CONFLICT (id) DO NOTHING
        """,
        user_id,
        email,
    )


async def _seed_topic(conn: asyncpg.Connection, user_id: UUID, path: str) -> int:
    """INSERT a topic row for the user; return topic_id."""
    topic_id = await conn.fetchval(
        """
        INSERT INTO topics (path, title, description, user_id, created_at, updated_at)
        VALUES ($1, $2, '', $3, now(), now())
        RETURNING id
        """,
        path,
        path.split("/")[-1].title(),
        user_id,
    )
    return int(topic_id)


async def _seed_entry(
    conn: asyncpg.Connection,
    user_id: UUID,
    topic_id: int,
    *,
    deleted: bool = False,
) -> None:
    """INSERT an entry row; ``deleted`` sets deleted_at for the soft-delete case."""
    await conn.execute(
        """
        INSERT INTO entries
            (topic_id, user_id, content_encrypted, content_nonce, deleted_at)
        VALUES ($1, $2, $3, $4, CASE WHEN $5 THEN now() ELSE NULL END)
        """,
        topic_id,
        user_id,
        _CT,
        _NONCE,
        deleted,
    )


async def _seed_conversation(
    conn: asyncpg.Connection,
    user_id: UUID,
    topic_id: int,
    slug: str,
    *,
    source: str,
    platform_id: str | None,
) -> None:
    """INSERT a conversation row with the given source / platform_id."""
    await conn.execute(
        """
        INSERT INTO conversations
            (topic_id, user_id, slug, source,
             title_encrypted, title_nonce, summary_encrypted, summary_nonce,
             platform_id, created_at, updated_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, now(), now())
        """,
        topic_id,
        user_id,
        slug,
        source,
        _CT,
        _NONCE,
        _CT,
        _NONCE,
        platform_id,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def user_a(admin_pool: asyncpg.Pool) -> UUID:
    """Ensure user A exists; tear down after test."""
    async with admin_pool.acquire() as conn:
        await _seed_user(conn, _USER_A, "setup-signals-a@test.local")
    yield _USER_A
    async with admin_pool.acquire() as conn:
        await conn.execute("DELETE FROM conversations WHERE user_id = $1", _USER_A)
        await conn.execute("DELETE FROM entries WHERE user_id = $1", _USER_A)
        await conn.execute("DELETE FROM topics WHERE user_id = $1", _USER_A)
        await conn.execute("DELETE FROM users WHERE id = $1", _USER_A)


@pytest_asyncio.fixture
async def user_b(admin_pool: asyncpg.Pool) -> UUID:
    """Ensure user B exists; tear down after test."""
    async with admin_pool.acquire() as conn:
        await _seed_user(conn, _USER_B, "setup-signals-b@test.local")
    yield _USER_B
    async with admin_pool.acquire() as conn:
        await conn.execute("DELETE FROM conversations WHERE user_id = $1", _USER_B)
        await conn.execute("DELETE FROM entries WHERE user_id = $1", _USER_B)
        await conn.execute("DELETE FROM topics WHERE user_id = $1", _USER_B)
        await conn.execute("DELETE FROM users WHERE id = $1", _USER_B)


@pytest_asyncio.fixture
async def client_a(
    app_pool: asyncpg.Pool,
    clean_rls_db: asyncpg.Pool,
    user_a: UUID,
) -> AsyncClient:
    """AsyncClient authenticated as user A (clean DB)."""
    app = _make_app(app_pool, auth_user_id=_USER_A)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest_asyncio.fixture
async def client_b(
    app_pool: asyncpg.Pool,
    clean_rls_db: asyncpg.Pool,
    user_b: UUID,
) -> AsyncClient:
    """AsyncClient authenticated as user B (shares user A's seeded data)."""
    app = _make_app(app_pool, auth_user_id=_USER_B)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ---------------------------------------------------------------------------
# Auth test (no DB needed -- auth runs before the handler touches the pool)
# ---------------------------------------------------------------------------


class TestSetupSignalsAuth:
    """Auth rejection (pre-handler)."""

    async def test_missing_token_returns_401(self, app_pool: asyncpg.Pool) -> None:
        """No credentials and trust_gateway=False -> 401."""
        settings = Settings(
            db={"app_url": ""},
            auth={
                "api_key": "test-api-key-for-unit-tests-only",
                "operator_email": "setup-signals-test@test.local",
                "trust_gateway": False,
            },
            server={"url": "http://localhost:8100"},
            data_dir=str(Path(__file__).parent),
        )
        cipher = ContentCipher({1: bytes([1]) * 32})
        app_ctx = AppContext(
            pool=app_pool,
            embedding_service=EmbeddingService(),
            settings=settings,
            logger=structlog.get_logger("test"),
            operator_user_id=None,
            cipher=cipher,
        )
        app = FastAPI()
        app.state.app_ctx = app_ctx
        app.state.auth_strategies = [
            TrustGatewayStrategy(gateway_secret=None, gateway_require_signature=False),
        ]
        app.include_router(user_router, prefix=API_PREFIX)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(ENDPOINT)
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# DB-backed tests (require the RLS test database)
# ---------------------------------------------------------------------------


class TestSetupSignals:
    """Signal derivation under RLS."""

    async def test_empty_journal_both_false(self, client_a: AsyncClient) -> None:
        """No entries and no conversations -> both signals false."""
        resp = await client_a.get(ENDPOINT, headers={"X-Auth-User-Id": str(_USER_A)})
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"has_entries": False, "has_synced_conversations": False}

    async def test_cache_control_private_no_store(self, client_a: AsyncClient) -> None:
        """Response carries Cache-Control: private, no-store."""
        resp = await client_a.get(ENDPOINT, headers={"X-Auth-User-Id": str(_USER_A)})
        assert resp.status_code == 200, resp.text
        cache_header = resp.headers.get("cache-control", "")
        assert "private" in cache_header
        assert "no-store" in cache_header

    async def test_has_entries_true_after_entry(
        self,
        client_a: AsyncClient,
        admin_pool: asyncpg.Pool,
        user_a: UUID,
    ) -> None:
        """A non-deleted entry flips has_entries true; synced stays false."""
        async with admin_pool.acquire() as conn:
            topic_id = await _seed_topic(conn, _USER_A, "work")
            await _seed_entry(conn, _USER_A, topic_id)

        resp = await client_a.get(ENDPOINT, headers={"X-Auth-User-Id": str(_USER_A)})
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"has_entries": True, "has_synced_conversations": False}

    async def test_soft_deleted_entry_does_not_count(
        self,
        client_a: AsyncClient,
        admin_pool: asyncpg.Pool,
        user_a: UUID,
    ) -> None:
        """A soft-deleted entry leaves has_entries false."""
        async with admin_pool.acquire() as conn:
            topic_id = await _seed_topic(conn, _USER_A, "work")
            await _seed_entry(conn, _USER_A, topic_id, deleted=True)

        resp = await client_a.get(ENDPOINT, headers={"X-Auth-User-Id": str(_USER_A)})
        assert resp.status_code == 200, resp.text
        assert resp.json()["has_entries"] is False

    async def test_synced_conversation_true_via_platform_id(
        self,
        client_a: AsyncClient,
        admin_pool: asyncpg.Pool,
        user_a: UUID,
    ) -> None:
        """An ingested conversation (platform_id set) flips synced true."""
        async with admin_pool.acquire() as conn:
            topic_id = await _seed_topic(conn, _USER_A, "inbox")
            await _seed_conversation(
                conn,
                _USER_A,
                topic_id,
                "synced-1",
                source="chatgpt",
                platform_id="chatgpt-abc123",
            )

        resp = await client_a.get(ENDPOINT, headers={"X-Auth-User-Id": str(_USER_A)})
        assert resp.status_code == 200, resp.text
        assert resp.json()["has_synced_conversations"] is True

    async def test_hand_saved_conversation_does_not_count(
        self,
        client_a: AsyncClient,
        admin_pool: asyncpg.Pool,
        user_a: UUID,
    ) -> None:
        """A hand-saved conversation (no platform_id, plain source) stays false."""
        async with admin_pool.acquire() as conn:
            topic_id = await _seed_topic(conn, _USER_A, "notes")
            await _seed_conversation(
                conn,
                _USER_A,
                topic_id,
                "manual-1",
                source="claude",
                platform_id=None,
            )

        resp = await client_a.get(ENDPOINT, headers={"X-Auth-User-Id": str(_USER_A)})
        assert resp.status_code == 200, resp.text
        assert resp.json()["has_synced_conversations"] is False


class TestSetupSignalsRLS:
    """RLS isolation: user B's signals are unaffected by user A's data."""

    async def test_user_b_unaffected_by_user_a_data(
        self,
        client_a: AsyncClient,
        client_b: AsyncClient,
        admin_pool: asyncpg.Pool,
        user_a: UUID,
        user_b: UUID,
    ) -> None:
        """User A has entries + a synced conversation; user B sees both false."""
        async with admin_pool.acquire() as conn:
            topic_id = await _seed_topic(conn, _USER_A, "work")
            await _seed_entry(conn, _USER_A, topic_id)
            await _seed_conversation(
                conn,
                _USER_A,
                topic_id,
                "synced-1",
                source="chatgpt",
                platform_id="chatgpt-abc123",
            )

        resp_a = await client_a.get(ENDPOINT, headers={"X-Auth-User-Id": str(_USER_A)})
        assert resp_a.status_code == 200
        assert resp_a.json() == {"has_entries": True, "has_synced_conversations": True}

        resp_b = await client_b.get(ENDPOINT, headers={"X-Auth-User-Id": str(_USER_B)})
        assert resp_b.status_code == 200
        assert resp_b.json() == {"has_entries": False, "has_synced_conversations": False}
