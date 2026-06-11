"""Tests for the web search endpoint.

``GET /api/v1/search`` -- hybrid FTS + semantic journal search.

Covers:
- Auth: missing token -> 401.
- Validation: missing ``q`` -> 422; ``limit`` over 50 -> 422; ``limit`` 0 -> 422;
  ``q`` over 2000 chars -> 422.
- Response shape: spec field set per result; ``{results, total, query}`` body;
  Cache-Control ``private, no-store``.
- RLS isolation: user B's search never returns user A's content.

The DB-backed tests require the RLS test database. They auto-skip when the
database is unreachable (the shared pool fixtures call ``pytest.skip``), so
``pytest tests/api/v1/test_web_search.py`` exits PASS in environments without a
running Postgres. The 422/401 tests need no DB -- FastAPI rejects before the
handler touches the pool.
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
from gubbi_common.db.user_scoped import user_scoped_connection
from httpx import ASGITransport, AsyncClient

from gubbi.api.v1.web.search import router as search_router
from gubbi.app_context import AppContext
from gubbi.auth.strategies import TrustGatewayStrategy
from gubbi.config import Settings
from gubbi.crypto.cipher import ContentCipher
from gubbi.storage.embedding_service import EmbeddingService
from gubbi.storage.repositories import entries as entry_repo

pytestmark = pytest.mark.asyncio(loop_scope="session")

API_PREFIX = "/api/v1"
SEARCH_ENDPOINT = f"{API_PREFIX}/search"

_USER_A = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
_USER_B = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")

# Deterministic 32-byte content key so the cipher can decrypt seeded rows.
_CIPHER = ContentCipher({1: bytes([1]) * 32})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app(pool: asyncpg.Pool, *, auth_user_id: UUID | None = None) -> FastAPI:
    """Minimal FastAPI with an AppContext backed by the given pool.

    ``auth_user_id`` sets operator_user_id so the X-Auth-User-Id gateway
    header authenticates as that user (matches the topics test harness: the
    auth strategy trusts the gateway header when trust_gateway=True).
    """
    settings = Settings(
        db={"app_url": ""},
        auth={
            "api_key": "test-api-key-for-unit-tests-only",
            "operator_email": "web-search-test@test.local",
            "trust_gateway": True,
        },
        server={"url": "http://localhost:8100"},
        data_dir=str(Path(__file__).parent),
    )
    app_ctx = AppContext(
        pool=pool,
        embedding_service=EmbeddingService(),
        settings=settings,
        logger=structlog.get_logger("test"),
        admin_pool=None,
        operator_user_id=auth_user_id,
        cipher=_CIPHER,
    )
    app = FastAPI()
    app.state.app_ctx = app_ctx
    app.state.auth_strategies = [
        TrustGatewayStrategy(gateway_secret=None, gateway_require_signature=False),
    ]
    app.include_router(search_router, prefix=API_PREFIX)

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


async def _seed_topic(admin_conn: asyncpg.Connection, user_id: UUID, path: str) -> None:
    """INSERT a topic row for the user."""
    await admin_conn.execute(
        """
        INSERT INTO topics (path, title, description, user_id, created_at, updated_at)
        VALUES ($1, $2, $3, $4, now(), now())
        """,
        path,
        path.split("/")[-1].title(),
        f"desc for {path}",
        user_id,
    )


async def _seed_entry(pool: asyncpg.Pool, user_id: UUID, topic: str, content: str) -> int:
    """Append a real, FTS-indexed entry through the repo (RLS-scoped)."""
    async with user_scoped_connection(pool, user_id=user_id) as conn, conn.transaction():
        return await entry_repo.append(conn, _CIPHER, topic, content)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def user_a(admin_pool: asyncpg.Pool) -> UUID:
    """Ensure user A exists; tear down after test."""
    async with admin_pool.acquire() as conn:
        await _seed_user(conn, _USER_A, "web-search-a@test.local")
    yield _USER_A
    async with admin_pool.acquire() as conn:
        await conn.execute("DELETE FROM entries WHERE user_id = $1", _USER_A)
        await conn.execute("DELETE FROM topics WHERE user_id = $1", _USER_A)
        await conn.execute("DELETE FROM users WHERE id = $1", _USER_A)


@pytest_asyncio.fixture
async def user_b(admin_pool: asyncpg.Pool) -> UUID:
    """Ensure user B exists; tear down after test."""
    async with admin_pool.acquire() as conn:
        await _seed_user(conn, _USER_B, "web-search-b@test.local")
    yield _USER_B
    async with admin_pool.acquire() as conn:
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
# Validation tests (no DB needed -- FastAPI rejects before the handler runs)
# ---------------------------------------------------------------------------


class TestWebSearchValidation:
    """Param-validation and auth rejection (pre-handler)."""

    async def test_missing_token_returns_401(self, app_pool: asyncpg.Pool) -> None:
        """No credentials and trust_gateway=False -> 401."""
        settings = Settings(
            db={"app_url": ""},
            auth={
                "api_key": "test-api-key-for-unit-tests-only",
                "operator_email": "web-search-test@test.local",
                "trust_gateway": False,
            },
            server={"url": "http://localhost:8100"},
            data_dir=str(Path(__file__).parent),
        )
        app_ctx = AppContext(
            pool=app_pool,
            embedding_service=EmbeddingService(),
            settings=settings,
            logger=structlog.get_logger("test"),
            operator_user_id=None,
            cipher=_CIPHER,
        )
        app = FastAPI()
        app.state.app_ctx = app_ctx
        app.state.auth_strategies = [
            TrustGatewayStrategy(gateway_secret=None, gateway_require_signature=False),
        ]
        app.include_router(search_router, prefix=API_PREFIX)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(SEARCH_ENDPOINT, params={"q": "anything"})
        assert resp.status_code == 401

    async def test_missing_q_returns_422(self, app_pool: asyncpg.Pool) -> None:
        """Required ``q`` absent -> 422."""
        app = _make_app(app_pool, auth_user_id=_USER_A)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(SEARCH_ENDPOINT, headers={"X-Auth-User-Id": str(_USER_A)})
        assert resp.status_code == 422

    async def test_empty_q_returns_422(self, app_pool: asyncpg.Pool) -> None:
        """Empty ``q`` (min_length 1) -> 422."""
        app = _make_app(app_pool, auth_user_id=_USER_A)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(
                SEARCH_ENDPOINT,
                params={"q": ""},
                headers={"X-Auth-User-Id": str(_USER_A)},
            )
        assert resp.status_code == 422

    async def test_q_over_max_returns_422(self, app_pool: asyncpg.Pool) -> None:
        """``q`` over 2000 chars -> 422."""
        app = _make_app(app_pool, auth_user_id=_USER_A)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(
                SEARCH_ENDPOINT,
                params={"q": "x" * 2001},
                headers={"X-Auth-User-Id": str(_USER_A)},
            )
        assert resp.status_code == 422

    async def test_limit_over_max_returns_422(self, app_pool: asyncpg.Pool) -> None:
        """limit above the 50 cap -> 422 (Pydantic Query constraint)."""
        app = _make_app(app_pool, auth_user_id=_USER_A)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(
                SEARCH_ENDPOINT,
                params={"q": "x", "limit": 51},
                headers={"X-Auth-User-Id": str(_USER_A)},
            )
        assert resp.status_code == 422

    async def test_limit_zero_returns_422(self, app_pool: asyncpg.Pool) -> None:
        """limit below 1 -> 422."""
        app = _make_app(app_pool, auth_user_id=_USER_A)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(
                SEARCH_ENDPOINT,
                params={"q": "x", "limit": 0},
                headers={"X-Auth-User-Id": str(_USER_A)},
            )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# DB-backed tests (require the RLS test database)
# ---------------------------------------------------------------------------


class TestWebSearchResults:
    """Result shape, body envelope, and Cache-Control."""

    async def test_empty_results(self, client_a: AsyncClient) -> None:
        """No matching content -> 200 with empty results and total 0."""
        resp = await client_a.get(
            SEARCH_ENDPOINT,
            params={"q": "nonexistent-term-zzzz"},
            headers={"X-Auth-User-Id": str(_USER_A)},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"results": [], "total": 0, "query": "nonexistent-term-zzzz"}

    async def test_entry_result_shape(
        self,
        client_a: AsyncClient,
        app_pool: asyncpg.Pool,
        admin_pool: asyncpg.Pool,
        user_a: UUID,
    ) -> None:
        """A matching entry surfaces the entry result field set."""
        async with admin_pool.acquire() as conn:
            await _seed_topic(conn, _USER_A, "work/acme")
        await _seed_entry(app_pool, _USER_A, "work/acme", "marathon training schedule")

        resp = await client_a.get(
            SEARCH_ENDPOINT,
            params={"q": "marathon"},
            headers={"X-Auth-User-Id": str(_USER_A)},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["query"] == "marathon"
        assert data["total"] >= 1
        entry = next(r for r in data["results"] if r["doc_type"] == "entry")
        assert set(entry) == {
            "doc_type",
            "topic",
            "date",
            "entry_id",
            "conversation_id",
            "content",
            "decryption_failed",
        }
        assert entry["topic"] == "work/acme"
        assert entry["conversation_id"] is None
        assert entry["decryption_failed"] is False
        assert "marathon" in entry["content"]

    async def test_cache_control_private_no_store(
        self,
        client_a: AsyncClient,
        user_a: UUID,
    ) -> None:
        """Search response carries Cache-Control: private, no-store."""
        resp = await client_a.get(
            SEARCH_ENDPOINT,
            params={"q": "anything"},
            headers={"X-Auth-User-Id": str(_USER_A)},
        )
        assert resp.status_code == 200, resp.text
        cache_header = resp.headers.get("cache-control", "")
        assert "private" in cache_header
        assert "no-store" in cache_header

    async def test_limit_caps_result_set(
        self,
        client_a: AsyncClient,
        app_pool: asyncpg.Pool,
        admin_pool: asyncpg.Pool,
        user_a: UUID,
    ) -> None:
        """limit bounds the returned result count."""
        async with admin_pool.acquire() as conn:
            await _seed_topic(conn, _USER_A, "work/acme")
        for i in range(3):
            await _seed_entry(app_pool, _USER_A, "work/acme", f"shared keyword entry {i}")

        resp = await client_a.get(
            SEARCH_ENDPOINT,
            params={"q": "keyword", "limit": 1},
            headers={"X-Auth-User-Id": str(_USER_A)},
        )
        assert resp.status_code == 200, resp.text
        assert len(resp.json()["results"]) == 1


class TestWebSearchRLS:
    """RLS isolation: user B's search never returns user A's content."""

    async def test_user_b_search_excludes_user_a_entries(
        self,
        client_a: AsyncClient,
        client_b: AsyncClient,
        app_pool: asyncpg.Pool,
        admin_pool: asyncpg.Pool,
        user_a: UUID,
        user_b: UUID,
    ) -> None:
        """User A's entry is invisible to user B's search for the same term."""
        async with admin_pool.acquire() as conn:
            await _seed_topic(conn, _USER_A, "work/acme")
        await _seed_entry(app_pool, _USER_A, "work/acme", "confidential marathon plan")

        resp_a = await client_a.get(
            SEARCH_ENDPOINT,
            params={"q": "marathon"},
            headers={"X-Auth-User-Id": str(_USER_A)},
        )
        assert resp_a.status_code == 200
        assert resp_a.json()["total"] >= 1

        resp_b = await client_b.get(
            SEARCH_ENDPOINT,
            params={"q": "marathon"},
            headers={"X-Auth-User-Id": str(_USER_B)},
        )
        assert resp_b.status_code == 200
        assert resp_b.json()["total"] == 0
        assert resp_b.json()["results"] == []
