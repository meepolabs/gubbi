"""Tests for the web topics endpoints.

``GET /api/v1/topics``        -- paginated topic list.
``GET /api/v1/topics/{path}`` -- single topic metadata.

Covers:
- Auth: missing token -> 401.
- Pagination: total correctness; over-max limit -> 422; offset paging.
- Response shape: spec field set, ``topics`` envelope key.
- Cache-Control: ``private, no-store``.
- RLS isolation: user B sees an empty list and a 404 for user A's topics.

The DB-backed tests require the RLS test database. They auto-skip when the
database is unreachable (the shared pool fixtures call ``pytest.skip``), so
``pytest tests/api/v1/test_web_topics.py`` exits PASS in environments without
a running Postgres. The 422/401 tests need no DB -- FastAPI rejects before the
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
from httpx import ASGITransport, AsyncClient

from gubbi.api.v1.web.topics import router as topics_router
from gubbi.app_context import AppContext
from gubbi.auth.strategies import TrustGatewayStrategy
from gubbi.config import Settings
from gubbi.crypto.cipher import ContentCipher
from gubbi.storage.embedding_service import EmbeddingService

pytestmark = pytest.mark.asyncio(loop_scope="session")

API_PREFIX = "/api/v1"
LIST_ENDPOINT = f"{API_PREFIX}/topics"

_USER_A = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
_USER_B = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app(pool: asyncpg.Pool, *, auth_user_id: UUID | None = None) -> FastAPI:
    """Minimal FastAPI with an AppContext backed by the given pool.

    ``auth_user_id`` sets operator_user_id so the X-Auth-User-Id gateway
    header authenticates as that user (matches the extraction/ingest test
    harness: the auth strategy trusts the gateway header when
    trust_gateway=True).
    """
    settings = Settings(
        db={"app_url": ""},
        auth={
            "api_key": "test-api-key-for-unit-tests-only",
            "operator_email": "web-topics-test@test.local",
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
    app.include_router(topics_router, prefix=API_PREFIX)

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


async def _seed_topic(
    admin_conn: asyncpg.Connection,
    user_id: UUID,
    path: str,
) -> int:
    """INSERT a topic row for the user; return topic_id."""
    topic_id = await admin_conn.fetchval(
        """
        INSERT INTO topics (path, title, description, user_id, created_at, updated_at)
        VALUES ($1, $2, $3, $4, now(), now())
        RETURNING id
        """,
        path,
        path.split("/")[-1].title(),
        f"desc for {path}",
        user_id,
    )
    return int(topic_id)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def user_a(admin_pool: asyncpg.Pool) -> UUID:
    """Ensure user A exists; tear down after test."""
    async with admin_pool.acquire() as conn:
        await _seed_user(conn, _USER_A, "web-topics-a@test.local")
    yield _USER_A
    async with admin_pool.acquire() as conn:
        await conn.execute("DELETE FROM topics WHERE user_id = $1", _USER_A)
        await conn.execute("DELETE FROM users WHERE id = $1", _USER_A)


@pytest_asyncio.fixture
async def user_b(admin_pool: asyncpg.Pool) -> UUID:
    """Ensure user B exists; tear down after test."""
    async with admin_pool.acquire() as conn:
        await _seed_user(conn, _USER_B, "web-topics-b@test.local")
    yield _USER_B
    async with admin_pool.acquire() as conn:
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


class TestWebTopicsValidation:
    """Param-validation and auth rejection (pre-handler)."""

    async def test_missing_token_returns_401(self, app_pool: asyncpg.Pool) -> None:
        """No credentials and trust_gateway=False -> 401."""
        settings = Settings(
            db={"app_url": ""},
            auth={
                "api_key": "test-api-key-for-unit-tests-only",
                "operator_email": "web-topics-test@test.local",
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
        app.include_router(topics_router, prefix=API_PREFIX)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(LIST_ENDPOINT)
        assert resp.status_code == 401

    async def test_limit_over_max_returns_422(self, app_pool: asyncpg.Pool) -> None:
        """limit above the 200 cap -> 422 (Pydantic Query constraint)."""
        app = _make_app(app_pool, auth_user_id=_USER_A)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(
                LIST_ENDPOINT,
                params={"limit": 201},
                headers={"X-Auth-User-Id": str(_USER_A)},
            )
        assert resp.status_code == 422

    async def test_limit_zero_returns_422(self, app_pool: asyncpg.Pool) -> None:
        """limit below 1 -> 422."""
        app = _make_app(app_pool, auth_user_id=_USER_A)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(
                LIST_ENDPOINT,
                params={"limit": 0},
                headers={"X-Auth-User-Id": str(_USER_A)},
            )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# DB-backed tests (require the RLS test database)
# ---------------------------------------------------------------------------


class TestWebTopicsList:
    """List shape, pagination, and Cache-Control."""

    async def test_empty_list(self, client_a: AsyncClient) -> None:
        """No topics -> 200 with empty list and total 0."""
        resp = await client_a.get(LIST_ENDPOINT, headers={"X-Auth-User-Id": str(_USER_A)})
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data == {"topics": [], "total": 0, "limit": 50, "offset": 0}

    async def test_shape_and_fields(
        self,
        client_a: AsyncClient,
        admin_pool: asyncpg.Pool,
        user_a: UUID,
    ) -> None:
        """Seeded topic surfaces all spec fields under the topics envelope."""
        async with admin_pool.acquire() as conn:
            await _seed_topic(conn, _USER_A, "work/acme")

        resp = await client_a.get(LIST_ENDPOINT, headers={"X-Auth-User-Id": str(_USER_A)})
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["total"] == 1
        assert data["limit"] == 50
        assert data["offset"] == 0
        item = data["topics"][0]
        assert set(item) == {
            "id",
            "path",
            "title",
            "description",
            "entry_count",
            "created_at",
            "updated_at",
        }
        assert item["path"] == "work/acme"
        assert item["entry_count"] == 0
        assert isinstance(item["id"], int)

    async def test_total_reflects_full_set_under_limit(
        self,
        client_a: AsyncClient,
        admin_pool: asyncpg.Pool,
        user_a: UUID,
    ) -> None:
        """total is the full filtered count; a small limit pages the list."""
        async with admin_pool.acquire() as conn:
            for i in range(5):
                await _seed_topic(conn, _USER_A, f"area/topic-{i}")

        resp = await client_a.get(
            LIST_ENDPOINT,
            params={"limit": 2, "offset": 0},
            headers={"X-Auth-User-Id": str(_USER_A)},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["total"] == 5
        assert data["limit"] == 2
        assert len(data["topics"]) == 2

    async def test_prefix_filter(
        self,
        client_a: AsyncClient,
        admin_pool: asyncpg.Pool,
        user_a: UUID,
    ) -> None:
        """prefix narrows the list and the total."""
        async with admin_pool.acquire() as conn:
            await _seed_topic(conn, _USER_A, "work/acme")
            await _seed_topic(conn, _USER_A, "work/beta")
            await _seed_topic(conn, _USER_A, "health")

        resp = await client_a.get(
            LIST_ENDPOINT,
            params={"prefix": "work"},
            headers={"X-Auth-User-Id": str(_USER_A)},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["total"] == 2
        assert {t["path"] for t in data["topics"]} == {"work/acme", "work/beta"}

    async def test_cache_control_private_no_store(self, client_a: AsyncClient) -> None:
        """List response carries Cache-Control: private, no-store."""
        resp = await client_a.get(LIST_ENDPOINT, headers={"X-Auth-User-Id": str(_USER_A)})
        assert resp.status_code == 200, resp.text
        cache_header = resp.headers.get("cache-control", "")
        assert "private" in cache_header
        assert "no-store" in cache_header


class TestWebTopicsDetail:
    """Detail endpoint: hit, miss, and Cache-Control."""

    async def test_get_existing_topic(
        self,
        client_a: AsyncClient,
        admin_pool: asyncpg.Pool,
        user_a: UUID,
    ) -> None:
        """Existing path -> 200 with the topic item shape."""
        async with admin_pool.acquire() as conn:
            await _seed_topic(conn, _USER_A, "work/acme")

        resp = await client_a.get(
            f"{LIST_ENDPOINT}/work/acme",
            headers={"X-Auth-User-Id": str(_USER_A)},
        )
        assert resp.status_code == 200, resp.text
        item = resp.json()
        assert item["path"] == "work/acme"
        assert set(item) == {
            "id",
            "path",
            "title",
            "description",
            "entry_count",
            "created_at",
            "updated_at",
        }
        assert "private" in resp.headers.get("cache-control", "")
        assert "no-store" in resp.headers.get("cache-control", "")

    async def test_missing_topic_returns_404(self, client_a: AsyncClient) -> None:
        """Absent path -> 404 {"detail": "topic_not_found"}."""
        resp = await client_a.get(
            f"{LIST_ENDPOINT}/work/nope",
            headers={"X-Auth-User-Id": str(_USER_A)},
        )
        assert resp.status_code == 404
        assert resp.json() == {"detail": "topic_not_found"}

    async def test_invalid_path_returns_404(self, client_a: AsyncClient) -> None:
        """Syntactically invalid path -> 404 (cannot name an existing topic)."""
        resp = await client_a.get(
            f"{LIST_ENDPOINT}/a/b/c/d",
            headers={"X-Auth-User-Id": str(_USER_A)},
        )
        assert resp.status_code == 404
        assert resp.json() == {"detail": "topic_not_found"}


class TestWebTopicsRLS:
    """RLS isolation: user B cannot see user A's topics."""

    async def test_user_b_list_excludes_user_a_topics(
        self,
        client_a: AsyncClient,
        client_b: AsyncClient,
        admin_pool: asyncpg.Pool,
        user_a: UUID,
        user_b: UUID,
    ) -> None:
        """User A's topics are invisible to user B's list."""
        async with admin_pool.acquire() as conn:
            await _seed_topic(conn, _USER_A, "work/acme")
            await _seed_topic(conn, _USER_A, "health")

        resp_a = await client_a.get(LIST_ENDPOINT, headers={"X-Auth-User-Id": str(_USER_A)})
        assert resp_a.status_code == 200
        assert resp_a.json()["total"] == 2

        resp_b = await client_b.get(LIST_ENDPOINT, headers={"X-Auth-User-Id": str(_USER_B)})
        assert resp_b.status_code == 200
        data_b = resp_b.json()
        assert data_b["total"] == 0
        assert data_b["topics"] == []

    async def test_user_b_detail_is_404_for_user_a_topic(
        self,
        client_a: AsyncClient,
        client_b: AsyncClient,
        admin_pool: asyncpg.Pool,
        user_a: UUID,
        user_b: UUID,
    ) -> None:
        """User A's topic detail is a 404 for user B (no cross-tenant signal)."""
        async with admin_pool.acquire() as conn:
            await _seed_topic(conn, _USER_A, "work/acme")

        resp_a = await client_a.get(
            f"{LIST_ENDPOINT}/work/acme",
            headers={"X-Auth-User-Id": str(_USER_A)},
        )
        assert resp_a.status_code == 200

        resp_b = await client_b.get(
            f"{LIST_ENDPOINT}/work/acme",
            headers={"X-Auth-User-Id": str(_USER_B)},
        )
        assert resp_b.status_code == 404
        assert resp_b.json() == {"detail": "topic_not_found"}
