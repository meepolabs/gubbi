"""Tests for the web timeline endpoint.

``GET /api/v1/timeline`` -- per-bucket counts of entries and conversations.

Covers:
- Auth: missing token -> 401.
- Param validation: missing/invalid dates -> 422; inverted range -> 422;
  span over the 366-day cap -> 422; bad bucket value -> 422.
- Bucket-count correctness over a seeded set (day and month).
- topic_prefix filter narrows the counts.
- Cache-Control: ``private, no-store``.
- RLS isolation: user B's timeline excludes user A's rows.

The pure ``count_rows_to_buckets`` mapper and the ``_timeline_bucket_expr`` SQL
fragment builder are unit-tested separately in
``tests/unit/test_timeline_buckets.py`` (no DB).

The DB-backed tests require the RLS test database and auto-skip when it is
unreachable (the shared pool fixtures call ``pytest.skip``). The 401/422 tests
need no DB.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import asyncpg
import pytest
import pytest_asyncio
import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from httpx import ASGITransport, AsyncClient

from gubbi.api.v1.web.timeline import router as timeline_router
from gubbi.app_context import AppContext
from gubbi.auth.strategies import TrustGatewayStrategy
from gubbi.config import Settings
from gubbi.crypto.cipher import ContentCipher
from gubbi.storage.embedding_service import EmbeddingService

pytestmark = pytest.mark.asyncio(loop_scope="session")

API_PREFIX = "/api/v1"
TIMELINE_ENDPOINT = f"{API_PREFIX}/timeline"

_USER_A = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
_USER_B = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")

# Title-only path never decrypts, so any non-empty bytes satisfy NOT NULL.
_CT = b"\x00" * 16
_NONCE = b"\x00" * 12


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app(pool: asyncpg.Pool, *, auth_user_id: UUID | None = None) -> FastAPI:
    """Minimal FastAPI with an AppContext backed by the given pool."""
    settings = Settings(
        db={"app_url": ""},
        auth={
            "api_key": "test-api-key-for-unit-tests-only",
            "operator_email": "web-timeline-test@test.local",
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
    app.include_router(timeline_router, prefix=API_PREFIX)

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
    entry_date: date,
) -> None:
    """INSERT a single entry on a specific date (title-only path skips decryption)."""
    await conn.execute(
        """
        INSERT INTO entries
            (topic_id, user_id, date, content_encrypted, content_nonce,
             tags, created_at, updated_at)
        VALUES ($1, $2, $3, $4, $5, $6, now(), now())
        """,
        topic_id,
        user_id,
        entry_date,
        _CT,
        _NONCE,
        ["seed"],
    )


async def _seed_conversation(
    conn: asyncpg.Connection,
    user_id: UUID,
    topic_id: int,
    created_on: date,
) -> None:
    """INSERT a single conversation whose created_at::date is ``created_on``."""
    await conn.execute(
        """
        INSERT INTO conversations
            (topic_id, user_id, title_encrypted, title_nonce, slug, source,
             summary_encrypted, summary_nonce, tags, participants,
             message_count, created_at, updated_at, json_path)
        VALUES ($1, $2, $3, $4, $5, 'claude', $6, $7, $8, $9, 0, $10, $10, $11)
        """,
        topic_id,
        user_id,
        _CT,
        _NONCE,
        f"conv-{uuid4().hex[:8]}",
        _CT,
        _NONCE,
        ["seed"],
        ["user", "assistant"],
        datetime.combine(created_on, time(12, 0), tzinfo=UTC),
        f"conversations_json/{uuid4()}.json",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def user_a(admin_pool: asyncpg.Pool) -> UUID:
    """Ensure user A exists; tear down after test."""
    async with admin_pool.acquire() as conn:
        await _seed_user(conn, _USER_A, "web-timeline-a@test.local")
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
        await _seed_user(conn, _USER_B, "web-timeline-b@test.local")
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
# Validation tests (no DB needed -- rejected before the handler touches the pool)
# ---------------------------------------------------------------------------


class TestWebTimelineValidation:
    """Param-validation and auth rejection."""

    async def test_missing_token_returns_401(self, app_pool: asyncpg.Pool) -> None:
        """No credentials and trust_gateway=False -> 401."""
        settings = Settings(
            db={"app_url": ""},
            auth={
                "api_key": "test-api-key-for-unit-tests-only",
                "operator_email": "web-timeline-test@test.local",
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
        app.include_router(timeline_router, prefix=API_PREFIX)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(
                TIMELINE_ENDPOINT,
                params={"date_from": "2026-06-01", "date_to": "2026-06-30"},
            )
        assert resp.status_code == 401

    async def test_missing_date_params_returns_422(self, app_pool: asyncpg.Pool) -> None:
        """date_from / date_to are required -> 422 when absent."""
        app = _make_app(app_pool, auth_user_id=_USER_A)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(TIMELINE_ENDPOINT, headers={"X-Auth-User-Id": str(_USER_A)})
        assert resp.status_code == 422

    async def test_invalid_date_format_returns_422(self, app_pool: asyncpg.Pool) -> None:
        """Non-YYYY-MM-DD date -> 422."""
        app = _make_app(app_pool, auth_user_id=_USER_A)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(
                TIMELINE_ENDPOINT,
                params={"date_from": "06/01/2026", "date_to": "2026-06-30"},
                headers={"X-Auth-User-Id": str(_USER_A)},
            )
        assert resp.status_code == 422

    async def test_inverted_range_returns_422(self, app_pool: asyncpg.Pool) -> None:
        """date_from after date_to -> 422."""
        app = _make_app(app_pool, auth_user_id=_USER_A)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(
                TIMELINE_ENDPOINT,
                params={"date_from": "2026-06-30", "date_to": "2026-06-01"},
                headers={"X-Auth-User-Id": str(_USER_A)},
            )
        assert resp.status_code == 422

    async def test_span_over_max_returns_422(self, app_pool: asyncpg.Pool) -> None:
        """Span beyond the 366-day cap -> 422."""
        app = _make_app(app_pool, auth_user_id=_USER_A)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(
                TIMELINE_ENDPOINT,
                params={"date_from": "2025-01-01", "date_to": "2026-12-31"},
                headers={"X-Auth-User-Id": str(_USER_A)},
            )
        assert resp.status_code == 422

    async def test_bad_bucket_returns_422(self, app_pool: asyncpg.Pool) -> None:
        """bucket outside {day, month} -> 422 (Literal constraint)."""
        app = _make_app(app_pool, auth_user_id=_USER_A)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(
                TIMELINE_ENDPOINT,
                params={
                    "date_from": "2026-06-01",
                    "date_to": "2026-06-30",
                    "bucket": "week",
                },
                headers={"X-Auth-User-Id": str(_USER_A)},
            )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# DB-backed tests (require the RLS test database)
# ---------------------------------------------------------------------------


class TestWebTimelineCounts:
    """Bucket-count correctness, prefix filter, and Cache-Control."""

    async def test_empty_range_returns_empty_buckets(self, client_a: AsyncClient) -> None:
        """No rows -> 200 with empty bucket list and echoed params."""
        resp = await client_a.get(
            TIMELINE_ENDPOINT,
            params={"date_from": "2026-06-01", "date_to": "2026-06-30"},
            headers={"X-Auth-User-Id": str(_USER_A)},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json() == {
            "buckets": [],
            "date_from": "2026-06-01",
            "date_to": "2026-06-30",
            "bucket": "day",
        }

    async def test_day_bucket_counts(
        self,
        client_a: AsyncClient,
        admin_pool: asyncpg.Pool,
        user_a: UUID,
    ) -> None:
        """Entries and conversations tally into per-day buckets."""
        async with admin_pool.acquire() as conn:
            topic_id = await _seed_topic(conn, _USER_A, "work/acme")
            await _seed_entry(conn, _USER_A, topic_id, date(2026, 6, 10))
            await _seed_entry(conn, _USER_A, topic_id, date(2026, 6, 10))
            await _seed_conversation(conn, _USER_A, topic_id, date(2026, 6, 10))
            await _seed_entry(conn, _USER_A, topic_id, date(2026, 6, 12))

        resp = await client_a.get(
            TIMELINE_ENDPOINT,
            params={"date_from": "2026-06-01", "date_to": "2026-06-30"},
            headers={"X-Auth-User-Id": str(_USER_A)},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["bucket"] == "day"
        assert data["buckets"] == [
            {"date": "2026-06-10", "entry_count": 2, "conversation_count": 1},
            {"date": "2026-06-12", "entry_count": 1, "conversation_count": 0},
        ]

    async def test_month_bucket_counts(
        self,
        client_a: AsyncClient,
        admin_pool: asyncpg.Pool,
        user_a: UUID,
    ) -> None:
        """bucket=month collapses days within a month."""
        async with admin_pool.acquire() as conn:
            topic_id = await _seed_topic(conn, _USER_A, "work/acme")
            await _seed_entry(conn, _USER_A, topic_id, date(2026, 5, 5))
            await _seed_entry(conn, _USER_A, topic_id, date(2026, 6, 1))
            await _seed_conversation(conn, _USER_A, topic_id, date(2026, 6, 20))

        resp = await client_a.get(
            TIMELINE_ENDPOINT,
            params={
                "date_from": "2026-05-01",
                "date_to": "2026-06-30",
                "bucket": "month",
            },
            headers={"X-Auth-User-Id": str(_USER_A)},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["bucket"] == "month"
        assert data["buckets"] == [
            {"date": "2026-05", "entry_count": 1, "conversation_count": 0},
            {"date": "2026-06", "entry_count": 1, "conversation_count": 1},
        ]

    async def test_topic_prefix_filters_counts(
        self,
        client_a: AsyncClient,
        admin_pool: asyncpg.Pool,
        user_a: UUID,
    ) -> None:
        """topic_prefix restricts the tally to matching topic paths."""
        async with admin_pool.acquire() as conn:
            work_id = await _seed_topic(conn, _USER_A, "work/acme")
            health_id = await _seed_topic(conn, _USER_A, "health")
            await _seed_entry(conn, _USER_A, work_id, date(2026, 6, 10))
            await _seed_entry(conn, _USER_A, health_id, date(2026, 6, 10))

        resp = await client_a.get(
            TIMELINE_ENDPOINT,
            params={
                "date_from": "2026-06-01",
                "date_to": "2026-06-30",
                "topic_prefix": "work",
            },
            headers={"X-Auth-User-Id": str(_USER_A)},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["buckets"] == [
            {"date": "2026-06-10", "entry_count": 1, "conversation_count": 0},
        ]

    async def test_cache_control_private_no_store(self, client_a: AsyncClient) -> None:
        """Response carries Cache-Control: private, no-store."""
        resp = await client_a.get(
            TIMELINE_ENDPOINT,
            params={"date_from": "2026-06-01", "date_to": "2026-06-30"},
            headers={"X-Auth-User-Id": str(_USER_A)},
        )
        assert resp.status_code == 200, resp.text
        cache_header = resp.headers.get("cache-control", "")
        assert "private" in cache_header
        assert "no-store" in cache_header

    async def test_max_span_boundary_allowed(self, client_a: AsyncClient) -> None:
        """A span exactly at the 366-day cap is allowed (200, not 422)."""
        start = date(2026, 1, 1)
        end = start + timedelta(days=366)
        resp = await client_a.get(
            TIMELINE_ENDPOINT,
            params={"date_from": start.isoformat(), "date_to": end.isoformat()},
            headers={"X-Auth-User-Id": str(_USER_A)},
        )
        assert resp.status_code == 200, resp.text


class TestWebTimelineRLS:
    """RLS isolation: user B's timeline excludes user A's rows."""

    async def test_user_b_timeline_excludes_user_a_rows(
        self,
        client_a: AsyncClient,
        client_b: AsyncClient,
        admin_pool: asyncpg.Pool,
        user_a: UUID,
        user_b: UUID,
    ) -> None:
        """User A's entries are invisible to user B's timeline."""
        async with admin_pool.acquire() as conn:
            topic_id = await _seed_topic(conn, _USER_A, "work/acme")
            await _seed_entry(conn, _USER_A, topic_id, date(2026, 6, 10))

        params = {"date_from": "2026-06-01", "date_to": "2026-06-30"}

        resp_a = await client_a.get(
            TIMELINE_ENDPOINT, params=params, headers={"X-Auth-User-Id": str(_USER_A)}
        )
        assert resp_a.status_code == 200
        assert resp_a.json()["buckets"] == [
            {"date": "2026-06-10", "entry_count": 1, "conversation_count": 0},
        ]

        resp_b = await client_b.get(
            TIMELINE_ENDPOINT, params=params, headers={"X-Auth-User-Id": str(_USER_B)}
        )
        assert resp_b.status_code == 200
        assert resp_b.json()["buckets"] == []
