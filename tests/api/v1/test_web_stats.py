"""Tests for the web stats endpoint.

``GET /api/v1/stats`` -- dashboard aggregate: topic/entry/conversation totals,
this-week entry count (user timezone), most-recent entry timestamp, and the
top-5 most-active topics over the last 30 days.

Covers:
- Auth: missing token -> 401.
- Empty journal -> all-zero counts, null last_entry_at, empty active list.
- Correct counts over a seeded set (topics/entries/conversations totals).
- entries_this_week boundary: an entry inside the current week counts, one
  outside does not.
- most_active_topics ordering (by recent count) and the top-5 cap.
- Cache-Control: ``private, no-store``.
- RLS isolation: user B sees only their own zeros, never user A's data.

The DB-backed tests require the RLS test database and auto-skip when it is
unreachable (the shared pool fixtures call ``pytest.skip``). The 401 test needs
no DB. The pure ``user_week_bounds`` helper is unit-tested separately in
``tests/unit/test_stats_week_bounds.py``.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import asyncpg
import pytest
import pytest_asyncio
import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from httpx import ASGITransport, AsyncClient

from gubbi.api.v1.web.stats import router as stats_router
from gubbi.app_context import AppContext
from gubbi.auth.strategies import TrustGatewayStrategy
from gubbi.config import Settings
from gubbi.crypto.cipher import ContentCipher
from gubbi.storage.embedding_service import EmbeddingService
from gubbi.storage.repositories.stats import user_week_bounds

pytestmark = pytest.mark.asyncio(loop_scope="session")

API_PREFIX = "/api/v1"
STATS_ENDPOINT = f"{API_PREFIX}/stats"

_USER_A = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
_USER_B = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")

# Stats never decrypt, so any non-empty bytes satisfy the NOT NULL columns.
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
            "operator_email": "web-stats-test@test.local",
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
    app.include_router(stats_router, prefix=API_PREFIX)

    @app.exception_handler(Exception)
    async def _handler(request: Request, exc: Exception) -> JSONResponse:
        raise exc

    return app


async def _seed_user(conn: asyncpg.Connection, user_id: UUID, email: str) -> None:
    """INSERT user with UTC timezone; ON CONFLICT DO NOTHING for idempotency."""
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
    """INSERT a single entry on a specific date (no decryption path used)."""
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
) -> None:
    """INSERT a single conversation row for the user."""
    await conn.execute(
        """
        INSERT INTO conversations
            (topic_id, user_id, title_encrypted, title_nonce, slug, source,
             summary_encrypted, summary_nonce, tags, participants,
             message_count, created_at, updated_at, json_path)
        VALUES ($1, $2, $3, $4, $5, 'claude', $6, $7, $8, $9, 0, now(), now(), $10)
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
        f"conversations_json/{uuid4()}.json",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def user_a(admin_pool: asyncpg.Pool) -> UUID:
    """Ensure user A exists; tear down after test."""
    async with admin_pool.acquire() as conn:
        await _seed_user(conn, _USER_A, "web-stats-a@test.local")
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
        await _seed_user(conn, _USER_B, "web-stats-b@test.local")
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
# Auth (no DB needed -- rejected before the handler touches the pool)
# ---------------------------------------------------------------------------


class TestWebStatsAuth:
    """Auth rejection."""

    async def test_missing_token_returns_401(self, app_pool: asyncpg.Pool) -> None:
        """No credentials and trust_gateway=False -> 401."""
        settings = Settings(
            db={"app_url": ""},
            auth={
                "api_key": "test-api-key-for-unit-tests-only",
                "operator_email": "web-stats-test@test.local",
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
        app.include_router(stats_router, prefix=API_PREFIX)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(STATS_ENDPOINT)
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# DB-backed tests (require the RLS test database)
# ---------------------------------------------------------------------------


class TestWebStatsCounts:
    """Totals, this-week boundary, active topics, and Cache-Control."""

    async def test_empty_journal_returns_zeros(self, client_a: AsyncClient) -> None:
        """No data -> all-zero counts, null last_entry_at, empty active list."""
        resp = await client_a.get(STATS_ENDPOINT, headers={"X-Auth-User-Id": str(_USER_A)})
        assert resp.status_code == 200, resp.text
        assert resp.json() == {
            "topics_total": 0,
            "entries_total": 0,
            "entries_this_week": 0,
            "conversations_total": 0,
            "last_entry_at": None,
            "most_active_topics": [],
        }

    async def test_totals_over_seeded_set(
        self,
        client_a: AsyncClient,
        admin_pool: asyncpg.Pool,
        user_a: UUID,
    ) -> None:
        """topics/entries/conversations totals reflect the seeded rows."""
        week_from, _week_to = user_week_bounds("UTC")
        async with admin_pool.acquire() as conn:
            t1 = await _seed_topic(conn, _USER_A, "work/acme")
            t2 = await _seed_topic(conn, _USER_A, "health")
            await _seed_entry(conn, _USER_A, t1, week_from)
            await _seed_entry(conn, _USER_A, t1, week_from)
            await _seed_entry(conn, _USER_A, t2, week_from)
            await _seed_conversation(conn, _USER_A, t1)

        resp = await client_a.get(STATS_ENDPOINT, headers={"X-Auth-User-Id": str(_USER_A)})
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["topics_total"] == 2
        assert data["entries_total"] == 3
        assert data["conversations_total"] == 1
        assert data["last_entry_at"] is not None

    async def test_soft_deleted_entries_excluded_from_total(
        self,
        client_a: AsyncClient,
        admin_pool: asyncpg.Pool,
        user_a: UUID,
    ) -> None:
        """entries_total counts only non-soft-deleted entries."""
        week_from, _week_to = user_week_bounds("UTC")
        async with admin_pool.acquire() as conn:
            topic_id = await _seed_topic(conn, _USER_A, "work/acme")
            await _seed_entry(conn, _USER_A, topic_id, week_from)
            await conn.execute(
                """
                INSERT INTO entries
                    (topic_id, user_id, date, content_encrypted, content_nonce,
                     tags, created_at, updated_at, deleted_at)
                VALUES ($1, $2, $3, $4, $5, $6, now(), now(), now())
                """,
                topic_id,
                _USER_A,
                week_from,
                _CT,
                _NONCE,
                ["seed"],
            )

        resp = await client_a.get(STATS_ENDPOINT, headers={"X-Auth-User-Id": str(_USER_A)})
        assert resp.status_code == 200, resp.text
        assert resp.json()["entries_total"] == 1

    async def test_entries_this_week_boundary(
        self,
        client_a: AsyncClient,
        admin_pool: asyncpg.Pool,
        user_a: UUID,
    ) -> None:
        """An entry inside the current week counts; one before it does not."""
        week_from, week_to = user_week_bounds("UTC")
        inside = week_from
        inside_end = week_to
        outside = week_from - timedelta(days=1)  # last day of the previous week
        async with admin_pool.acquire() as conn:
            topic_id = await _seed_topic(conn, _USER_A, "work/acme")
            await _seed_entry(conn, _USER_A, topic_id, inside)
            await _seed_entry(conn, _USER_A, topic_id, inside_end)
            await _seed_entry(conn, _USER_A, topic_id, outside)

        resp = await client_a.get(STATS_ENDPOINT, headers={"X-Auth-User-Id": str(_USER_A)})
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["entries_total"] == 3
        assert data["entries_this_week"] == 2

    async def test_most_active_topics_ordering_and_cap(
        self,
        client_a: AsyncClient,
        admin_pool: asyncpg.Pool,
        user_a: UUID,
    ) -> None:
        """Active list is ordered by recent count desc and capped at five."""
        recent = user_week_bounds("UTC")[0]  # within the last 30 days
        async with admin_pool.acquire() as conn:
            # Seven topics with descending recent-entry counts (7..1).
            for rank in range(7):
                topic_id = await _seed_topic(conn, _USER_A, f"area/topic-{rank}")
                for _ in range(7 - rank):
                    await _seed_entry(conn, _USER_A, topic_id, recent)

        resp = await client_a.get(STATS_ENDPOINT, headers={"X-Auth-User-Id": str(_USER_A)})
        assert resp.status_code == 200, resp.text
        active = resp.json()["most_active_topics"]
        assert len(active) == 5  # top-5 cap
        counts = [t["entries_last_30d"] for t in active]
        assert counts == sorted(counts, reverse=True)
        assert counts == [7, 6, 5, 4, 3]
        assert active[0]["path"] == "area/topic-0"
        assert set(active[0]) == {"path", "title", "entries_last_30d"}

    async def test_active_topics_excludes_old_entries(
        self,
        client_a: AsyncClient,
        admin_pool: asyncpg.Pool,
        user_a: UUID,
    ) -> None:
        """Entries older than the 30-day window do not count toward active topics."""
        recent = user_week_bounds("UTC")[1]
        old = recent - timedelta(days=60)
        async with admin_pool.acquire() as conn:
            recent_topic = await _seed_topic(conn, _USER_A, "work/recent")
            old_topic = await _seed_topic(conn, _USER_A, "work/old")
            await _seed_entry(conn, _USER_A, recent_topic, recent)
            await _seed_entry(conn, _USER_A, old_topic, old)

        resp = await client_a.get(STATS_ENDPOINT, headers={"X-Auth-User-Id": str(_USER_A)})
        assert resp.status_code == 200, resp.text
        active = resp.json()["most_active_topics"]
        assert [t["path"] for t in active] == ["work/recent"]

    async def test_cache_control_private_no_store(self, client_a: AsyncClient) -> None:
        """Response carries Cache-Control: private, no-store."""
        resp = await client_a.get(STATS_ENDPOINT, headers={"X-Auth-User-Id": str(_USER_A)})
        assert resp.status_code == 200, resp.text
        cache_header = resp.headers.get("cache-control", "")
        assert "private" in cache_header
        assert "no-store" in cache_header


class TestWebStatsRLS:
    """RLS isolation: user B never sees user A's aggregate."""

    async def test_user_b_sees_only_own_zeros(
        self,
        client_a: AsyncClient,
        client_b: AsyncClient,
        admin_pool: asyncpg.Pool,
        user_a: UUID,
        user_b: UUID,
    ) -> None:
        """User A's seeded data does not leak into user B's stats."""
        week_from = user_week_bounds("UTC")[0]
        async with admin_pool.acquire() as conn:
            topic_id = await _seed_topic(conn, _USER_A, "work/acme")
            await _seed_entry(conn, _USER_A, topic_id, week_from)
            await _seed_conversation(conn, _USER_A, topic_id)

        resp_a = await client_a.get(STATS_ENDPOINT, headers={"X-Auth-User-Id": str(_USER_A)})
        assert resp_a.status_code == 200
        data_a = resp_a.json()
        assert data_a["topics_total"] == 1
        assert data_a["entries_total"] == 1
        assert data_a["conversations_total"] == 1

        resp_b = await client_b.get(STATS_ENDPOINT, headers={"X-Auth-User-Id": str(_USER_B)})
        assert resp_b.status_code == 200
        assert resp_b.json() == {
            "topics_total": 0,
            "entries_total": 0,
            "entries_this_week": 0,
            "conversations_total": 0,
            "last_entry_at": None,
            "most_active_topics": [],
        }
