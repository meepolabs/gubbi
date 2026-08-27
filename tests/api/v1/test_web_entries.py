"""Tests for the web entries read + mutation endpoints.

``GET    /api/v1/entries``        -- cross-topic list with filters.
``GET    /api/v1/entries/{id}``   -- single-entry detail (adds reasoning).
``PATCH  /api/v1/entries/{id}``   -- replace-semantics partial update.
``DELETE /api/v1/entries/{id}``   -- soft-delete (idempotent 404 once gone).
``POST   /api/v1/entries/move``   -- bulk-move up to 100 entries.

Covers:
- Auth: missing token -> 401 on a read and a write endpoint.
- List filters: topic, tags (AND), date range, source, sort; pagination caps
  (over-max / zero limit -> 422); list omits reasoning, detail includes it.
- Decryption-failure path: a corrupted row yields the sentinel + flag, 200.
- PATCH: replace content (re-embed path exercised), 422 on empty body, move via
  topic_path, 404 on missing entry.
- DELETE: soft-delete then idempotent 404.
- Move: happy path + audit row asserted; >100 ids -> 422; missing id -> 404.
- RLS isolation: user B cannot read or mutate user A's entries.

The DB-backed tests require the RLS test database; they auto-skip when it is
unreachable (the shared pool fixtures call ``pytest.skip``). The 401/422 tests
need no DB -- FastAPI rejects before the handler touches the pool.
"""

from __future__ import annotations

import json
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

from gubbi.api.v1.web.entries import router as entries_router
from gubbi.api.v1.web.entries_admin import router as entries_admin_router
from gubbi.app_context import AppContext
from gubbi.auth.strategies import TrustGatewayStrategy
from gubbi.config import Settings
from gubbi.crypto.cipher import ContentCipher
from gubbi.storage.embedding_service import EmbeddingService
from gubbi.storage.repositories import entries as entries_repo

pytestmark = pytest.mark.asyncio(loop_scope="session")

API_PREFIX = "/api/v1"
ENTRIES_ENDPOINT = f"{API_PREFIX}/entries"
MOVE_ENDPOINT = f"{ENTRIES_ENDPOINT}/move"

_USER_A = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
_USER_B = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")

# Single shared key so seeded ciphertext decrypts under the cipher the app uses.
_CIPHER = ContentCipher({1: bytes([1]) * 32})

_HDR_A = {"X-Auth-User-Id": str(_USER_A)}
_HDR_B = {"X-Auth-User-Id": str(_USER_B)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app(pool: asyncpg.Pool, *, auth_user_id: UUID | None = None) -> FastAPI:
    """Minimal FastAPI mounting both entries routers, backed by the given pool.

    ``auth_user_id`` sets operator_user_id so the X-Auth-User-Id gateway header
    authenticates as that user. With no X-Auth-Scopes header the gateway strategy
    grants ``journal:read journal:write``, so write endpoints authorize.
    """
    settings = Settings(
        db={"app_url": ""},
        auth={
            "api_key": "test-api-key-for-unit-tests-only",
            "operator_email": "web-entries-test@test.local",
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
    app.include_router(entries_router, prefix=API_PREFIX)
    app.include_router(entries_admin_router, prefix=API_PREFIX)

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


async def _seed_topic(admin_conn: asyncpg.Connection, user_id: UUID, path: str) -> int:
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


async def _seed_entry(
    pool: asyncpg.Pool,
    user_id: UUID,
    topic: str,
    content: str,
    *,
    reasoning: str | None = None,
    tags: list[str] | None = None,
    date: str | None = None,
) -> int:
    """Append a real, FTS-indexed entry through the repo (RLS-scoped). Return id.

    Uses ``entries.append`` so the seeded ciphertext decrypts under the shared
    test cipher; the topic must already exist.
    """
    async with user_scoped_connection(pool, user_id=user_id) as conn, conn.transaction():
        return await entries_repo.append(
            conn, _CIPHER, topic, content, reasoning=reasoning, tags=tags, date=date
        )


async def _link_entry_source(
    admin_conn: asyncpg.Connection,
    entry_id: int,
    user_id: UUID,
    topic_id: int,
    source: str,
) -> None:
    """Attach a conversation with ``source`` to an entry (for the source filter)."""
    conv_id = await admin_conn.fetchval(
        """
        INSERT INTO conversations
            (topic_id, user_id, title_encrypted, title_nonce, slug, source,
             summary_encrypted, summary_nonce, tags, participants,
             message_count, created_at, updated_at, json_path, search_vector)
        VALUES ($1, $2, $3, $4, gen_random_uuid()::text, $5, $6, $7, '{}', '{}',
                0, now(), now(), 'test.json', to_tsvector('english', 'seed'))
        RETURNING id
        """,
        topic_id,
        user_id,
        *_CIPHER.encrypt("t"),
        source,
        *_CIPHER.encrypt("s"),
    )
    await admin_conn.execute(
        "UPDATE entries SET conversation_id = $1 WHERE id = $2", int(conv_id), entry_id
    )


async def _corrupt_entry_content(admin_conn: asyncpg.Connection, entry_id: int) -> None:
    """Overwrite an entry's content ciphertext with undecryptable bytes."""
    await admin_conn.execute(
        "UPDATE entries SET content_encrypted = $1 WHERE id = $2",
        b"\x00\x01\x02\x03not-real-ciphertext",
        entry_id,
    )


async def _audit_rows(admin_conn: asyncpg.Connection, action: str) -> list[asyncpg.Record]:
    """Fetch audit rows for an action, newest first."""
    rows: list[asyncpg.Record] = await admin_conn.fetch(
        "SELECT actor_id, action, target_kind, target_id, metadata"
        " FROM audit_log WHERE action = $1 ORDER BY id DESC",
        action,
    )
    return rows


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


async def _teardown(admin_pool: asyncpg.Pool, user_id: UUID) -> None:
    """Delete the user's rows innermost-first.

    Every FK in this chain is ON DELETE RESTRICT, so the order is load-bearing:
    entries reference conversations, and both reference topics.
    """
    async with admin_pool.acquire() as conn:
        await conn.execute("DELETE FROM entries WHERE user_id = $1", user_id)
        await conn.execute("DELETE FROM conversations WHERE user_id = $1", user_id)
        await conn.execute("DELETE FROM topics WHERE user_id = $1", user_id)
        await conn.execute("DELETE FROM users WHERE id = $1", user_id)


@pytest_asyncio.fixture
async def user_a(admin_pool: asyncpg.Pool) -> UUID:
    """Ensure user A exists; tear down their data after test."""
    async with admin_pool.acquire() as conn:
        await _seed_user(conn, _USER_A, "web-entries-a@test.local")
    yield _USER_A
    await _teardown(admin_pool, _USER_A)


@pytest_asyncio.fixture
async def user_b(admin_pool: asyncpg.Pool) -> UUID:
    """Ensure user B exists; tear down their data after test."""
    async with admin_pool.acquire() as conn:
        await _seed_user(conn, _USER_B, "web-entries-b@test.local")
    yield _USER_B
    await _teardown(admin_pool, _USER_B)


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
# Validation / auth (no DB needed -- rejected before the handler runs)
# ---------------------------------------------------------------------------


class TestWebEntriesValidation:
    """Auth and param-validation rejection (pre-handler)."""

    def _no_trust_app(self, app_pool: asyncpg.Pool) -> FastAPI:
        settings = Settings(
            db={"app_url": ""},
            auth={
                "api_key": "test-api-key-for-unit-tests-only",
                "operator_email": "web-entries-test@test.local",
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
        app.include_router(entries_router, prefix=API_PREFIX)
        app.include_router(entries_admin_router, prefix=API_PREFIX)
        return app

    async def test_list_missing_token_returns_401(self, app_pool: asyncpg.Pool) -> None:
        """GET /entries with no credentials -> 401 (read scope)."""
        app = self._no_trust_app(app_pool)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(ENTRIES_ENDPOINT)
        assert resp.status_code == 401

    async def test_move_missing_token_returns_401(self, app_pool: asyncpg.Pool) -> None:
        """POST /entries/move with no credentials -> 401 (write scope)."""
        app = self._no_trust_app(app_pool)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(MOVE_ENDPOINT, json={"entry_ids": [1], "topic_id": 1})
        assert resp.status_code == 401

    async def test_limit_over_max_returns_422(self, app_pool: asyncpg.Pool) -> None:
        """limit above the 100 cap -> 422."""
        app = _make_app(app_pool, auth_user_id=_USER_A)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(ENTRIES_ENDPOINT, params={"limit": 101}, headers=_HDR_A)
        assert resp.status_code == 422

    async def test_limit_zero_returns_422(self, app_pool: asyncpg.Pool) -> None:
        """limit below 1 -> 422."""
        app = _make_app(app_pool, auth_user_id=_USER_A)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(ENTRIES_ENDPOINT, params={"limit": 0}, headers=_HDR_A)
        assert resp.status_code == 422

    async def test_bad_sort_returns_422(self, app_pool: asyncpg.Pool) -> None:
        """An unrecognized sort value -> 422 (Query pattern constraint)."""
        app = _make_app(app_pool, auth_user_id=_USER_A)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(ENTRIES_ENDPOINT, params={"sort": "sideways"}, headers=_HDR_A)
        assert resp.status_code == 422

    async def test_move_over_max_ids_returns_422(self, app_pool: asyncpg.Pool) -> None:
        """More than 100 ids -> 422 before any DB work."""
        app = _make_app(app_pool, auth_user_id=_USER_A)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                MOVE_ENDPOINT,
                json={"entry_ids": list(range(1, 102)), "topic_id": 1},
                headers=_HDR_A,
            )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# List (DB-backed)
# ---------------------------------------------------------------------------


class TestWebEntriesList:
    """List shape, filters, pagination, and Cache-Control."""

    async def test_empty_list(self, client_a: AsyncClient) -> None:
        """No entries -> 200 with empty list and total 0."""
        resp = await client_a.get(ENTRIES_ENDPOINT, headers=_HDR_A)
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"entries": [], "total": 0, "limit": 20, "offset": 0}

    async def test_shape_omits_reasoning(
        self, client_a: AsyncClient, app_pool: asyncpg.Pool, admin_pool: asyncpg.Pool, user_a: UUID
    ) -> None:
        """List item carries the spec fields and NO reasoning key."""
        async with admin_pool.acquire() as conn:
            await _seed_topic(conn, _USER_A, "work/acme")
        await _seed_entry(
            app_pool, _USER_A, "work/acme", "hello world", reasoning="because", tags=["decision"]
        )

        resp = await client_a.get(ENTRIES_ENDPOINT, headers=_HDR_A)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["total"] == 1
        item = data["entries"][0]
        assert set(item) == {
            "id",
            "topic_path",
            "date",
            "content",
            "tags",
            "conversation_id",
            "created_at",
            "updated_at",
            "decryption_failed",
        }
        assert "reasoning" not in item
        assert item["topic_path"] == "work/acme"
        assert item["content"] == "hello world"
        assert item["tags"] == ["decision"]
        assert item["decryption_failed"] is False

    async def test_topic_filter_exact(
        self, client_a: AsyncClient, app_pool: asyncpg.Pool, admin_pool: asyncpg.Pool, user_a: UUID
    ) -> None:
        """The topic filter matches an exact path, not a prefix."""
        async with admin_pool.acquire() as conn:
            await _seed_topic(conn, _USER_A, "work/acme")
            await _seed_topic(conn, _USER_A, "work/other")
        await _seed_entry(app_pool, _USER_A, "work/acme", "a")
        await _seed_entry(app_pool, _USER_A, "work/other", "b")

        resp = await client_a.get(ENTRIES_ENDPOINT, params={"topic": "work/acme"}, headers=_HDR_A)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["total"] == 1
        assert data["entries"][0]["topic_path"] == "work/acme"

    async def test_tags_and_semantics(
        self, client_a: AsyncClient, app_pool: asyncpg.Pool, admin_pool: asyncpg.Pool, user_a: UUID
    ) -> None:
        """Repeated tags AND together: a row must carry every requested tag."""
        async with admin_pool.acquire() as conn:
            await _seed_topic(conn, _USER_A, "work/acme")
        await _seed_entry(app_pool, _USER_A, "work/acme", "both", tags=["a", "b"])
        await _seed_entry(app_pool, _USER_A, "work/acme", "one", tags=["a"])

        resp = await client_a.get(
            ENTRIES_ENDPOINT, params=[("tags", "a"), ("tags", "b")], headers=_HDR_A
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["total"] == 1
        assert data["entries"][0]["content"] == "both"

    async def test_date_range_filter(
        self, client_a: AsyncClient, app_pool: asyncpg.Pool, admin_pool: asyncpg.Pool, user_a: UUID
    ) -> None:
        """date_from / date_to bound the list inclusively."""
        async with admin_pool.acquire() as conn:
            await _seed_topic(conn, _USER_A, "work/acme")
        await _seed_entry(app_pool, _USER_A, "work/acme", "old", date="2026-01-01")
        await _seed_entry(app_pool, _USER_A, "work/acme", "mid", date="2026-06-10")
        await _seed_entry(app_pool, _USER_A, "work/acme", "new", date="2026-12-31")

        resp = await client_a.get(
            ENTRIES_ENDPOINT,
            params={"date_from": "2026-06-01", "date_to": "2026-06-30"},
            headers=_HDR_A,
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["total"] == 1
        assert data["entries"][0]["content"] == "mid"

    async def test_source_filter(
        self, client_a: AsyncClient, app_pool: asyncpg.Pool, admin_pool: asyncpg.Pool, user_a: UUID
    ) -> None:
        """The source filter restricts to entries whose conversation has it."""
        async with admin_pool.acquire() as conn:
            topic_id = await _seed_topic(conn, _USER_A, "work/acme")
        linked = await _seed_entry(app_pool, _USER_A, "work/acme", "linked")
        await _seed_entry(app_pool, _USER_A, "work/acme", "unlinked")
        async with admin_pool.acquire() as conn:
            await _link_entry_source(conn, linked, _USER_A, topic_id, "extension_chatgpt")

        resp = await client_a.get(
            ENTRIES_ENDPOINT, params={"source": "extension_chatgpt"}, headers=_HDR_A
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["total"] == 1
        assert data["entries"][0]["content"] == "linked"

    async def test_sort_oldest_first(
        self, client_a: AsyncClient, app_pool: asyncpg.Pool, admin_pool: asyncpg.Pool, user_a: UUID
    ) -> None:
        """sort=oldest returns ascending by date; default is newest-first."""
        async with admin_pool.acquire() as conn:
            await _seed_topic(conn, _USER_A, "work/acme")
        await _seed_entry(app_pool, _USER_A, "work/acme", "first", date="2026-01-01")
        await _seed_entry(app_pool, _USER_A, "work/acme", "second", date="2026-02-01")

        oldest = await client_a.get(ENTRIES_ENDPOINT, params={"sort": "oldest"}, headers=_HDR_A)
        assert [e["content"] for e in oldest.json()["entries"]] == ["first", "second"]
        newest = await client_a.get(ENTRIES_ENDPOINT, params={"sort": "newest"}, headers=_HDR_A)
        assert [e["content"] for e in newest.json()["entries"]] == ["second", "first"]

    async def test_total_reflects_full_set_under_limit(
        self, client_a: AsyncClient, app_pool: asyncpg.Pool, admin_pool: asyncpg.Pool, user_a: UUID
    ) -> None:
        """total is the full filtered count; a small limit pages the list."""
        async with admin_pool.acquire() as conn:
            await _seed_topic(conn, _USER_A, "work/acme")
        for i in range(5):
            await _seed_entry(app_pool, _USER_A, "work/acme", f"e{i}")

        resp = await client_a.get(
            ENTRIES_ENDPOINT, params={"limit": 2, "offset": 0}, headers=_HDR_A
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["total"] == 5
        assert len(data["entries"]) == 2

    async def test_cache_control_private_no_store(self, client_a: AsyncClient) -> None:
        """List response carries Cache-Control: private, no-store."""
        resp = await client_a.get(ENTRIES_ENDPOINT, headers=_HDR_A)
        assert resp.status_code == 200, resp.text
        cache = resp.headers.get("cache-control", "")
        assert "private" in cache
        assert "no-store" in cache

    async def test_decryption_failure_surfaces_sentinel(
        self, client_a: AsyncClient, app_pool: asyncpg.Pool, admin_pool: asyncpg.Pool, user_a: UUID
    ) -> None:
        """A corrupt row yields the sentinel + decryption_failed, still 200."""
        async with admin_pool.acquire() as conn:
            await _seed_topic(conn, _USER_A, "work/acme")
        entry_id = await _seed_entry(app_pool, _USER_A, "work/acme", "secret")
        async with admin_pool.acquire() as conn:
            await _corrupt_entry_content(conn, entry_id)

        resp = await client_a.get(ENTRIES_ENDPOINT, headers=_HDR_A)
        assert resp.status_code == 200, resp.text
        item = resp.json()["entries"][0]
        assert item["decryption_failed"] is True
        assert item["content"] == "[decryption failed]"


# ---------------------------------------------------------------------------
# Detail (DB-backed)
# ---------------------------------------------------------------------------


class TestWebEntriesDetail:
    """Detail endpoint: includes reasoning; 404 on miss."""

    async def test_detail_includes_reasoning(
        self, client_a: AsyncClient, app_pool: asyncpg.Pool, admin_pool: asyncpg.Pool, user_a: UUID
    ) -> None:
        """The detail shape adds decrypted reasoning to the list-item fields."""
        async with admin_pool.acquire() as conn:
            await _seed_topic(conn, _USER_A, "work/acme")
        entry_id = await _seed_entry(
            app_pool, _USER_A, "work/acme", "content here", reasoning="my reasoning"
        )

        resp = await client_a.get(f"{ENTRIES_ENDPOINT}/{entry_id}", headers=_HDR_A)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["content"] == "content here"
        assert data["reasoning"] == "my reasoning"
        assert "private" in resp.headers.get("cache-control", "")

    async def test_detail_reasoning_null_when_absent(
        self, client_a: AsyncClient, app_pool: asyncpg.Pool, admin_pool: asyncpg.Pool, user_a: UUID
    ) -> None:
        """An entry with no reasoning surfaces reasoning: null."""
        async with admin_pool.acquire() as conn:
            await _seed_topic(conn, _USER_A, "work/acme")
        entry_id = await _seed_entry(app_pool, _USER_A, "work/acme", "no reasoning")

        resp = await client_a.get(f"{ENTRIES_ENDPOINT}/{entry_id}", headers=_HDR_A)
        assert resp.status_code == 200, resp.text
        assert resp.json()["reasoning"] is None

    async def test_missing_entry_returns_404(self, client_a: AsyncClient) -> None:
        """Absent id -> 404 {"detail": "entry_not_found"}."""
        resp = await client_a.get(f"{ENTRIES_ENDPOINT}/99999999", headers=_HDR_A)
        assert resp.status_code == 404
        assert resp.json() == {"detail": "entry_not_found"}


# ---------------------------------------------------------------------------
# PATCH (DB-backed)
# ---------------------------------------------------------------------------


class TestWebEntriesUpdate:
    """PATCH: replace, empty-body guard, topic move, miss."""

    async def test_patch_replaces_content(
        self, client_a: AsyncClient, app_pool: asyncpg.Pool, admin_pool: asyncpg.Pool, user_a: UUID
    ) -> None:
        """PATCH content overwrites (replace) and returns the detail shape."""
        async with admin_pool.acquire() as conn:
            await _seed_topic(conn, _USER_A, "work/acme")
        entry_id = await _seed_entry(app_pool, _USER_A, "work/acme", "original")

        resp = await client_a.patch(
            f"{ENTRIES_ENDPOINT}/{entry_id}", json={"content": "rewritten"}, headers=_HDR_A
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["content"] == "rewritten"
        assert "reasoning" in data

    async def test_patch_replaces_tags(
        self, client_a: AsyncClient, app_pool: asyncpg.Pool, admin_pool: asyncpg.Pool, user_a: UUID
    ) -> None:
        """PATCH tags replaces the full tag set (date/tags-only path)."""
        async with admin_pool.acquire() as conn:
            await _seed_topic(conn, _USER_A, "work/acme")
        entry_id = await _seed_entry(app_pool, _USER_A, "work/acme", "x", tags=["old"])

        resp = await client_a.patch(
            f"{ENTRIES_ENDPOINT}/{entry_id}", json={"tags": ["new"]}, headers=_HDR_A
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["tags"] == ["new"]

    async def test_patch_empty_body_returns_422(
        self, client_a: AsyncClient, app_pool: asyncpg.Pool, admin_pool: asyncpg.Pool, user_a: UUID
    ) -> None:
        """No fields supplied -> 422 no_fields_to_update."""
        async with admin_pool.acquire() as conn:
            await _seed_topic(conn, _USER_A, "work/acme")
        entry_id = await _seed_entry(app_pool, _USER_A, "work/acme", "x")

        resp = await client_a.patch(f"{ENTRIES_ENDPOINT}/{entry_id}", json={}, headers=_HDR_A)
        assert resp.status_code == 422
        assert resp.json() == {"detail": "no_fields_to_update"}

    async def test_patch_topic_path_moves_entry(
        self, client_a: AsyncClient, app_pool: asyncpg.Pool, admin_pool: asyncpg.Pool, user_a: UUID
    ) -> None:
        """PATCH topic_path moves the entry and reports the new path."""
        async with admin_pool.acquire() as conn:
            await _seed_topic(conn, _USER_A, "work/acme")
            await _seed_topic(conn, _USER_A, "work/dest")
        entry_id = await _seed_entry(app_pool, _USER_A, "work/acme", "movable")

        resp = await client_a.patch(
            f"{ENTRIES_ENDPOINT}/{entry_id}", json={"topic_path": "work/dest"}, headers=_HDR_A
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["topic_path"] == "work/dest"

    async def test_patch_missing_entry_returns_404(self, client_a: AsyncClient) -> None:
        """PATCH an absent id -> 404 entry_not_found."""
        resp = await client_a.patch(
            f"{ENTRIES_ENDPOINT}/99999999", json={"content": "x"}, headers=_HDR_A
        )
        assert resp.status_code == 404
        assert resp.json() == {"detail": "entry_not_found"}


# ---------------------------------------------------------------------------
# DELETE (DB-backed)
# ---------------------------------------------------------------------------


class TestWebEntriesDelete:
    """DELETE: soft-delete then idempotent 404."""

    async def test_delete_then_idempotent_404(
        self, client_a: AsyncClient, app_pool: asyncpg.Pool, admin_pool: asyncpg.Pool, user_a: UUID
    ) -> None:
        """First delete -> 204; the entry is gone from list; second -> 404."""
        async with admin_pool.acquire() as conn:
            await _seed_topic(conn, _USER_A, "work/acme")
        entry_id = await _seed_entry(app_pool, _USER_A, "work/acme", "doomed")

        first = await client_a.delete(f"{ENTRIES_ENDPOINT}/{entry_id}", headers=_HDR_A)
        assert first.status_code == 204

        listing = await client_a.get(ENTRIES_ENDPOINT, headers=_HDR_A)
        assert listing.json()["total"] == 0

        second = await client_a.delete(f"{ENTRIES_ENDPOINT}/{entry_id}", headers=_HDR_A)
        assert second.status_code == 404
        assert second.json() == {"detail": "entry_not_found"}


# ---------------------------------------------------------------------------
# Move (DB-backed)
# ---------------------------------------------------------------------------


class TestWebEntriesMove:
    """Bulk move: happy path + audit, missing id, destination miss."""

    async def test_move_happy_path_writes_audit(
        self, client_a: AsyncClient, app_pool: asyncpg.Pool, admin_pool: asyncpg.Pool, user_a: UUID
    ) -> None:
        """Move reassigns the entries and writes one entry.moved audit row."""
        async with admin_pool.acquire() as conn:
            await _seed_topic(conn, _USER_A, "work/acme")
            dest_id = await _seed_topic(conn, _USER_A, "work/dest")
        e1 = await _seed_entry(app_pool, _USER_A, "work/acme", "one")
        e2 = await _seed_entry(app_pool, _USER_A, "work/acme", "two")

        resp = await client_a.post(
            MOVE_ENDPOINT, json={"entry_ids": [e1, e2], "topic_id": dest_id}, headers=_HDR_A
        )
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"entries_moved": 2}

        moved = await client_a.get(ENTRIES_ENDPOINT, params={"topic": "work/dest"}, headers=_HDR_A)
        assert moved.json()["total"] == 2

        async with admin_pool.acquire() as conn:
            audits = await _audit_rows(conn, "entry.moved")
        assert len(audits) >= 1
        latest = audits[0]
        assert latest["target_kind"] == "topic"
        assert latest["target_id"] == str(dest_id)
        assert latest["actor_id"] == str(_USER_A)
        meta = latest["metadata"]
        meta = meta if isinstance(meta, dict) else json.loads(meta)
        assert meta["dest_path"] == "work/dest"
        assert meta["entries_moved"] == 2

    async def test_move_missing_id_returns_404(
        self, client_a: AsyncClient, app_pool: asyncpg.Pool, admin_pool: asyncpg.Pool, user_a: UUID
    ) -> None:
        """Any unresolved id -> 404 listing the missing ids; nothing moves."""
        async with admin_pool.acquire() as conn:
            await _seed_topic(conn, _USER_A, "work/acme")
            dest_id = await _seed_topic(conn, _USER_A, "work/dest")
        e1 = await _seed_entry(app_pool, _USER_A, "work/acme", "real")

        resp = await client_a.post(
            MOVE_ENDPOINT, json={"entry_ids": [e1, 99999999], "topic_id": dest_id}, headers=_HDR_A
        )
        assert resp.status_code == 404
        detail = resp.json()["detail"]
        assert detail["code"] == "entries_not_found"
        assert 99999999 in detail["missing_ids"]

        # All-or-nothing: the real entry stayed put.
        still = await client_a.get(ENTRIES_ENDPOINT, params={"topic": "work/acme"}, headers=_HDR_A)
        assert still.json()["total"] == 1

    async def test_move_missing_destination_returns_404(
        self, client_a: AsyncClient, app_pool: asyncpg.Pool, admin_pool: asyncpg.Pool, user_a: UUID
    ) -> None:
        """A destination topic that does not resolve -> 404 topic_not_found."""
        async with admin_pool.acquire() as conn:
            await _seed_topic(conn, _USER_A, "work/acme")
        e1 = await _seed_entry(app_pool, _USER_A, "work/acme", "x")

        resp = await client_a.post(
            MOVE_ENDPOINT, json={"entry_ids": [e1], "topic_id": 99999999}, headers=_HDR_A
        )
        assert resp.status_code == 404
        assert resp.json() == {"detail": "topic_not_found"}


# ---------------------------------------------------------------------------
# RLS isolation (DB-backed)
# ---------------------------------------------------------------------------


class TestWebEntriesRLS:
    """User B cannot read or mutate user A's entries."""

    async def test_user_b_list_excludes_user_a(
        self,
        client_a: AsyncClient,
        client_b: AsyncClient,
        app_pool: asyncpg.Pool,
        admin_pool: asyncpg.Pool,
        user_a: UUID,
        user_b: UUID,
    ) -> None:
        """User A's entries are invisible to user B's list."""
        async with admin_pool.acquire() as conn:
            await _seed_topic(conn, _USER_A, "work/acme")
        await _seed_entry(app_pool, _USER_A, "work/acme", "private")

        resp_a = await client_a.get(ENTRIES_ENDPOINT, headers=_HDR_A)
        assert resp_a.json()["total"] == 1
        resp_b = await client_b.get(ENTRIES_ENDPOINT, headers=_HDR_B)
        assert resp_b.json() == {"entries": [], "total": 0, "limit": 20, "offset": 0}

    async def test_user_b_detail_is_404(
        self,
        client_a: AsyncClient,
        client_b: AsyncClient,
        app_pool: asyncpg.Pool,
        admin_pool: asyncpg.Pool,
        user_a: UUID,
        user_b: UUID,
    ) -> None:
        """User A's entry detail is a 404 for user B (no cross-tenant signal)."""
        async with admin_pool.acquire() as conn:
            await _seed_topic(conn, _USER_A, "work/acme")
        entry_id = await _seed_entry(app_pool, _USER_A, "work/acme", "private")

        assert (
            await client_a.get(f"{ENTRIES_ENDPOINT}/{entry_id}", headers=_HDR_A)
        ).status_code == 200
        resp_b = await client_b.get(f"{ENTRIES_ENDPOINT}/{entry_id}", headers=_HDR_B)
        assert resp_b.status_code == 404

    async def test_user_b_cannot_patch_user_a(
        self,
        client_a: AsyncClient,
        client_b: AsyncClient,
        app_pool: asyncpg.Pool,
        admin_pool: asyncpg.Pool,
        user_a: UUID,
        user_b: UUID,
    ) -> None:
        """User B's PATCH of user A's entry -> 404 (RLS hides the row)."""
        async with admin_pool.acquire() as conn:
            await _seed_topic(conn, _USER_A, "work/acme")
        entry_id = await _seed_entry(app_pool, _USER_A, "work/acme", "private")

        resp_b = await client_b.patch(
            f"{ENTRIES_ENDPOINT}/{entry_id}", json={"content": "hacked"}, headers=_HDR_B
        )
        assert resp_b.status_code == 404

    async def test_user_b_cannot_delete_user_a(
        self,
        client_a: AsyncClient,
        client_b: AsyncClient,
        app_pool: asyncpg.Pool,
        admin_pool: asyncpg.Pool,
        user_a: UUID,
        user_b: UUID,
    ) -> None:
        """User B's DELETE of user A's entry -> 404; the entry survives."""
        async with admin_pool.acquire() as conn:
            await _seed_topic(conn, _USER_A, "work/acme")
        entry_id = await _seed_entry(app_pool, _USER_A, "work/acme", "private")

        resp_b = await client_b.delete(f"{ENTRIES_ENDPOINT}/{entry_id}", headers=_HDR_B)
        assert resp_b.status_code == 404
        assert (
            await client_a.get(f"{ENTRIES_ENDPOINT}/{entry_id}", headers=_HDR_A)
        ).status_code == 200

    async def test_user_b_cannot_move_user_a(
        self,
        client_a: AsyncClient,
        client_b: AsyncClient,
        app_pool: asyncpg.Pool,
        admin_pool: asyncpg.Pool,
        user_a: UUID,
        user_b: UUID,
    ) -> None:
        """User B moving user A's entry -> 404 (the id is invisible under RLS)."""
        async with admin_pool.acquire() as conn:
            await _seed_topic(conn, _USER_A, "work/acme")
            await _seed_topic(conn, _USER_B, "b/dest")
            dest_b = await conn.fetchval("SELECT id FROM topics WHERE path = 'b/dest'")
        entry_id = await _seed_entry(app_pool, _USER_A, "work/acme", "private")

        resp_b = await client_b.post(
            MOVE_ENDPOINT,
            json={"entry_ids": [entry_id], "topic_id": int(dest_b)},
            headers=_HDR_B,
        )
        assert resp_b.status_code == 404
