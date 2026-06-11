"""Tests for the web topic-management write endpoints.

``POST   /api/v1/topics/{topic_id}/rename``
``DELETE /api/v1/topics/{topic_id}?move_entries_to={dest}``
``POST   /api/v1/topics/{topic_id}/merge``

Covers:
- Auth: missing token -> 401 (write scope enforced by the gateway strategy).
- Rename: happy path; 409 on a path collision.
- Delete-with-reassign: moves entries + conversations, then removes the topic;
  400 when entries exist and no destination is given.
- Merge: moves everything, deletes the source, guards a self-merge.
- Audit: a row with the expected action + attributes per mutation.
- RLS isolation: user B cannot rename/merge/delete user A's topics (404).
- Atomicity: a forced mid-merge failure leaves no partial move.

The DB-backed tests require the RLS test database. They auto-skip when the
database is unreachable (the shared pool fixtures call ``pytest.skip``), so this
module exits PASS in environments without a running Postgres. The 401 test needs
no DB -- the auth strategy rejects before the handler touches the pool.
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
from httpx import ASGITransport, AsyncClient

from gubbi.api.v1.web.topic_admin import router as topic_admin_router
from gubbi.app_context import AppContext
from gubbi.auth.strategies import TrustGatewayStrategy
from gubbi.config import Settings
from gubbi.crypto.cipher import ContentCipher
from gubbi.storage.embedding_service import EmbeddingService

pytestmark = pytest.mark.asyncio(loop_scope="session")

API_PREFIX = "/api/v1"
TOPICS_ENDPOINT = f"{API_PREFIX}/topics"

_USER_A = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
_USER_B = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app(pool: asyncpg.Pool, *, auth_user_id: UUID | None = None) -> FastAPI:
    """Minimal FastAPI with an AppContext backed by the given pool.

    ``auth_user_id`` sets operator_user_id so the X-Auth-User-Id gateway header
    authenticates as that user. With no X-Auth-Scopes header the gateway
    strategy grants ``journal:read journal:write`` (the default), so write
    endpoints authorize.
    """
    settings = Settings(
        db={"app_url": ""},
        auth={
            "api_key": "test-api-key-for-unit-tests-only",
            "operator_email": "web-topic-admin-test@test.local",
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
    app.include_router(topic_admin_router, prefix=API_PREFIX)

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


async def _seed_entry(admin_conn: asyncpg.Connection, topic_id: int, content: str = "note") -> int:
    """INSERT an active entry under a topic; return entry_id."""
    entry_id = await admin_conn.fetchval(
        """
        INSERT INTO entries (topic_id, date, content, created_at, updated_at)
        VALUES ($1, CURRENT_DATE, $2, now(), now())
        RETURNING id
        """,
        topic_id,
        content,
    )
    return int(entry_id)


async def _seed_conversation(
    admin_conn: asyncpg.Connection,
    topic_id: int,
    slug: str,
) -> int:
    """INSERT a conversation under a topic; return conversation_id."""
    conv_id = await admin_conn.fetchval(
        """
        INSERT INTO conversations (topic_id, title, slug, source, created_at, updated_at)
        VALUES ($1, $2, $3, 'claude', now(), now())
        RETURNING id
        """,
        topic_id,
        f"conv {slug}",
        slug,
    )
    return int(conv_id)


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


@pytest_asyncio.fixture
async def user_a(admin_pool: asyncpg.Pool) -> UUID:
    """Ensure user A exists; tear down after test."""
    async with admin_pool.acquire() as conn:
        await _seed_user(conn, _USER_A, "web-topic-admin-a@test.local")
    yield _USER_A
    async with admin_pool.acquire() as conn:
        await conn.execute("DELETE FROM users WHERE id = $1", _USER_A)


@pytest_asyncio.fixture
async def user_b(admin_pool: asyncpg.Pool) -> UUID:
    """Ensure user B exists; tear down after test."""
    async with admin_pool.acquire() as conn:
        await _seed_user(conn, _USER_B, "web-topic-admin-b@test.local")
    yield _USER_B
    async with admin_pool.acquire() as conn:
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


_HDR_A = {"X-Auth-User-Id": str(_USER_A)}
_HDR_B = {"X-Auth-User-Id": str(_USER_B)}


# ---------------------------------------------------------------------------
# Auth (no DB needed)
# ---------------------------------------------------------------------------


class TestWebTopicAdminAuth:
    """Write endpoints reject unauthenticated requests."""

    async def test_rename_missing_token_returns_401(self, app_pool: asyncpg.Pool) -> None:
        """No credentials and trust_gateway=False -> 401."""
        settings = Settings(
            db={"app_url": ""},
            auth={
                "api_key": "test-api-key-for-unit-tests-only",
                "operator_email": "web-topic-admin-test@test.local",
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
        app.include_router(topic_admin_router, prefix=API_PREFIX)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(f"{TOPICS_ENDPOINT}/1/rename", json={"path": "new"})
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Rename
# ---------------------------------------------------------------------------


class TestWebTopicRename:
    """Rename happy path, conflict, and audit."""

    async def test_rename_happy_path(
        self,
        client_a: AsyncClient,
        admin_pool: asyncpg.Pool,
        user_a: UUID,
    ) -> None:
        """Rename updates the path and writes a topic.renamed audit row."""
        async with admin_pool.acquire() as conn:
            tid = await _seed_topic(conn, _USER_A, "work/old")

        resp = await client_a.post(
            f"{TOPICS_ENDPOINT}/{tid}/rename",
            json={"path": "work/new", "title": "Renamed"},
            headers=_HDR_A,
        )
        assert resp.status_code == 200, resp.text

        async with admin_pool.acquire() as conn:
            row = await conn.fetchrow("SELECT path, title FROM topics WHERE id = $1", tid)
            audits = await _audit_rows(conn, "topic.renamed")
        assert row["path"] == "work/new"
        assert row["title"] == "Renamed"
        assert len(audits) == 1
        assert audits[0]["target_kind"] == "topic"
        assert audits[0]["target_id"] == str(tid)
        assert audits[0]["actor_id"] == str(_USER_A)
        meta = audits[0]["metadata"]
        meta = meta if isinstance(meta, dict) else json.loads(meta)
        assert meta["old_path"] == "work/old"
        assert meta["new_path"] == "work/new"

    async def test_rename_path_collision_returns_409(
        self,
        client_a: AsyncClient,
        admin_pool: asyncpg.Pool,
        user_a: UUID,
    ) -> None:
        """Renaming onto an existing path -> 409 topic_path_exists."""
        async with admin_pool.acquire() as conn:
            await _seed_topic(conn, _USER_A, "work/taken")
            tid = await _seed_topic(conn, _USER_A, "work/source")

        resp = await client_a.post(
            f"{TOPICS_ENDPOINT}/{tid}/rename",
            json={"path": "work/taken"},
            headers=_HDR_A,
        )
        assert resp.status_code == 409
        assert resp.json() == {"detail": "topic_path_exists"}

    async def test_rename_missing_topic_returns_404(self, client_a: AsyncClient) -> None:
        """Renaming an absent topic -> 404 topic_not_found."""
        resp = await client_a.post(
            f"{TOPICS_ENDPOINT}/99999/rename",
            json={"path": "work/x"},
            headers=_HDR_A,
        )
        assert resp.status_code == 404
        assert resp.json() == {"detail": "topic_not_found"}


# ---------------------------------------------------------------------------
# Delete-with-reassign
# ---------------------------------------------------------------------------


class TestWebTopicDelete:
    """Delete moves rows then removes the topic; guards a missing destination."""

    async def test_delete_reassigns_then_removes(
        self,
        client_a: AsyncClient,
        admin_pool: asyncpg.Pool,
        user_a: UUID,
    ) -> None:
        """Entries + conversations move to the destination; source topic gone."""
        async with admin_pool.acquire() as conn:
            src = await _seed_topic(conn, _USER_A, "work/src")
            dest = await _seed_topic(conn, _USER_A, "work/dest")
            e1 = await _seed_entry(conn, src)
            await _seed_conversation(conn, src, "chat-1")

        resp = await client_a.delete(
            f"{TOPICS_ENDPOINT}/{src}",
            params={"move_entries_to": dest},
            headers=_HDR_A,
        )
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"entries_moved": 1, "conversations_moved": 1}

        async with admin_pool.acquire() as conn:
            src_exists = await conn.fetchval("SELECT 1 FROM topics WHERE id = $1", src)
            entry_topic = await conn.fetchval("SELECT topic_id FROM entries WHERE id = $1", e1)
            conv_topic = await conn.fetchval(
                "SELECT topic_id FROM conversations WHERE topic_id = $1", dest
            )
            audits = await _audit_rows(conn, "topic.deleted")
        assert src_exists is None
        assert entry_topic == dest
        assert conv_topic == dest
        assert len(audits) == 1
        meta = audits[0]["metadata"]
        meta = meta if isinstance(meta, dict) else json.loads(meta)
        assert meta["entries_moved"] == 1
        assert meta["conversations_moved"] == 1
        assert meta["dest_path"] == "work/dest"

    async def test_delete_empty_topic_no_destination(
        self,
        client_a: AsyncClient,
        admin_pool: asyncpg.Pool,
        user_a: UUID,
    ) -> None:
        """An empty topic deletes without a destination."""
        async with admin_pool.acquire() as conn:
            tid = await _seed_topic(conn, _USER_A, "work/empty")

        resp = await client_a.delete(f"{TOPICS_ENDPOINT}/{tid}", headers=_HDR_A)
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"entries_moved": 0, "conversations_moved": 0}

        async with admin_pool.acquire() as conn:
            exists = await conn.fetchval("SELECT 1 FROM topics WHERE id = $1", tid)
        assert exists is None

    async def test_delete_with_entries_no_destination_returns_400(
        self,
        client_a: AsyncClient,
        admin_pool: asyncpg.Pool,
        user_a: UUID,
    ) -> None:
        """Entries present but no destination -> 400 stating the count."""
        async with admin_pool.acquire() as conn:
            tid = await _seed_topic(conn, _USER_A, "work/full")
            await _seed_entry(conn, tid)
            await _seed_entry(conn, tid)

        resp = await client_a.delete(f"{TOPICS_ENDPOINT}/{tid}", headers=_HDR_A)
        assert resp.status_code == 400
        assert "2" in resp.json()["detail"]

        async with admin_pool.acquire() as conn:
            still_there = await conn.fetchval("SELECT 1 FROM topics WHERE id = $1", tid)
        assert still_there == 1

    async def test_delete_missing_destination_returns_404(
        self,
        client_a: AsyncClient,
        admin_pool: asyncpg.Pool,
        user_a: UUID,
    ) -> None:
        """A nonexistent destination topic -> 404."""
        async with admin_pool.acquire() as conn:
            src = await _seed_topic(conn, _USER_A, "work/src")
            await _seed_entry(conn, src)

        resp = await client_a.delete(
            f"{TOPICS_ENDPOINT}/{src}",
            params={"move_entries_to": 99999},
            headers=_HDR_A,
        )
        assert resp.status_code == 404
        assert resp.json() == {"detail": "topic_not_found"}


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


class TestWebTopicMerge:
    """Merge moves everything, deletes the source, guards self-merge."""

    async def test_merge_moves_and_deletes_source(
        self,
        client_a: AsyncClient,
        admin_pool: asyncpg.Pool,
        user_a: UUID,
    ) -> None:
        """All rows land in the destination; source removed; audit written."""
        async with admin_pool.acquire() as conn:
            src = await _seed_topic(conn, _USER_A, "work/src")
            dest = await _seed_topic(conn, _USER_A, "work/dest")
            await _seed_entry(conn, src)
            await _seed_entry(conn, src)
            await _seed_conversation(conn, src, "chat-1")

        resp = await client_a.post(
            f"{TOPICS_ENDPOINT}/{src}/merge",
            json={"into_topic_id": dest},
            headers=_HDR_A,
        )
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"entries_moved": 2, "conversations_moved": 1}

        async with admin_pool.acquire() as conn:
            src_exists = await conn.fetchval("SELECT 1 FROM topics WHERE id = $1", src)
            dest_entries = await conn.fetchval(
                "SELECT COUNT(*) FROM entries WHERE topic_id = $1", dest
            )
            audits = await _audit_rows(conn, "topic.merged")
        assert src_exists is None
        assert dest_entries == 2
        assert len(audits) == 1
        assert audits[0]["target_id"] == str(dest)
        meta = audits[0]["metadata"]
        meta = meta if isinstance(meta, dict) else json.loads(meta)
        assert meta["source_path"] == "work/src"
        assert meta["dest_path"] == "work/dest"
        assert meta["entries_moved"] == 2

    async def test_self_merge_returns_422(
        self,
        client_a: AsyncClient,
        admin_pool: asyncpg.Pool,
        user_a: UUID,
    ) -> None:
        """Merging a topic into itself -> 422; nothing changes."""
        async with admin_pool.acquire() as conn:
            tid = await _seed_topic(conn, _USER_A, "work/self")

        resp = await client_a.post(
            f"{TOPICS_ENDPOINT}/{tid}/merge",
            json={"into_topic_id": tid},
            headers=_HDR_A,
        )
        assert resp.status_code == 422

        async with admin_pool.acquire() as conn:
            still_there = await conn.fetchval("SELECT 1 FROM topics WHERE id = $1", tid)
        assert still_there == 1

    async def test_merge_slug_collision_is_atomic(
        self,
        client_a: AsyncClient,
        admin_pool: asyncpg.Pool,
        user_a: UUID,
    ) -> None:
        """A conversation slug collision -> 409 and no partial move (atomic).

        Source and destination each hold a conversation with the same slug, so
        moving the source conversation violates UNIQUE(topic_id, slug). The
        transaction must roll back: the source topic and its rows stay put.
        """
        async with admin_pool.acquire() as conn:
            src = await _seed_topic(conn, _USER_A, "work/src")
            dest = await _seed_topic(conn, _USER_A, "work/dest")
            await _seed_entry(conn, src)
            await _seed_conversation(conn, src, "dup")
            await _seed_conversation(conn, dest, "dup")

        resp = await client_a.post(
            f"{TOPICS_ENDPOINT}/{src}/merge",
            json={"into_topic_id": dest},
            headers=_HDR_A,
        )
        assert resp.status_code == 409
        assert resp.json() == {"detail": "topic_path_exists"}

        async with admin_pool.acquire() as conn:
            src_exists = await conn.fetchval("SELECT 1 FROM topics WHERE id = $1", src)
            src_entries = await conn.fetchval(
                "SELECT COUNT(*) FROM entries WHERE topic_id = $1", src
            )
            merged_audits = await _audit_rows(conn, "topic.merged")
        assert src_exists == 1
        assert src_entries == 1
        assert merged_audits == []


# ---------------------------------------------------------------------------
# RLS isolation
# ---------------------------------------------------------------------------


class TestWebTopicAdminRLS:
    """User B cannot mutate user A's topics (404, no cross-tenant signal)."""

    async def test_user_b_cannot_rename_user_a_topic(
        self,
        client_a: AsyncClient,
        client_b: AsyncClient,
        admin_pool: asyncpg.Pool,
        user_a: UUID,
        user_b: UUID,
    ) -> None:
        """User B renaming user A's topic -> 404; the topic is unchanged."""
        async with admin_pool.acquire() as conn:
            tid = await _seed_topic(conn, _USER_A, "work/a-owned")

        resp = await client_b.post(
            f"{TOPICS_ENDPOINT}/{tid}/rename",
            json={"path": "work/hijacked"},
            headers=_HDR_B,
        )
        assert resp.status_code == 404
        assert resp.json() == {"detail": "topic_not_found"}

        async with admin_pool.acquire() as conn:
            path = await conn.fetchval("SELECT path FROM topics WHERE id = $1", tid)
        assert path == "work/a-owned"

    async def test_user_b_cannot_delete_user_a_topic(
        self,
        client_a: AsyncClient,
        client_b: AsyncClient,
        admin_pool: asyncpg.Pool,
        user_a: UUID,
        user_b: UUID,
    ) -> None:
        """User B deleting user A's topic -> 404; the topic survives."""
        async with admin_pool.acquire() as conn:
            tid = await _seed_topic(conn, _USER_A, "work/a-owned")

        resp = await client_b.delete(f"{TOPICS_ENDPOINT}/{tid}", headers=_HDR_B)
        assert resp.status_code == 404
        assert resp.json() == {"detail": "topic_not_found"}

        async with admin_pool.acquire() as conn:
            exists = await conn.fetchval("SELECT 1 FROM topics WHERE id = $1", tid)
        assert exists == 1

    async def test_user_b_cannot_merge_user_a_topics(
        self,
        client_a: AsyncClient,
        client_b: AsyncClient,
        admin_pool: asyncpg.Pool,
        user_a: UUID,
        user_b: UUID,
    ) -> None:
        """User B merging user A's topics -> 404; both topics survive."""
        async with admin_pool.acquire() as conn:
            src = await _seed_topic(conn, _USER_A, "work/a-src")
            dest = await _seed_topic(conn, _USER_A, "work/a-dest")

        resp = await client_b.post(
            f"{TOPICS_ENDPOINT}/{src}/merge",
            json={"into_topic_id": dest},
            headers=_HDR_B,
        )
        assert resp.status_code == 404
        assert resp.json() == {"detail": "topic_not_found"}

        async with admin_pool.acquire() as conn:
            count = await conn.fetchval(
                "SELECT COUNT(*) FROM topics WHERE id = ANY($1::int[])", [src, dest]
            )
        assert count == 2
