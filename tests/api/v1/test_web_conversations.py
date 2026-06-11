"""Tests for the web conversations endpoints.

``GET /api/v1/conversations``       -- paginated conversation list.
``GET /api/v1/conversations/{id}``  -- conversation meta + a page of messages.

Covers:
- Auth: missing token -> 401.
- Pagination: total correctness; over-max limit / messages_limit -> 422; caps.
- Response shape: spec field set, ``conversations`` envelope key, message page.
- Summary truncation in the list view.
- Cache-Control: ``private, no-store``.
- RLS isolation: user B sees an empty list and a 404 for user A's conversation.

The DB-backed tests require the RLS test database. They auto-skip when the
database is unreachable (the shared pool fixtures call ``pytest.skip``), so
``pytest tests/api/v1/test_web_conversations.py`` exits PASS in environments
without a running Postgres. The 422/401 tests need no DB -- FastAPI rejects
before the handler touches the pool.
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

from gubbi.api.v1.web.conversations import router as conversations_router
from gubbi.app_context import AppContext
from gubbi.auth.strategies import TrustGatewayStrategy
from gubbi.config import Settings
from gubbi.crypto.cipher import ContentCipher
from gubbi.storage.embedding_service import EmbeddingService

pytestmark = pytest.mark.asyncio(loop_scope="session")

API_PREFIX = "/api/v1"
LIST_ENDPOINT = f"{API_PREFIX}/conversations"

_USER_A = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
_USER_B = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")

# Single shared key so seeded ciphertext decrypts under the same cipher the app
# uses. ContentCipher({1: 32-byte key}) matches the topics-test harness.
_CIPHER = ContentCipher({1: bytes([1]) * 32})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app(pool: asyncpg.Pool, *, auth_user_id: UUID | None = None) -> FastAPI:
    """Minimal FastAPI with an AppContext backed by the given pool.

    ``auth_user_id`` sets operator_user_id so the X-Auth-User-Id gateway header
    authenticates as that user (matches the topics/extraction test harness: the
    auth strategy trusts the gateway header when trust_gateway=True).
    """
    settings = Settings(
        db={"app_url": ""},
        auth={
            "api_key": "test-api-key-for-unit-tests-only",
            "operator_email": "web-conversations-test@test.local",
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
    app.include_router(conversations_router, prefix=API_PREFIX)

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


async def _seed_conversation(
    admin_conn: asyncpg.Connection,
    user_id: UUID,
    topic_id: int,
    *,
    title: str,
    summary: str,
    source: str = "extension_chatgpt",
    messages: list[tuple[str, str]] | None = None,
) -> int:
    """INSERT a conversation with real ciphertext + messages; return its id.

    Title/summary/message content are encrypted with the shared test cipher so
    the repo's internal decrypt succeeds. ``messages`` is a list of
    ``(role, content)`` pairs inserted in order with sequential positions.
    """
    title_ct, title_nonce = _CIPHER.encrypt(title)
    summary_ct, summary_nonce = _CIPHER.encrypt(summary)
    msgs = messages or []
    conv_id = await admin_conn.fetchval(
        """
        INSERT INTO conversations
            (topic_id, user_id, title_encrypted, title_nonce, slug, source,
             summary_encrypted, summary_nonce, tags, participants,
             message_count, created_at, updated_at, json_path, search_vector,
             platform, platform_id)
        VALUES ($1, $2, $3, $4, gen_random_uuid()::text, $5,
                $6, $7, '{}', '{}', $8, now(), now(), 'test.json',
                to_tsvector('english', 'seed conversation'),
                'chatgpt', gen_random_uuid()::text)
        RETURNING id
        """,
        topic_id,
        user_id,
        title_ct,
        title_nonce,
        source,
        summary_ct,
        summary_nonce,
        len(msgs),
    )
    for position, (role, content) in enumerate(msgs):
        content_ct, content_nonce = _CIPHER.encrypt(content)
        await admin_conn.execute(
            """
            INSERT INTO messages
                (conversation_id, user_id, role, position, timestamp,
                 content_encrypted, content_nonce, search_vector)
            VALUES ($1, $2, $3, $4, now(), $5, $6,
                    to_tsvector('english', 'seed message'))
            """,
            int(conv_id),
            user_id,
            role,
            position,
            content_ct,
            content_nonce,
        )
    return int(conv_id)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def user_a(admin_pool: asyncpg.Pool) -> UUID:
    """Ensure user A exists; tear down conversations + topics after test."""
    async with admin_pool.acquire() as conn:
        await _seed_user(conn, _USER_A, "web-conv-a@test.local")
    yield _USER_A
    async with admin_pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM messages WHERE conversation_id IN"
            " (SELECT id FROM conversations WHERE user_id = $1)",
            _USER_A,
        )
        await conn.execute("DELETE FROM conversations WHERE user_id = $1", _USER_A)
        await conn.execute("DELETE FROM topics WHERE user_id = $1", _USER_A)
        await conn.execute("DELETE FROM users WHERE id = $1", _USER_A)


@pytest_asyncio.fixture
async def user_b(admin_pool: asyncpg.Pool) -> UUID:
    """Ensure user B exists; tear down conversations + topics after test."""
    async with admin_pool.acquire() as conn:
        await _seed_user(conn, _USER_B, "web-conv-b@test.local")
    yield _USER_B
    async with admin_pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM messages WHERE conversation_id IN"
            " (SELECT id FROM conversations WHERE user_id = $1)",
            _USER_B,
        )
        await conn.execute("DELETE FROM conversations WHERE user_id = $1", _USER_B)
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


class TestWebConversationsValidation:
    """Param-validation and auth rejection (pre-handler)."""

    async def test_missing_token_returns_401(self, app_pool: asyncpg.Pool) -> None:
        """No credentials and trust_gateway=False -> 401."""
        settings = Settings(
            db={"app_url": ""},
            auth={
                "api_key": "test-api-key-for-unit-tests-only",
                "operator_email": "web-conversations-test@test.local",
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
        app.include_router(conversations_router, prefix=API_PREFIX)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(LIST_ENDPOINT)
        assert resp.status_code == 401

    async def test_limit_over_max_returns_422(self, app_pool: asyncpg.Pool) -> None:
        """limit above the 100 cap -> 422 (Pydantic Query constraint)."""
        app = _make_app(app_pool, auth_user_id=_USER_A)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(
                LIST_ENDPOINT,
                params={"limit": 101},
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

    async def test_messages_limit_over_max_returns_422(self, app_pool: asyncpg.Pool) -> None:
        """messages_limit above the 200 cap -> 422 on the detail endpoint."""
        app = _make_app(app_pool, auth_user_id=_USER_A)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(
                f"{LIST_ENDPOINT}/1",
                params={"messages_limit": 201},
                headers={"X-Auth-User-Id": str(_USER_A)},
            )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# DB-backed tests (require the RLS test database)
# ---------------------------------------------------------------------------


class TestWebConversationsList:
    """List shape, pagination, truncation, and Cache-Control."""

    async def test_empty_list(self, client_a: AsyncClient) -> None:
        """No conversations -> 200 with empty list and total 0."""
        resp = await client_a.get(LIST_ENDPOINT, headers={"X-Auth-User-Id": str(_USER_A)})
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"conversations": [], "total": 0, "limit": 20, "offset": 0}

    async def test_shape_and_fields(
        self,
        client_a: AsyncClient,
        admin_pool: asyncpg.Pool,
        user_a: UUID,
    ) -> None:
        """Seeded conversation surfaces all spec fields under the envelope."""
        async with admin_pool.acquire() as conn:
            topic_id = await _seed_topic(conn, _USER_A, "work/acme")
            await _seed_conversation(
                conn, _USER_A, topic_id, title="Planning", summary="A short summary."
            )

        resp = await client_a.get(LIST_ENDPOINT, headers={"X-Auth-User-Id": str(_USER_A)})
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["total"] == 1
        assert data["limit"] == 20
        assert data["offset"] == 0
        item = data["conversations"][0]
        assert set(item) == {
            "id",
            "topic_path",
            "title",
            "summary",
            "source",
            "platform",
            "message_count",
            "created_at",
            "updated_at",
            "decryption_failed",
        }
        assert item["topic_path"] == "work/acme"
        assert item["title"] == "Planning"
        assert item["summary"] == "A short summary."
        assert item["decryption_failed"] is False
        assert isinstance(item["id"], int)

    async def test_summary_truncated_to_preview(
        self,
        client_a: AsyncClient,
        admin_pool: asyncpg.Pool,
        user_a: UUID,
    ) -> None:
        """A long summary is truncated to the 280-char preview in the list view."""
        long_summary = "x" * 500
        async with admin_pool.acquire() as conn:
            topic_id = await _seed_topic(conn, _USER_A, "work/acme")
            await _seed_conversation(conn, _USER_A, topic_id, title="Long", summary=long_summary)

        resp = await client_a.get(LIST_ENDPOINT, headers={"X-Auth-User-Id": str(_USER_A)})
        assert resp.status_code == 200, resp.text
        item = resp.json()["conversations"][0]
        assert len(item["summary"]) == 280

    async def test_total_reflects_full_set_under_limit(
        self,
        client_a: AsyncClient,
        admin_pool: asyncpg.Pool,
        user_a: UUID,
    ) -> None:
        """total is the full filtered count; a small limit pages the list."""
        async with admin_pool.acquire() as conn:
            topic_id = await _seed_topic(conn, _USER_A, "work/acme")
            for i in range(5):
                await _seed_conversation(conn, _USER_A, topic_id, title=f"Conv {i}", summary="s")

        resp = await client_a.get(
            LIST_ENDPOINT,
            params={"limit": 2, "offset": 0},
            headers={"X-Auth-User-Id": str(_USER_A)},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["total"] == 5
        assert data["limit"] == 2
        assert len(data["conversations"]) == 2

    async def test_topic_prefix_filter(
        self,
        client_a: AsyncClient,
        admin_pool: asyncpg.Pool,
        user_a: UUID,
    ) -> None:
        """topic_prefix narrows the list and the total."""
        async with admin_pool.acquire() as conn:
            work = await _seed_topic(conn, _USER_A, "work/acme")
            health = await _seed_topic(conn, _USER_A, "health")
            await _seed_conversation(conn, _USER_A, work, title="W", summary="s")
            await _seed_conversation(conn, _USER_A, health, title="H", summary="s")

        resp = await client_a.get(
            LIST_ENDPOINT,
            params={"topic_prefix": "work"},
            headers={"X-Auth-User-Id": str(_USER_A)},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["total"] == 1
        assert data["conversations"][0]["topic_path"] == "work/acme"

    async def test_cache_control_private_no_store(self, client_a: AsyncClient) -> None:
        """List response carries Cache-Control: private, no-store."""
        resp = await client_a.get(LIST_ENDPOINT, headers={"X-Auth-User-Id": str(_USER_A)})
        assert resp.status_code == 200, resp.text
        cache_header = resp.headers.get("cache-control", "")
        assert "private" in cache_header
        assert "no-store" in cache_header


class TestWebConversationsDetail:
    """Detail endpoint: hit, message paging, miss, and Cache-Control."""

    async def test_get_existing_conversation(
        self,
        client_a: AsyncClient,
        admin_pool: asyncpg.Pool,
        user_a: UUID,
    ) -> None:
        """Existing id -> 200 with meta + decrypted message page."""
        async with admin_pool.acquire() as conn:
            topic_id = await _seed_topic(conn, _USER_A, "work/acme")
            conv_id = await _seed_conversation(
                conn,
                _USER_A,
                topic_id,
                title="Chat",
                summary="full summary text",
                messages=[("user", "hello"), ("assistant", "hi there")],
            )

        resp = await client_a.get(
            f"{LIST_ENDPOINT}/{conv_id}",
            headers={"X-Auth-User-Id": str(_USER_A)},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert set(data) == {
            "conversation",
            "messages",
            "messages_total",
            "messages_limit",
            "messages_offset",
        }
        assert data["conversation"]["title"] == "Chat"
        assert data["conversation"]["summary"] == "full summary text"
        assert data["messages_total"] == 2
        assert data["messages_limit"] == 50
        assert data["messages_offset"] == 0
        assert data["messages"][0] == {
            "role": "user",
            "content": "hello",
            "timestamp": data["messages"][0]["timestamp"],
            "position": 0,
            "decryption_failed": False,
        }
        assert data["messages"][1]["content"] == "hi there"
        assert data["messages"][1]["position"] == 1
        assert "private" in resp.headers.get("cache-control", "")
        assert "no-store" in resp.headers.get("cache-control", "")

    async def test_message_paging_offset_position(
        self,
        client_a: AsyncClient,
        admin_pool: asyncpg.Pool,
        user_a: UUID,
    ) -> None:
        """messages_offset pages the transcript; position reflects the offset."""
        async with admin_pool.acquire() as conn:
            topic_id = await _seed_topic(conn, _USER_A, "work/acme")
            conv_id = await _seed_conversation(
                conn,
                _USER_A,
                topic_id,
                title="Chat",
                summary="s",
                messages=[("user", f"m{i}") for i in range(5)],
            )

        resp = await client_a.get(
            f"{LIST_ENDPOINT}/{conv_id}",
            params={"messages_limit": 2, "messages_offset": 2},
            headers={"X-Auth-User-Id": str(_USER_A)},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["messages_total"] == 5
        assert [m["position"] for m in data["messages"]] == [2, 3]
        assert [m["content"] for m in data["messages"]] == ["m2", "m3"]

    async def test_missing_conversation_returns_404(self, client_a: AsyncClient) -> None:
        """Absent id -> 404 {"detail": "conversation_not_found"}."""
        resp = await client_a.get(
            f"{LIST_ENDPOINT}/99999999",
            headers={"X-Auth-User-Id": str(_USER_A)},
        )
        assert resp.status_code == 404
        assert resp.json() == {"detail": "conversation_not_found"}


class TestWebConversationsRLS:
    """RLS isolation: user B cannot see user A's conversations."""

    async def test_user_b_list_excludes_user_a_conversations(
        self,
        client_a: AsyncClient,
        client_b: AsyncClient,
        admin_pool: asyncpg.Pool,
        user_a: UUID,
        user_b: UUID,
    ) -> None:
        """User A's conversations are invisible to user B's list."""
        async with admin_pool.acquire() as conn:
            topic_id = await _seed_topic(conn, _USER_A, "work/acme")
            await _seed_conversation(conn, _USER_A, topic_id, title="A", summary="s")

        resp_a = await client_a.get(LIST_ENDPOINT, headers={"X-Auth-User-Id": str(_USER_A)})
        assert resp_a.status_code == 200
        assert resp_a.json()["total"] == 1

        resp_b = await client_b.get(LIST_ENDPOINT, headers={"X-Auth-User-Id": str(_USER_B)})
        assert resp_b.status_code == 200
        data_b = resp_b.json()
        assert data_b["total"] == 0
        assert data_b["conversations"] == []

    async def test_user_b_detail_is_404_for_user_a_conversation(
        self,
        client_a: AsyncClient,
        client_b: AsyncClient,
        admin_pool: asyncpg.Pool,
        user_a: UUID,
        user_b: UUID,
    ) -> None:
        """User A's conversation detail is a 404 for user B (no cross-tenant signal)."""
        async with admin_pool.acquire() as conn:
            topic_id = await _seed_topic(conn, _USER_A, "work/acme")
            conv_id = await _seed_conversation(conn, _USER_A, topic_id, title="A", summary="s")

        resp_a = await client_a.get(
            f"{LIST_ENDPOINT}/{conv_id}",
            headers={"X-Auth-User-Id": str(_USER_A)},
        )
        assert resp_a.status_code == 200

        resp_b = await client_b.get(
            f"{LIST_ENDPOINT}/{conv_id}",
            headers={"X-Auth-User-Id": str(_USER_B)},
        )
        assert resp_b.status_code == 404
        assert resp_b.json() == {"detail": "conversation_not_found"}
