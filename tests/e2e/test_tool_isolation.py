"""Tool-level cross-tenant isolation tests (TASK-03.01).

Verifies that MCP tool handlers enforce per-user RLS when called through
the app_pool (journal_app role, BYPASSRLS=False).  Each test sets the
current_user_id ContextVar to simulate what BearerAuthMiddleware does in
production, then asserts cross-tenant data is invisible.

Unlike tests/integration/test_rls_isolation.py (repository-level), these
tests exercise the full tool handler code path.

Fixtures (session-scoped):
  app_pool   -- journal_app role pool (RLS enforced)
  admin_pool -- journal_admin role pool (BYPASSRLS, seeding only)

The dual_users session fixture seeds one topic + entries per user.
Ad-hoc mutation tests (append, search) create topics with user-unique
paths so concurrent tests do not collide even without per-test truncation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

import asyncpg
import pytest
import pytest_asyncio
import structlog
from mcp.server.fastmcp import FastMCP

from journalctl.config import get_settings
from journalctl.core.auth_context import current_user_id
from journalctl.core.context import AppContext
from journalctl.core.crypto import ContentCipher
from journalctl.tools.registry import register_tools
from tests.fixtures.tenants import TenantSeed, seed_for  # noqa: F401 -- pytest discovery

pytestmark = pytest.mark.asyncio(loop_scope="session")


# ---------------------------------------------------------------------------
# Stub embedding service (no ONNX in isolation tests)
# ---------------------------------------------------------------------------


class _StubEmbeddingService:
    def encode(self, text: str) -> list[float]:
        return [0.0] * 384

    async def store_by_vector(self, conn: Any, entry_id: int, embedding: list[float]) -> None:
        pass

    async def search_by_vector(
        self,
        conn: Any,
        query_embedding: list[float],
        limit: int = 10,
        topic_prefix: str | None = None,
        date_from: Any = None,
        date_to: Any = None,
    ) -> list[dict]:
        return []


# ---------------------------------------------------------------------------
# Dataclass holding both users' seed handles
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DualUsers:
    user_a: UUID
    user_b: UUID
    seed_a: TenantSeed
    seed_b: TenantSeed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _with_user(user_id: UUID, coro: Any) -> Any:
    """Set current_user_id, await coro, reset on exit."""
    token = current_user_id.set(user_id)
    try:
        return await coro
    finally:
        current_user_id.reset(token)


async def _insert_user(admin_pool: asyncpg.Pool, email: str) -> UUID:
    async with admin_pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO users (id, email, timezone, created_at, updated_at)
            VALUES (gen_random_uuid(), $1, 'UTC', now(), now())
            RETURNING id
            """,
            email,
        )
    if row is None:
        raise RuntimeError(f"Failed to insert user {email}")
    return UUID(str(row["id"]))


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture(scope="session")
async def rls_tools(app_pool: asyncpg.Pool, tmp_path_factory: Any) -> dict:
    """MCP tools registered against the RLS-enforced app_pool.

    Session-scoped to match app_pool.  tmp_path_factory is the correct
    session-compatible alternative to tmp_path for session fixtures.
    """
    tmp_dir: Path = tmp_path_factory.mktemp("rls_tool_tests")
    settings = get_settings()
    _cipher = ContentCipher({1: bytes([1]) * 32})  # safe test-only key
    app_ctx = AppContext(
        pool=app_pool,
        embedding_service=_StubEmbeddingService(),  # type: ignore[arg-type]
        settings=settings,
        logger=structlog.get_logger("test"),
        cipher=_cipher,
    )
    (tmp_dir / "knowledge").mkdir(exist_ok=True)
    (tmp_dir / "conversations_json").mkdir(exist_ok=True)

    mcp = FastMCP("test-journalctl-isolation")
    register_tools(mcp, app_ctx)

    return {name: tool.fn for name, tool in mcp._tool_manager._tools.items()}


@pytest_asyncio.fixture(scope="session")
async def dual_users(admin_pool: asyncpg.Pool) -> DualUsers:
    """Create two users + seed data once per test session."""
    user_a = await _insert_user(admin_pool, "tool-isolation-a@test.local")
    user_b = await _insert_user(admin_pool, "tool-isolation-b@test.local")

    seed_a = await seed_for(
        admin_pool,
        user_a,
        topic_path="tool-iso-a/notes",
        topic_title="Tool Isolation A Notes",
        n_entries=2,
        include_conversation=False,
    )
    seed_b = await seed_for(
        admin_pool,
        user_b,
        topic_path="tool-iso-b/notes",
        topic_title="Tool Isolation B Notes",
        n_entries=2,
        include_conversation=False,
    )
    return DualUsers(user_a=user_a, user_b=user_b, seed_a=seed_a, seed_b=seed_b)


# ---------------------------------------------------------------------------
# Tests: journal_list_topics isolation
# ---------------------------------------------------------------------------


class TestListTopicsIsolation:
    """journal_list_topics returns only the authenticated user's topics."""

    async def test_user_a_sees_own_topic(self, rls_tools: dict, dual_users: DualUsers) -> None:
        result = await _with_user(
            dual_users.user_a,
            rls_tools["journal_list_topics"](),
        )
        paths = {t["topic"] for t in result["topics"]}
        assert dual_users.seed_a.topic_path in paths

    async def test_user_b_sees_own_topic(self, rls_tools: dict, dual_users: DualUsers) -> None:
        result = await _with_user(
            dual_users.user_b,
            rls_tools["journal_list_topics"](),
        )
        paths = {t["topic"] for t in result["topics"]}
        assert dual_users.seed_b.topic_path in paths

    async def test_cross_tenant_topic_invisible(
        self, rls_tools: dict, dual_users: DualUsers
    ) -> None:
        result_a = await _with_user(
            dual_users.user_a,
            rls_tools["journal_list_topics"](),
        )
        paths_a = {t["topic"] for t in result_a["topics"]}
        assert (
            dual_users.seed_b.topic_path not in paths_a
        ), f"User A must not see User B topic '{dual_users.seed_b.topic_path}'"

        result_b = await _with_user(
            dual_users.user_b,
            rls_tools["journal_list_topics"](),
        )
        paths_b = {t["topic"] for t in result_b["topics"]}
        assert (
            dual_users.seed_a.topic_path not in paths_b
        ), f"User B must not see User A topic '{dual_users.seed_a.topic_path}'"


# ---------------------------------------------------------------------------
# Tests: journal_read_topic isolation
# ---------------------------------------------------------------------------


class TestReadTopicIsolation:
    """journal_read_topic is scoped per tenant."""

    async def test_user_a_reads_own_entries(self, rls_tools: dict, dual_users: DualUsers) -> None:
        result = await _with_user(
            dual_users.user_a,
            rls_tools["journal_read_topic"](topic=dual_users.seed_a.topic_path),
        )
        # seed_for creates n_entries=2 rows
        assert result.get("total", 0) == 2

    async def test_user_b_cannot_read_user_a_topic(
        self, rls_tools: dict, dual_users: DualUsers
    ) -> None:
        # RLS hides the topic row itself, so journal_read_topic raises
        # TopicNotFoundError internally and returns error_code=NOT_FOUND.
        result = await _with_user(
            dual_users.user_b,
            rls_tools["journal_read_topic"](topic=dual_users.seed_a.topic_path),
        )
        assert (
            result.get("error_code") == "NOT_FOUND"
        ), f"User B reading User A's topic should return NOT_FOUND, got: {result}"


# ---------------------------------------------------------------------------
# Tests: journal_append_entry + journal_read_topic isolation
# ---------------------------------------------------------------------------


class TestAppendIsolation:
    """Entries written by user A are not readable by user B.

    Each test uses a unique topic path so tests are order-independent
    without requiring a per-test table truncation.
    """

    async def test_written_entry_invisible_to_other_user(
        self, rls_tools: dict, dual_users: DualUsers
    ) -> None:
        topic = "iso-private/entries"

        r = await _with_user(
            dual_users.user_a,
            rls_tools["journal_create_topic"](topic=topic, title="A Private"),
        )
        assert r.get("status") == "created"

        r = await _with_user(
            dual_users.user_a,
            rls_tools["journal_append_entry"](
                topic=topic,
                content="ULTRA_SECRET_CONTENT_USER_A_ONLY",
            ),
        )
        assert r.get("status") == "appended"

        # User B cannot access the topic at all -- RLS hides the topic row.
        result_b = await _with_user(
            dual_users.user_b,
            rls_tools["journal_read_topic"](topic=topic),
        )
        assert (
            result_b.get("error_code") == "NOT_FOUND"
        ), f"User B must not see User A's topic/entries, got: {result_b}"

    async def test_author_reads_own_entry(self, rls_tools: dict, dual_users: DualUsers) -> None:
        topic = "iso-private/verify"

        await _with_user(
            dual_users.user_a,
            rls_tools["journal_create_topic"](topic=topic, title="A Private Verify"),
        )
        await _with_user(
            dual_users.user_a,
            rls_tools["journal_append_entry"](
                topic=topic,
                content="ULTRA_SECRET_CONTENT_USER_A_ONLY",
            ),
        )

        result_a = await _with_user(
            dual_users.user_a,
            rls_tools["journal_read_topic"](topic=topic),
        )
        assert result_a["total"] >= 1
        assert any(
            "ULTRA_SECRET_CONTENT_USER_A_ONLY" in e.get("content", "") for e in result_a["entries"]
        )


# ---------------------------------------------------------------------------
# Tests: journal_search isolation
# ---------------------------------------------------------------------------


class TestSearchIsolation:
    """journal_search does not leak entries across tenants."""

    async def test_search_no_cross_tenant_leak(
        self, rls_tools: dict, dual_users: DualUsers
    ) -> None:
        topic = "iso-search/hidden"

        await _with_user(
            dual_users.user_a,
            rls_tools["journal_create_topic"](topic=topic, title="A Search Topic"),
        )
        await _with_user(
            dual_users.user_a,
            rls_tools["journal_append_entry"](
                topic=topic,
                content="UNIQ_SECRET_KEYWORD_ONLY_FOR_USER_A",
            ),
        )

        result_b = await _with_user(
            dual_users.user_b,
            rls_tools["journal_search"](query="UNIQ_SECRET_KEYWORD_ONLY_FOR_USER_A"),
        )
        assert (
            result_b["total"] == 0
        ), f"User B search must not find User A's entry, got {result_b['total']}"

    async def test_user_finds_own_entry_by_search(
        self, rls_tools: dict, dual_users: DualUsers
    ) -> None:
        topic = "iso-search/visible"

        await _with_user(
            dual_users.user_a,
            rls_tools["journal_create_topic"](topic=topic, title="A Search Topic 2"),
        )
        await _with_user(
            dual_users.user_a,
            rls_tools["journal_append_entry"](
                topic=topic,
                content="UNIQ_SECRET_KEYWORD_ONLY_FOR_USER_A",
            ),
        )

        result_a = await _with_user(
            dual_users.user_a,
            rls_tools["journal_search"](query="UNIQ_SECRET_KEYWORD_ONLY_FOR_USER_A"),
        )
        assert result_a["total"] >= 1, "User A should find their own entry"
