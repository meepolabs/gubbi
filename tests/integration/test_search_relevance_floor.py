"""Relevance floor on the semantic search backend, via the MCP ``journal_search`` tool.

The MCP tool and ``GET /api/v1/search`` both go through
``services.search.run_journal_search``, so the floor lives below the fork. This
module pins the tool surface; ``tests/api/v1/test_web_search.py`` pins the REST
one with the same positive/negative pairing.

Each test seeds a populated, EMBEDDED corpus and asserts BOTH halves: a nonsense
query returns nothing AND a real term returns something. The positive control is
what makes the negative assertion meaningful -- an empty or unembedded corpus
returns nothing for every query, so a lone negative assertion passes even with no
floor at all.

Uses the real ONNX model rather than a stub: a stub vector cannot demonstrate that
the floor separates genuine matches from noise, since the separation is a property
of the model's embedding geometry.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any
from uuid import UUID

import asyncpg
import pytest
import pytest_asyncio
import structlog
from gubbi_common.db.user_scoped import user_scoped_connection
from mcp.server.fastmcp import FastMCP

from gubbi.app_context import AppContext
from gubbi.auth_context import current_token_scopes, current_user_id
from gubbi.config import get_settings
from gubbi.crypto.cipher import ContentCipher
from gubbi.storage.embedding_service import EmbeddingService
from gubbi.storage.repositories import entries as entry_repo
from gubbi.storage.repositories import topics as topic_repo
from gubbi.tools.registry import register_tools

pytestmark = pytest.mark.asyncio(loop_scope="session")

_TOPIC = "floor/entries"
_CONTENT = "marathon training schedule"


@lru_cache(maxsize=1)
def _embeddings() -> EmbeddingService:
    """One shared real EmbeddingService -- ONNX session construction is slow.

    Seeded and query vectors must come from the same model instance for a
    similarity threshold to mean anything.
    """
    return EmbeddingService()


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
        raise RuntimeError("failed to insert test user")
    return UUID(str(row["id"]))


async def _search_as(user_id: UUID, tools: dict[str, Any], query: str) -> dict[str, Any]:
    """Call ``journal_search`` inside the auth context the tool layer expects."""
    user_token = current_user_id.set(user_id)
    scope_token = current_token_scopes.set(frozenset({"journal"}))
    try:
        result: dict[str, Any] = await tools["journal_search"](query=query, limit=10)
        return result
    finally:
        current_token_scopes.reset(scope_token)
        current_user_id.reset(user_token)


@pytest_asyncio.fixture
async def tools(app_pool: asyncpg.Pool, cipher: ContentCipher) -> dict[str, Any]:
    """Registered MCP tool callables backed by the real embedding service."""
    app_ctx = AppContext(
        pool=app_pool,
        embedding_service=_embeddings(),
        settings=get_settings(),
        logger=structlog.get_logger("test"),
        cipher=cipher,
    )
    mcp = FastMCP("test-search-relevance-floor")
    register_tools(mcp, app_ctx)
    return {name: tool.fn for name, tool in mcp._tool_manager._tools.items()}


@pytest_asyncio.fixture
async def seeded_user(
    app_pool: asyncpg.Pool,
    admin_pool: asyncpg.Pool,
    clean_rls_db: asyncpg.Pool,
    cipher: ContentCipher,
) -> UUID:
    """A user owning one entry that has BOTH an FTS vector and an embedding."""
    user_id = await _insert_user(admin_pool, "search-floor@test.local")
    service = _embeddings()
    async with user_scoped_connection(app_pool, user_id=user_id) as conn:
        await topic_repo.create(conn, topic=_TOPIC, title="Floor Entries")
        entry_id = await entry_repo.append(conn, cipher, topic=_TOPIC, content=_CONTENT)
        await service.save_by_vector(conn, entry_id, service.encode(_CONTENT))
    return user_id


async def test_nonsense_query_returns_nothing_but_real_term_matches(
    tools: dict[str, Any],
    seeded_user: UUID,
) -> None:
    """Populated, embedded corpus: a real term matches, a nonsense token does not."""
    positive = await _search_as(seeded_user, tools, "marathon")
    assert positive["total"] >= 1

    negative = await _search_as(seeded_user, tools, "zzzznomatchxyz")
    assert negative["total"] == 0
    assert negative["results"] == []


async def test_paraphrase_still_matches_semantically(
    tools: dict[str, Any],
    seeded_user: UUID,
) -> None:
    """No query term is in the entry text, so a hit proves semantic recall survives.

    Guards against "fixing" relevance by setting the floor so high that the
    semantic backend never contributes.
    """
    result = await _search_as(seeded_user, tools, "running plan for a race")
    assert result["total"] >= 1
