"""Fixtures for the ingest endpoint tests: a minimal app around the ingest router."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import UUID

import asyncpg
import pytest
import pytest_asyncio
import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from httpx import ASGITransport, AsyncClient

from gubbi.api.v1.ingest import router as ingest_router
from gubbi.app_context import AppContext
from gubbi.auth.strategies import TrustGatewayStrategy
from gubbi.config import Settings, get_settings
from gubbi.crypto.cipher import ContentCipher
from gubbi.storage.embedding_service import EmbeddingService
from tests.fixtures.ingest_app import API_PREFIX, MONTHLY, TEST_USER_ID


@pytest.fixture
def budget_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Enable budgets with a small monthly allowance."""
    monkeypatch.setenv("JOURNAL_LLM_BUDGET_ENABLED", "true")
    monkeypatch.setenv("JOURNAL_EXTRACTION_GLOBAL_MONTHLY_TOKEN_CEILING", "1000000000")
    monkeypatch.setenv("JOURNAL_LLM_WALLET_MONTHLY_AMOUNT", str(MONTHLY))
    get_settings.cache_clear()
    return monkeypatch


@pytest_asyncio.fixture
async def mock_arq_pool() -> AsyncMock:
    """An Arq pool mock; ingest must never call it."""
    pool = AsyncMock()
    pool.enqueue_job = AsyncMock(return_value=None)
    return pool


@pytest_asyncio.fixture
async def test_user(
    pool: asyncpg.Pool,
    clean_rls_db: asyncpg.Pool,
) -> AsyncGenerator[UUID, None]:
    """Ensure TEST_USER_ID exists in users table; tear down after test."""
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO users (id, email, timezone, created_at, updated_at)
            VALUES ($1, 'ingest-enqueue-test@test.local', 'UTC', now(), now())
            ON CONFLICT (id) DO NOTHING
            """,
            TEST_USER_ID,
        )
    yield TEST_USER_ID
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM extraction_jobs WHERE user_id = $1", TEST_USER_ID)
        await conn.execute("DELETE FROM entries WHERE user_id = $1", TEST_USER_ID)
        await conn.execute("DELETE FROM conversations WHERE user_id = $1", TEST_USER_ID)
        await conn.execute("DELETE FROM topics WHERE user_id = $1", TEST_USER_ID)
        # audit_log is append-only at the DB layer (immutability trigger);
        # clean_rls_db's per-test TRUNCATE handles audit_log cleanup via
        # journal_admin (BYPASSRLS) -- see tests/conftest.py:463.
        await conn.execute("DELETE FROM users WHERE id = $1", TEST_USER_ID)


@pytest_asyncio.fixture
async def app_with_arq(
    pool: asyncpg.Pool,
    tmp_path: Path,
    mock_arq_pool: AsyncMock,
    test_user: UUID,
) -> FastAPI:
    """Minimal FastAPI with an AppContext backed by the test pool and mock arq."""
    settings = Settings(
        db={"app_url": ""},
        auth={
            "api_key": "test-api-key-for-unit-tests-only",
            "operator_email": "ingest-enqueue-test@test.local",
            "trust_gateway": True,
        },
        server={"url": "http://localhost:8100"},
        data_dir=str(tmp_path),
    )
    cipher = ContentCipher({1: bytes([1]) * 32})
    app_ctx = AppContext(
        pool=pool,
        embedding_service=EmbeddingService(),
        settings=settings,
        logger=structlog.get_logger("test"),
        admin_pool=None,
        operator_user_id=TEST_USER_ID,
        cipher=cipher,
        arq_pool=mock_arq_pool,
    )
    app = FastAPI()
    app.state.app_ctx = app_ctx
    app.state.arq_pool = mock_arq_pool
    app.state.auth_strategies = [
        TrustGatewayStrategy(gateway_secret=None, gateway_require_signature=False),
    ]
    app.include_router(ingest_router, prefix=API_PREFIX)

    @app.exception_handler(Exception)
    async def _handler(request: Request, exc: Exception) -> JSONResponse:
        raise exc

    return app


@pytest_asyncio.fixture
async def client(app_with_arq: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    transport = ASGITransport(app=app_with_arq)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
