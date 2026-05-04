"""Integration test for the conversation ingest endpoint.

Tests the full flow: send a batch via HTTP, verify it was saved, then
verify dedupe skips a re-send of the same payload.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import asyncpg
import pytest
import pytest_asyncio
import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from httpx import ASGITransport, AsyncClient

from gubbi.api.v1.ingest import IngestConversationResponse
from gubbi.api.v1.ingest import router as ingest_router
from gubbi.config import Settings
from gubbi.core.context import AppContext
from gubbi.core.crypto import ContentCipher
from gubbi.storage.embedding_service import EmbeddingService

pytestmark = pytest.mark.asyncio(loop_scope="session")

API_PREFIX = "/api/v1"
ENDPOINT = f"{API_PREFIX}/ingest/conversations"

TEST_USER_ID = UUID("11111111-1111-1111-1111-111111111111")


def _build_conv_payload(
    platform_id: str,
    platform: str = "chatgpt",
    title: str = "",
    msg_count: int = 2,
) -> dict[str, Any]:
    """Build a single conversation payload dict."""
    now = datetime.now(UTC)
    return {
        "platform": platform,
        "platform_id": platform_id,
        "title": title,
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
        "messages": [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i}"}
            for i in range(msg_count)
        ],
    }


@pytest_asyncio.fixture
async def test_app(
    pool: asyncpg.Pool,
    clean_pool: asyncpg.Pool,  # noqa: ARG001 -- ensures clean tables
    tmp_path: Path,
) -> FastAPI:
    """Create a minimal FastAPI app with AppContext backed by the test pool."""
    settings = Settings(
        db={"app_url": ""},
        auth={
            "api_key": "test-api-key-for-unit-tests-only",
            "operator_email": "operator@test.local",
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
    )
    app = FastAPI()
    app.state.app_ctx = app_ctx
    app.include_router(ingest_router, prefix=API_PREFIX)

    @app.exception_handler(Exception)
    async def _handler(request: Request, exc: Exception) -> JSONResponse:
        raise exc

    return app


@pytest_asyncio.fixture
async def client(test_app: FastAPI) -> AsyncClient:
    """Async HTTP client against the test app."""
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestIngestEndpoint:
    """End-to-end tests for the ingest endpoint."""

    async def test_save_conversations(self, client: AsyncClient) -> None:
        """POST a valid batch, verify conversations_saved > 0."""
        payload = {
            "source": "extension_chatgpt",
            "conversations": [
                _build_conv_payload("conv-save-001", title="First chat"),
                _build_conv_payload("conv-save-002", title="Second chat"),
            ],
        }
        resp = await client.post(
            ENDPOINT,
            json=payload,
            headers={"X-Auth-User-Id": str(TEST_USER_ID)},
        )
        assert resp.status_code == 200, resp.text
        data = IngestConversationResponse.model_validate(resp.json())
        assert data.conversations_saved == 2
        assert data.conversations_skipped_dedupe == 0

    async def test_dedupe_skips_duplicates(self, client: AsyncClient) -> None:
        """Re-sending the same payload should skip via dedupe."""
        conv = _build_conv_payload("conv-dedup-001", title="Dedup test")
        payload = {
            "source": "extension_claude",
            "conversations": [conv],
        }

        resp1 = await client.post(
            ENDPOINT,
            json=payload,
            headers={"X-Auth-User-Id": str(TEST_USER_ID)},
        )
        assert resp1.status_code == 200
        data1 = IngestConversationResponse.model_validate(resp1.json())
        assert data1.conversations_saved == 1
        assert data1.conversations_skipped_dedupe == 0

        resp2 = await client.post(
            ENDPOINT,
            json=payload,
            headers={"X-Auth-User-Id": str(TEST_USER_ID)},
        )
        assert resp2.status_code == 200
        data2 = IngestConversationResponse.model_validate(resp2.json())
        assert data2.conversations_saved == 0
        assert data2.conversations_skipped_dedupe == 1

    async def test_save_and_read_back(
        self,
        client: AsyncClient,
        pool: asyncpg.Pool,
    ) -> None:
        """Save a conversation and verify it exists in the DB."""
        conv = _build_conv_payload(
            "conv-readback-001",
            platform="claude",
            title="Readback conversation",
            msg_count=3,
        )
        payload = {
            "source": "zip_upload",
            "conversations": [conv],
        }
        resp = await client.post(
            ENDPOINT,
            json=payload,
            headers={"X-Auth-User-Id": str(TEST_USER_ID)},
        )
        assert resp.status_code == 200

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT platform, platform_id, id FROM conversations"
                " WHERE platform = $1 AND platform_id = $2",
                "claude",
                "conv-readback-001",
            )
            assert row is not None, "Conversation not found in DB"
            assert row["platform"] == "claude"
            assert row["platform_id"] == "conv-readback-001"
            assert row["id"] is not None

    async def test_unauthenticated_request_rejected(self, client: AsyncClient) -> None:
        """Request without auth header should return 401."""
        payload = {
            "source": "extension_chatgpt",
            "conversations": [_build_conv_payload("no-auth-conv")],
        }
        resp = await client.post(ENDPOINT, json=payload)
        assert resp.status_code == 401

    async def test_validation_error_returns_422(self, client: AsyncClient) -> None:
        """Request with empty messages list should return 422 (Pydantic validation)."""
        payload = {
            "source": "extension_chatgpt",
            "conversations": [
                {
                    "platform": "chatgpt",
                    "platform_id": "bad-conv",
                    "created_at": datetime.now(UTC).isoformat(),
                    "messages": [],
                }
            ],
        }
        resp = await client.post(
            ENDPOINT,
            json=payload,
            headers={"X-Auth-User-Id": str(TEST_USER_ID)},
        )
        assert resp.status_code == 422  # noqa: PLR2004
