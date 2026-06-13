"""Malformed list filters return 422, not 500.

The list routers (entries, topics, conversations) delegate filter validation to
the storage repositories, which raise ``ValueError`` on a malformed
``date_from`` / ``date_to`` / topic-prefix. The router must translate that into
a 422 (contract-level filter error) rather than letting it surface as a 500.

These tests stub the user-scoped connection with a dummy that is never queried
(repository validation raises before any DB round-trip), so they need no
Postgres.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

import pytest
import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from httpx import ASGITransport, AsyncClient

from gubbi.api.v1.web import conversations as conv_router_mod
from gubbi.api.v1.web import entries as entries_router_mod
from gubbi.api.v1.web import topics as topics_router_mod
from gubbi.app_context import AppContext
from gubbi.auth.strategies import TrustGatewayStrategy
from gubbi.config import Settings
from gubbi.crypto.cipher import ContentCipher
from gubbi.storage.embedding_service import EmbeddingService

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

pytestmark = pytest.mark.asyncio(loop_scope="session")

API_PREFIX = "/api/v1"
_USER = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
_CIPHER = ContentCipher({1: bytes([1]) * 32})


class _DummyConn:
    """Connection stub. Repository validation raises before any method runs."""


@asynccontextmanager
async def _dummy_connection(*_args: object, **_kwargs: object) -> AsyncIterator[_DummyConn]:
    yield _DummyConn()


def _make_app() -> FastAPI:
    settings = Settings(
        db={"app_url": ""},
        auth={
            "api_key": "test-api-key-for-unit-tests-only",
            "operator_email": "web-filter-422-test@test.local",
            "trust_gateway": True,
        },
        server={"url": "http://localhost:8100"},
        data_dir=str(Path(__file__).parent),
    )
    app_ctx = AppContext(
        pool=None,  # type: ignore[arg-type]
        embedding_service=EmbeddingService(),
        settings=settings,
        logger=structlog.get_logger("test"),
        admin_pool=None,
        operator_user_id=_USER,
        cipher=_CIPHER,
    )
    app = FastAPI()
    app.state.app_ctx = app_ctx
    app.state.auth_strategies = [
        TrustGatewayStrategy(gateway_secret=None, gateway_require_signature=False),
    ]
    app.include_router(entries_router_mod.router, prefix=API_PREFIX)
    app.include_router(topics_router_mod.router, prefix=API_PREFIX)
    app.include_router(conv_router_mod.router, prefix=API_PREFIX)

    @app.exception_handler(Exception)
    async def _reraise(_request: Request, exc: Exception) -> JSONResponse:
        raise exc

    return app


@pytest.fixture
def client_fx(monkeypatch: pytest.MonkeyPatch) -> AsyncClient:
    monkeypatch.setattr(entries_router_mod, "safe_user_scoped_connection", _dummy_connection)
    monkeypatch.setattr(topics_router_mod, "safe_user_scoped_connection", _dummy_connection)
    monkeypatch.setattr(conv_router_mod, "safe_user_scoped_connection", _dummy_connection)
    transport = ASGITransport(app=_make_app())
    return AsyncClient(transport=transport, base_url="http://test")


async def test_entries_bad_date_from_returns_422(client_fx: AsyncClient) -> None:
    """A malformed date_from on /entries yields 422, not 500."""
    async with client_fx as client:
        resp = await client.get(
            f"{API_PREFIX}/entries",
            params={"date_from": "not-a-date"},
            headers={"X-Auth-User-Id": str(_USER)},
        )
    assert resp.status_code == 422


async def test_entries_bad_date_to_returns_422(client_fx: AsyncClient) -> None:
    """A malformed date_to on /entries yields 422, not 500."""
    async with client_fx as client:
        resp = await client.get(
            f"{API_PREFIX}/entries",
            params={"date_to": "2026-13-99"},
            headers={"X-Auth-User-Id": str(_USER)},
        )
    assert resp.status_code == 422


async def test_topics_bad_prefix_returns_422(client_fx: AsyncClient) -> None:
    """A malformed topic prefix on /topics yields 422, not 500."""
    async with client_fx as client:
        resp = await client.get(
            f"{API_PREFIX}/topics",
            params={"prefix": "Bad Prefix!!"},
            headers={"X-Auth-User-Id": str(_USER)},
        )
    assert resp.status_code == 422


async def test_conversations_bad_prefix_returns_422(client_fx: AsyncClient) -> None:
    """A malformed topic prefix on /conversations yields 422, not 500."""
    async with client_fx as client:
        resp = await client.get(
            f"{API_PREFIX}/conversations",
            params={"topic_prefix": "Bad Prefix!!"},
            headers={"X-Auth-User-Id": str(_USER)},
        )
    assert resp.status_code == 422
