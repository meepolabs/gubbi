"""The ingest route refuses request bodies over the size cap with 413.

Driven at the ASGI level so a body can arrive as several ``http.request``
messages with no Content-Length, which an HTTP client cannot produce. No
database: every case is decided before the handler runs.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi import FastAPI

from gubbi.api.v1.ingest import router as ingest_router
from gubbi.auth.strategies import TrustGatewayStrategy
from gubbi.constants import INGEST_MAX_BODY_BYTES
from tests.fixtures.ingest_app import API_PREFIX, ENDPOINT, headers

MIB = 1024 * 1024


def _app() -> FastAPI:
    app = FastAPI()
    app.state.auth_strategies = [
        TrustGatewayStrategy(gateway_secret=None, gateway_require_signature=False),
    ]
    app.include_router(ingest_router, prefix=API_PREFIX)
    return app


class _Exchange:
    """Feeds scripted request messages to the app and records its response."""

    def __init__(self, chunks: list[bytes], extra_headers: dict[str, str]) -> None:
        self.messages: list[dict[str, Any]] = [
            {"type": "http.request", "body": chunk, "more_body": i < len(chunks) - 1}
            for i, chunk in enumerate(chunks)
        ]
        self.headers = {"content-type": "application/json", **extra_headers}
        self.receive_calls = 0
        self.status: int | None = None
        self.body = b""

    async def receive(self) -> dict[str, Any]:
        self.receive_calls += 1
        if self.receive_calls <= len(self.messages):
            return self.messages[self.receive_calls - 1]
        return {"type": "http.disconnect"}

    async def send(self, message: dict[str, Any]) -> None:
        if message["type"] == "http.response.start":
            self.status = message["status"]
        elif message["type"] == "http.response.body":
            self.body += message.get("body", b"")

    async def run(self, app: FastAPI) -> None:
        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": ENDPOINT,
            "raw_path": ENDPOINT.encode(),
            "root_path": "",
            "query_string": b"",
            "headers": [(k.lower().encode(), v.encode()) for k, v in self.headers.items()],
            "client": ("127.0.0.1", 1234),
            "server": ("test", 80),
            "state": {},
        }
        await app(scope, self.receive, self.send)


async def test_chunked_body_over_cap_is_413_and_reading_stops_at_the_crossing_chunk() -> None:
    full_chunks = INGEST_MAX_BODY_BYTES // MIB
    chunks = [b" " * MIB] * full_chunks + [b"x", b"y"]
    exchange = _Exchange(chunks, headers())

    await exchange.run(_app())

    assert exchange.status == 413
    assert exchange.receive_calls == full_chunks + 1


async def test_declared_length_over_cap_is_413_without_reading_the_body() -> None:
    exchange = _Exchange([b"{}"], {**headers(), "content-length": str(INGEST_MAX_BODY_BYTES + 1)})

    await exchange.run(_app())

    assert exchange.status == 413
    assert exchange.receive_calls == 0


async def test_chunked_body_without_credentials_is_401_before_any_read() -> None:
    chunks = [b" " * MIB] * (INGEST_MAX_BODY_BYTES // MIB) + [b"x"]
    exchange = _Exchange(chunks, {})

    await exchange.run(_app())

    assert exchange.status == 401
    assert exchange.receive_calls == 0
    assert json.loads(exchange.body) == {"detail": "Missing X-Auth-User-Id header"}


async def test_oversize_declared_length_without_credentials_is_401_without_reading() -> None:
    exchange = _Exchange([b"{}"], {"content-length": str(INGEST_MAX_BODY_BYTES + 1)})

    await exchange.run(_app())

    assert exchange.status == 401
    assert exchange.receive_calls == 0


async def test_valid_body_with_credentials_passes_the_route_and_reaches_validation() -> None:
    exchange = _Exchange([b'{"conversations": "not-a-list"}'], headers())

    await exchange.run(_app())

    assert exchange.status == 422
    assert exchange.receive_calls >= 1


@pytest.mark.parametrize("declare_length", [True, False], ids=["content-length", "chunked"])
async def test_body_of_exactly_the_cap_reaches_validation(declare_length: bool) -> None:
    prefix = b'{"pad":1}'
    body = prefix + b" " * (INGEST_MAX_BODY_BYTES - len(prefix))
    extra = {"content-length": str(len(body))} if declare_length else {}
    exchange = _Exchange([body[:MIB], body[MIB:]], {**headers(), **extra})

    await exchange.run(_app())

    assert len(body) == INGEST_MAX_BODY_BYTES
    assert exchange.status == 422
    missing = {err["loc"][-1] for err in json.loads(exchange.body)["detail"]}
    assert "conversations" in missing
