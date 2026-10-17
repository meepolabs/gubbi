"""The app-wide 422 handler on the ingest route.

Body validation fails before the handler runs, so the app needs auth
strategies but no database.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gubbi.api.v1.ingest import router as ingest_router
from gubbi.auth.strategies import TrustGatewayStrategy
from gubbi.main import register_exception_handlers

pytestmark = pytest.mark.unit

ENDPOINT = "/api/v1/ingest/conversations"
HEADERS = {
    "X-Auth-User-Id": "dddddddd-dddd-dddd-dddd-dddddddddddd",
    "Content-Type": "application/json",
}
LONE_SURROGATE_ESCAPE = "\\ud800"


def _app(*, with_handlers: bool) -> TestClient:
    app = FastAPI()
    app.state.auth_strategies = [
        TrustGatewayStrategy(gateway_secret=None, gateway_require_signature=False),
    ]
    app.include_router(ingest_router, prefix="/api/v1")
    if with_handlers:
        register_exception_handlers(app)
    return TestClient(app, raise_server_exceptions=False)


def _conversation() -> dict[str, Any]:
    return {
        "platform": "chatgpt",
        "platform_id": "p-1",
        "created_at": "2026-03-01T17:00:00Z",
        "messages": [{"role": "user", "content": "hi", "message_id": "m-1"}],
    }


def _raw_body_with_lone_surrogate(field: str) -> bytes:
    conv = _conversation()
    placeholder = "SURROGATE_PLACEHOLDER"
    if field == "platform_id":
        conv["platform_id"] = placeholder
    else:
        conv["messages"][0][field] = placeholder
    text = json.dumps({"source": "extension_chatgpt", "conversations": [conv]})
    return text.replace(placeholder, LONE_SURROGATE_ESCAPE).encode("ascii")


@pytest.mark.parametrize("field", ["message_id", "platform_id", "role"])
def test_lone_surrogate_in_string_field_returns_422_not_500(field: str) -> None:
    client = _app(with_handlers=True)
    body = _raw_body_with_lone_surrogate(field)
    assert LONE_SURROGATE_ESCAPE.encode() in body

    resp = client.post(ENDPOINT, content=body, headers=HEADERS)

    assert resp.status_code == 422, resp.text
    detail = resp.json()["detail"]
    assert detail[0]["loc"][-1] == field
    assert "�" in json.dumps(detail, ensure_ascii=False)


def test_ordinary_422_body_matches_fastapi_default() -> None:
    conv = _conversation()
    del conv["platform_id"]
    payload = {"source": "extension_chatgpt", "conversations": [conv]}

    ours = _app(with_handlers=True).post(ENDPOINT, json=payload, headers=HEADERS)
    default = _app(with_handlers=False).post(ENDPOINT, json=payload, headers=HEADERS)

    assert ours.status_code == default.status_code == 422
    assert ours.json() == default.json()
    detail = ours.json()["detail"]
    assert detail[0]["loc"][-1] == "platform_id"
    assert {"loc", "msg", "type"} <= detail[0].keys()
