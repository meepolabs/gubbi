"""Unit tests for the ingest timestamp grammar and its UTC normalization.

The reject rows go through the ingest route: body validation fails before the
handler runs, so the app needs auth strategies but no database.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import TypeAdapter, ValidationError

from gubbi.api.v1.ingest import ConversationPayload, MessagePayload
from gubbi.api.v1.ingest import router as ingest_router
from gubbi.api.v1.ingest_time import IngestTimestamp
from gubbi.auth.strategies import TrustGatewayStrategy

pytestmark = pytest.mark.unit

ENDPOINT = "/api/v1/ingest/conversations"
HEADERS = {"X-Auth-User-Id": "dddddddd-dddd-dddd-dddd-dddddddddddd"}
VALID = "2026-03-01T17:00:00Z"

_adapter: TypeAdapter[datetime] = TypeAdapter(IngestTimestamp)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param("2026-03-01T17:00:00Z", datetime(2026, 3, 1, 17, tzinfo=UTC), id="zulu"),
        pytest.param(
            "2026-03-01T17:00:00+05:30", datetime(2026, 3, 1, 11, 30, tzinfo=UTC), id="plus_0530"
        ),
        pytest.param(
            "2026-03-01T17:00:00-00:00", datetime(2026, 3, 1, 17, tzinfo=UTC), id="minus_0000"
        ),
        pytest.param(
            "2026-03-01T17:00:00-08:00", datetime(2026, 3, 2, 1, tzinfo=UTC), id="minus_0800"
        ),
        pytest.param("2026-03-01T17:00:00", datetime(2026, 3, 1, 17, tzinfo=UTC), id="naive"),
        pytest.param("2026-03-01T17:05", datetime(2026, 3, 1, 17, 5, tzinfo=UTC), id="minutes"),
        pytest.param(
            "2026-03-01T17:00:00.123456789Z",
            datetime(2026, 3, 1, 17, 0, 0, 123456, tzinfo=UTC),
            id="nine_digit_fraction",
        ),
        pytest.param("0001-01-01T00:00Z", datetime(1, 1, 1, tzinfo=UTC), id="year_floor"),
        pytest.param(
            "2024-02-29T23:59:59Z", datetime(2024, 2, 29, 23, 59, 59, tzinfo=UTC), id="leap_day"
        ),
    ],
)
def test_canonical_timestamps_are_accepted_as_the_same_utc_instant(
    raw: str, expected: datetime
) -> None:
    parsed = _adapter.validate_python(raw)

    assert parsed == expected
    assert parsed.tzinfo is UTC


REJECTED: list[Any] = [
    pytest.param("2026-03-01", id="date_only"),
    pytest.param("2026-03-01 17:00:00Z", id="space_separator"),
    pytest.param("2026-03-01t17:00:00Z", id="lowercase_t"),
    pytest.param("2026-03-01T17:00:00z", id="lowercase_z"),
    pytest.param("2026-03-01T17:00:00+0530", id="offset_without_colon"),
    pytest.param(1772384400, id="epoch_int"),
    pytest.param("1772384400", id="epoch_string"),
    pytest.param("0000-01-01T00:00:00Z", id="year_0000"),
    pytest.param("2026-02-30T00:00:00Z", id="feb_30"),
    pytest.param("2026-03-01T24:00:00Z", id="hour_24"),
    pytest.param("2026-03-01T23:59:60Z", id="second_60"),
    pytest.param("9999-12-31T23:59:00-01:00", id="overflows_after_utc"),
    pytest.param("0001-01-01T00:00:00+01:00", id="underflows_after_utc"),
]


@pytest.mark.parametrize("raw", REJECTED)
def test_non_canonical_timestamps_are_rejected(raw: object) -> None:
    with pytest.raises(ValidationError):
        _adapter.validate_python(raw)


def test_optional_timestamps_accept_null() -> None:
    message = MessagePayload.model_validate({"role": "user", "content": "hi", "timestamp": None})
    conv = ConversationPayload.model_validate(
        {
            "platform": "chatgpt",
            "platform_id": "p-1",
            "created_at": VALID,
            "updated_at": None,
            "messages": [{"role": "user", "content": "hi"}],
        }
    )

    assert message.timestamp is None
    assert conv.updated_at is None


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.state.auth_strategies = [
        TrustGatewayStrategy(gateway_secret=None, gateway_require_signature=False),
    ]
    app.include_router(ingest_router, prefix="/api/v1")
    return TestClient(app, raise_server_exceptions=False)


def _body(field: str, value: object) -> dict[str, Any]:
    conv: dict[str, Any] = {
        "platform": "chatgpt",
        "platform_id": "p-1",
        "created_at": VALID,
        "updated_at": VALID,
        "messages": [{"role": "user", "content": "hi", "timestamp": VALID}],
    }
    if field == "timestamp":
        conv["messages"][0]["timestamp"] = value
    else:
        conv[field] = value
    return {"source": "extension_chatgpt", "conversations": [conv]}


@pytest.mark.parametrize("field", ["timestamp", "created_at", "updated_at"])
def test_route_passes_canonical_timestamps_past_validation(client: TestClient, field: str) -> None:
    resp = client.post(ENDPOINT, json=_body(field, VALID), headers=HEADERS)

    assert resp.status_code != 422, resp.text


@pytest.mark.parametrize("field", ["timestamp", "created_at", "updated_at"])
@pytest.mark.parametrize("raw", REJECTED)
def test_route_rejects_non_canonical_timestamps_with_422(
    client: TestClient, field: str, raw: object
) -> None:
    resp = client.post(ENDPOINT, json=_body(field, raw), headers=HEADERS)

    assert resp.status_code == 422, resp.text
    assert resp.json()["detail"][0]["loc"][-1] == field
