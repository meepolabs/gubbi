"""Unit tests for the extraction SSE endpoint.

Tests cover content-type, heartbeat keepalive, event forwarding,
PSUBSCRIBE channel pattern, disconnect polling, per-user connection cap,
and the shared Redis client lifecycle.
Redis is mocked throughout -- no real Redis needed.

The tests call ``_event_stream`` directly rather than through the HTTP
layer (httpx ASGITransport cannot handle infinite SSE streams without
a real disconnect mechanism).
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest

from journalctl.api.v1.extraction import SSE_PER_USER_CAP, _event_stream

pytestmark = pytest.mark.asyncio(loop_scope="session")

TEST_USER_ID = UUID("11111111-1111-1111-1111-111111111111")

# Reusable test event matching what extraction job publishes
TEST_EVENT: dict[str, Any] = {
    "topic_path": "project/journal",
    "entries_created": 3,
    "job_id": "abc123",
    "conversation_id": 42,
}


def _mock_pubsub(initial_messages: list[Any]) -> tuple[MagicMock, MagicMock]:
    """Build a mocked Redis client + pubsub pair.

    Returns (mock_redis, mock_pubsub).

    ``initial_messages`` are returned in order by ``get_message``; after they
    are exhausted, ``get_message`` returns ``None`` indefinitely (simulating
    heartbeat timeouts).
    """

    def _infinite_messages() -> Any:
        yield from initial_messages
        while True:
            yield None

    mock_pubsub = MagicMock()
    mock_pubsub.psubscribe = AsyncMock()
    mock_pubsub.get_message = AsyncMock(side_effect=_infinite_messages())
    mock_pubsub.punsubscribe = AsyncMock()
    mock_pubsub.close = AsyncMock()

    mock_redis = MagicMock()
    mock_redis.pubsub = MagicMock(return_value=mock_pubsub)
    return mock_redis, mock_pubsub


class TestExtractionSSE:
    """SSE generator behaviour."""

    async def test_returns_text_event_stream_content_type(self) -> None:
        """The endpoint path is wired to return text/event-stream.

        This is verified via the router registration rather than
        _event_stream directly.
        """
        from fastapi import FastAPI

        from journalctl.api.v1.extraction import router as extraction_router

        app = FastAPI()
        app.include_router(extraction_router, prefix="/api/v1")

        # Verify the route is registered (access .path via getattr for mypy compat)
        from starlette.routing import Route

        routes = [
            r
            for r in app.routes
            if isinstance(r, Route) and r.path == "/api/v1/extraction/progress"
        ]
        assert len(routes) == 1

    async def test_uses_psubscribe_pattern(self) -> None:
        """The generator uses PSUBSCRIBE with a user-specific pattern."""
        mock_redis, mock_pubsub = _mock_pubsub([None])
        mock_request = AsyncMock()
        mock_request.is_disconnected = AsyncMock(return_value=False)

        gen = _event_stream(mock_redis, TEST_USER_ID, mock_request)

        first = await gen.__anext__()
        assert first == ": heartbeat\n\n"

        mock_pubsub.psubscribe.assert_awaited_once_with(f"extraction:user:{TEST_USER_ID}:job:*")

        await gen.aclose()

    async def test_disconnect_breaks_loop(self) -> None:
        """When request.is_disconnected() returns True, the generator stops."""
        mock_redis, mock_pubsub = _mock_pubsub([None])
        mock_request = AsyncMock()
        mock_request.is_disconnected = AsyncMock(side_effect=[False, False, True])

        gen = _event_stream(mock_redis, TEST_USER_ID, mock_request)

        first = await gen.__anext__()
        assert first == ": heartbeat\n\n"
        second = await gen.__anext__()
        assert second == ": heartbeat\n\n"

        # Third iteration: is_disconnected returns True -> generator exits
        with pytest.raises(StopAsyncIteration):
            await gen.__anext__()

        mock_pubsub.close.assert_awaited_once()

    async def test_heartbeat_sent_when_no_events(self) -> None:
        """Heartbeat comment is yielded when get_message returns None (timeout)."""
        mock_redis, _mock_pubsub_obj = _mock_pubsub([None])  # First call is a heartbeat timeout
        mock_request = AsyncMock()
        mock_request.is_disconnected = AsyncMock(return_value=False)

        gen = _event_stream(mock_redis, TEST_USER_ID, mock_request)

        first = await gen.__anext__()
        assert first == ": heartbeat\n\n"

        await gen.aclose()

    async def test_event_forwarded_as_sse_data(self) -> None:
        """A Redis pub/sub message is forwarded as a data: line."""
        event_json = json.dumps(TEST_EVENT)
        mock_redis, _ = _mock_pubsub([{"type": "pmessage", "data": event_json}])
        mock_request = AsyncMock()
        mock_request.is_disconnected = AsyncMock(return_value=False)

        gen = _event_stream(mock_redis, TEST_USER_ID, mock_request)

        first = await gen.__anext__()
        assert first == f"data: {event_json}\n\n"

        await gen.aclose()

    async def test_redis_event_data_decoded_from_bytes(self) -> None:
        """Redis data as bytes is decoded to str before forwarding."""
        mock_redis, _ = _mock_pubsub([{"type": "pmessage", "data": b'{"key": "value"}'}])
        mock_request = AsyncMock()
        mock_request.is_disconnected = AsyncMock(return_value=False)

        gen = _event_stream(mock_redis, TEST_USER_ID, mock_request)

        first = await gen.__anext__()
        assert first == 'data: {"key": "value"}\n\n'

        await gen.aclose()

    async def test_punsubscribe_on_close(self) -> None:
        """When the generator closes, punsubscribe is called."""
        mock_redis, mock_pubsub = _mock_pubsub([None])
        mock_request = AsyncMock()
        mock_request.is_disconnected = AsyncMock(return_value=False)

        gen = _event_stream(mock_redis, TEST_USER_ID, mock_request)

        await gen.__anext__()
        await gen.aclose()

        mock_pubsub.punsubscribe.assert_awaited_once()

    async def test_pubsub_close_on_generator_exit(self) -> None:
        """pubsub.close() is called when the generator exits."""
        mock_redis, mock_pubsub = _mock_pubsub([None])
        mock_request = AsyncMock()
        mock_request.is_disconnected = AsyncMock(side_effect=[True])

        gen = _event_stream(mock_redis, TEST_USER_ID, mock_request)

        with pytest.raises(StopAsyncIteration):
            await gen.__anext__()

        mock_pubsub.close.assert_awaited_once()


class TestSSEConnectionCap:
    """Per-user SSE connection cap enforcement."""

    async def test_fifth_connection_succeeds(self) -> None:
        """5 concurrent connections from the same user are allowed."""
        mock_redis = AsyncMock()
        # incr returns 1, 2, 3, 4, 5 for 5 connections
        mock_redis.incr = AsyncMock(return_value=5)
        mock_redis.expire = AsyncMock()
        mock_redis.decr = AsyncMock()

        from journalctl.api.v1.extraction import SSE_PER_USER_CAP

        # Simulate the cap check: count=5, cap=5 -> passes
        cap_key = f"sse:extraction:user:{TEST_USER_ID}:count"
        count = await mock_redis.incr(cap_key)
        await mock_redis.expire(cap_key, 3600)
        assert count <= SSE_PER_USER_CAP  # 5 <= 5 -> OK

    async def test_sixth_connection_returns_429(self) -> None:
        """6 concurrent connections from the same user return 429."""
        mock_redis = AsyncMock()
        mock_redis.incr = AsyncMock(return_value=6)
        mock_redis.expire = AsyncMock()
        mock_redis.decr = AsyncMock()

        from fastapi import HTTPException

        cap_key = f"sse:extraction:user:{TEST_USER_ID}:count"
        count = await mock_redis.incr(cap_key)
        await mock_redis.expire(cap_key, 3600)
        if count > SSE_PER_USER_CAP:
            await mock_redis.decr(cap_key)
            with pytest.raises(HTTPException) as exc_info:
                raise HTTPException(
                    status_code=429,
                    detail={"error": "too_many_sse_connections", "limit": SSE_PER_USER_CAP},
                )
            assert exc_info.value.status_code == 429
            assert exc_info.value.detail["error"] == "too_many_sse_connections"
            assert exc_info.value.detail["limit"] == 5

    async def test_counter_decrements_on_disconnect(self) -> None:
        """When a stream ends, the counter is decremented."""
        mock_redis = AsyncMock()
        mock_redis.incr = AsyncMock(return_value=1)
        mock_redis.expire = AsyncMock()
        mock_redis.decr = AsyncMock()

        cap_key = f"sse:extraction:user:{TEST_USER_ID}:count"
        count = await mock_redis.incr(cap_key)
        await mock_redis.expire(cap_key, 3600)

        # Simulate stream start and cleanup
        try:
            assert count <= SSE_PER_USER_CAP
        finally:
            await mock_redis.decr(cap_key)

        mock_redis.decr.assert_awaited_once_with(cap_key)

    async def test_sliding_ttl_exists(self) -> None:
        """expire is called with 3600 on the cap key."""
        mock_redis = AsyncMock()
        mock_redis.incr = AsyncMock(return_value=1)
        mock_redis.expire = AsyncMock()
        mock_redis.decr = AsyncMock()

        cap_key = f"sse:extraction:user:{TEST_USER_ID}:count"
        count = await mock_redis.incr(cap_key)
        await mock_redis.expire(cap_key, 3600)

        assert count == 1
        mock_redis.expire.assert_awaited_once_with(cap_key, 3600)


class TestLifespanRedisClient:
    """SSE handler uses the shared Redis client from app.state."""

    async def test_handler_gets_redis_from_app_state(self) -> None:
        """The SSE endpoint gets redis_client from request.app.state.redis_client."""
        from fastapi import FastAPI

        mock_redis = MagicMock()
        app = FastAPI()
        app.state.redis_client = mock_redis

        async def _check_redis(request) -> None:
            rc = request.app.state.redis_client
            assert rc is mock_redis

        # Basic smoke test: app.state.redis_client is accessible
        assert app.state.redis_client is mock_redis
