"""SSE endpoint for real-time extraction progress.

Subscribes to Redis pub/sub channel ``extraction:user:{user_id}:job:*`` and
forwards events to connected clients as Server-Sent Events.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from redis.asyncio import Redis as RedisClient

from gubbi.api.v1.auth import require_scope

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/extraction", tags=["extraction"])

# How often (seconds) to send a keepalive comment when no events arrive.
# Prevents nginx/Cloudflare from timing out idle connections.
_HEARTBEAT_INTERVAL: float = 15.0

# Maximum simultaneous SSE connections per authenticated user.
# Further connections receive HTTP 429.
SSE_PER_USER_CAP: int = 5


async def _event_stream(
    redis_client: RedisClient,
    user_id: UUID,
    request: Request,
) -> AsyncGenerator[str, None]:
    """Async generator that yields SSE-formatted lines from Redis pub/sub.

    Uses PSUBSCRIBE with a pattern ``extraction:user:{user_id}:job:*`` so
    extraction jobs publishing to per-job channels are received.

    Args:
        redis_client: Shared Redis client from ``request.app.state.redis_client``.
        user_id: The authenticated user UUID whose extraction channel to
            subscribe to.
        request: The HTTP request, used for disconnect polling.

    Yields:
        SSE-formatted lines (either ``data: <json>\\n\\n`` or
        ``: heartbeat\\n\\n`` keepalive comments).
    """
    pubsub = redis_client.pubsub()
    subscribed = False
    try:
        pattern = f"extraction:user:{user_id}:job:*"
        await pubsub.psubscribe(pattern)
        subscribed = True
        while True:
            if await request.is_disconnected():
                break
            message = await pubsub.get_message(
                ignore_subscribe_messages=True,
                timeout=_HEARTBEAT_INTERVAL,
            )
            if message is None:
                yield ": heartbeat\n\n"
                continue
            if message["type"] == "pmessage":
                data = message["data"]
                if isinstance(data, bytes):
                    data = data.decode("utf-8")
                yield f"data: {data}\n\n"
    finally:
        if subscribed:
            await pubsub.punsubscribe()
        await pubsub.close()


@router.get("/progress")
async def extraction_progress(
    request: Request,
    auth: Annotated[tuple[UUID, frozenset[str]], Depends(require_scope("journal:read"))],
) -> StreamingResponse:
    """GET /api/v1/extraction/progress

    Returns a Server-Sent Events stream that publishes extraction progress
    events in real time. The client connects, receives events as they are
    published by the extraction worker, and stays connected via heartbeats.

    Enforces a per-user connection cap of 5 (``SSE_PER_USER_CAP``). The 6th
    concurrent connection from the same user receives HTTP 429.
    """
    user_id, _scopes = auth
    redis_client: RedisClient = request.app.state.redis_client

    cap_key = f"sse:extraction:user:{user_id}:count"
    count = await redis_client.incr(cap_key)
    await redis_client.expire(cap_key, 3600)
    if count > SSE_PER_USER_CAP:
        await redis_client.decr(cap_key)
        raise HTTPException(
            status_code=429,
            detail={"error": "too_many_sse_connections", "limit": SSE_PER_USER_CAP},
        )

    async def _capped_stream() -> AsyncGenerator[str, None]:
        try:
            async for chunk in _event_stream(redis_client, user_id, request):
                yield chunk
        finally:
            await redis_client.decr(cap_key)

    return StreamingResponse(_capped_stream(), media_type="text/event-stream")
