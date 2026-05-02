"""SSE endpoint for real-time extraction progress.

Subscribes to Redis pub/sub channel ``extraction:{user_id}`` and
forwards events to connected clients as Server-Sent Events.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import Annotated
from uuid import UUID

import redis.asyncio as aioredis
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from journalctl.api.v1.ingest import _resolve_user_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/extraction", tags=["extraction"])

# How often (seconds) to send a keepalive comment when no events arrive.
# Prevents nginx/Cloudflare from timing out idle connections.
_HEARTBEAT_INTERVAL: float = 15.0


async def _event_stream(redis_url: str, user_id: UUID) -> AsyncGenerator[str, None]:
    """Async generator that yields SSE-formatted lines from Redis pub/sub.

    Args:
        redis_url: Redis connection URL (e.g. ``redis://localhost:6379``).
        user_id: The authenticated user UUID whose extraction channel to
            subscribe to.

    Yields:
        SSE-formatted lines (either ``data: <json>\\n\\n`` or
        ``: heartbeat\\n\\n`` keepalive comments).
    """
    redis_client = aioredis.from_url(redis_url)
    try:
        pubsub = redis_client.pubsub()
        subscribed = False
        try:
            await pubsub.subscribe(f"extraction:{user_id}")
            subscribed = True
            while True:
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=_HEARTBEAT_INTERVAL,
                )
                if message is None:
                    yield ": heartbeat\n\n"
                    continue
                if message["type"] == "message":
                    data = message["data"]
                    if isinstance(data, bytes):
                        data = data.decode("utf-8")
                    yield f"data: {data}\n\n"
        finally:
            if subscribed:
                await pubsub.unsubscribe()
            await pubsub.close()
    finally:
        await redis_client.aclose()


@router.get("/progress")
async def extraction_progress(
    request: Request,
    user_id: Annotated[UUID, Depends(_resolve_user_id)],
) -> StreamingResponse:
    """GET /api/v1/extraction/progress

    Returns a Server-Sent Events stream that publishes extraction progress
    events in real time. The client connects, receives events as they are
    published by the extraction worker, and stays connected via heartbeats.
    """
    redis_url = str(request.app.state.app_ctx.settings.redis_url)
    return StreamingResponse(
        _event_stream(redis_url, user_id),
        media_type="text/event-stream",
    )
