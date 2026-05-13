"""Extraction API endpoints: live SSE progress stream and snapshot /me.

``/extraction/progress`` -- SSE stream of real-time extraction events.
``/extraction/me``        -- Snapshot of aggregate extraction job counts.

The SSE endpoint subscribes to Redis pub/sub channel
``extraction:user:{user_id}:job:*`` and forwards events as Server-Sent
Events.

The /me endpoint is the cold-start / refresh / reconnect complement to the
live stream: returns a single-row aggregate (in_flight_count, synced_count,
last_sync_at) from extraction_jobs via the repository layer.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from redis.asyncio import Redis as RedisClient

from gubbi.api.v1.auth import require_scope
from gubbi.app_state import require_app_ctx, require_redis_client
from gubbi.storage.connection import safe_user_scoped_connection
from gubbi.storage.repositories import extraction_jobs

__all__: list[str] = [
    "MeResponse",
    "SSE_PER_USER_CAP",
    "extraction_me",
    "extraction_progress",
    "router",
]

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
        redis_client: Shared Redis client from ``require_redis_client(request)``.
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
    """GET /api/v1/extraction/progress.

    Returns a Server-Sent Events stream that publishes extraction progress
    events in real time. The client connects, receives events as they are
    published by the extraction worker, and stays connected via heartbeats.

    Enforces a per-user connection cap of 5 (``SSE_PER_USER_CAP``). The 6th
    concurrent connection from the same user receives HTTP 429.
    """
    user_id, _scopes = auth
    redis_client = require_redis_client(request)

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


# ---------------------------------------------------------------------------
# /me -- snapshot of aggregate extraction job counts
# ---------------------------------------------------------------------------


class MeResponse(BaseModel):
    """Snapshot response for GET /v1/extraction/me.

    in_flight_count : jobs currently pending or running
    synced_count    : jobs that completed successfully
    last_sync_at    : completed_at of the most recent completed job, or None
    """

    in_flight_count: int
    synced_count: int
    last_sync_at: datetime | None


@router.get(
    "/me",
    response_model=MeResponse,
    responses={403: {"description": "missing scope"}},
)
async def extraction_me(
    request: Request,
    auth: Annotated[tuple[UUID, frozenset[str]], Depends(require_scope("journal:read"))],
) -> Response:
    """GET /api/v1/extraction/me.

    Returns a point-in-time snapshot of the authenticated user's extraction
    job counts. Intended as the cold-start / refresh complement to the live
    SSE stream at /extraction/progress.

    Response shape::

        {
            "in_flight_count": 5,
            "synced_count": 270,
            "last_sync_at": "2026-05-10T18:42:00Z"
        }

    ``last_sync_at`` is null when no jobs have completed. All counts are
    non-negative integers. Cache-Control: no-store is set unconditionally
    because the data reflects live worker state.
    """
    user_id, _scopes = auth
    app_ctx = require_app_ctx(request)
    async with safe_user_scoped_connection(app_ctx.pool, user_id=user_id) as conn:
        counts = await extraction_jobs.get_status_counts(conn)
    body = MeResponse(
        in_flight_count=counts.in_flight_count,
        synced_count=counts.synced_count,
        last_sync_at=counts.last_sync_at,
    )
    json_response = JSONResponse(body.model_dump(mode="json"))
    json_response.headers["Cache-Control"] = "no-store"
    return json_response
