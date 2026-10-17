"""Route class that caps the request body size, answering 413 when exceeded."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi.routing import APIRoute
from starlette.datastructures import Headers
from starlette.exceptions import HTTPException

from gubbi.constants import INGEST_MAX_BODY_BYTES

if TYPE_CHECKING:
    from starlette.types import Message, Receive, Scope, Send

__all__: list[str] = ["IngestBodyLimitRoute"]

_TOO_LARGE_DETAIL = "Request body too large"


def _declared_length(scope: Scope) -> int | None:
    value = Headers(scope=scope).get("content-length")
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _counting_receive(receive: Receive, limit: int) -> Receive:
    received = 0

    async def wrapped() -> Message:
        nonlocal received
        message = await receive()
        if message["type"] == "http.request":
            received += len(message.get("body", b""))
            if received > limit:
                raise HTTPException(status_code=413, detail=_TOO_LARGE_DETAIL)
        return message

    return wrapped


class IngestBodyLimitRoute(APIRoute):
    """APIRoute that rejects bodies over ``INGEST_MAX_BODY_BYTES`` with 413.

    The body is read before any dependency runs, so the cap applies before
    authentication. A declared Content-Length over the cap is refused without
    reading; otherwise received bytes are counted, which also bounds chunked
    bodies that declare no length. The HTTPException must be Starlette's:
    FastAPI re-raises it from the body read, but turns any other exception
    there into a 400.
    """

    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Refuse an over-cap declared length, then serve with a counting receive."""
        declared = _declared_length(scope)
        if declared is not None and declared > INGEST_MAX_BODY_BYTES:
            raise HTTPException(status_code=413, detail=_TOO_LARGE_DETAIL)
        await super().handle(scope, _counting_receive(receive, INGEST_MAX_BODY_BYTES), send)
