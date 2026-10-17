"""Route class that caps the request body size, answering 413 when exceeded."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi.routing import APIRoute
from starlette.datastructures import Headers
from starlette.exceptions import HTTPException
from starlette.requests import Request

from gubbi.api.v1.auth import resolve_user_id
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
    """APIRoute that authenticates before reading and caps bodies with 413.

    FastAPI reads and parses the body before any dependency runs, so the
    route's own auth dependency cannot keep an anonymous client away from
    the parser. The caller is therefore authenticated first, from headers
    alone (the request handed to the strategies has no receive channel),
    and the dependency still runs afterwards for scope checks. Then a
    declared Content-Length over the cap is refused without reading, and
    received bytes are counted, which also bounds chunked bodies that
    declare no length. The HTTPException must be Starlette's: FastAPI
    re-raises it from the body read, but turns any other exception there
    into a 400.
    """

    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Authenticate, refuse an over-cap declared length, then serve with a counting receive."""
        if self.methods is None or scope["method"] in self.methods:
            await resolve_user_id(Request(scope))
            declared = _declared_length(scope)
            if declared is not None and declared > INGEST_MAX_BODY_BYTES:
                raise HTTPException(status_code=413, detail=_TOO_LARGE_DETAIL)
        await super().handle(scope, _counting_receive(receive, INGEST_MAX_BODY_BYTES), send)
