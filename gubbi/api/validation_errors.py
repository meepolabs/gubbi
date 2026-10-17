"""App-wide handler for request validation errors (422)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

if TYPE_CHECKING:
    from fastapi import Request
    from fastapi.exceptions import RequestValidationError

__all__: list[str] = ["request_validation_handler"]


def _utf8_safe(value: Any) -> Any:
    if isinstance(value, str):
        return value.encode("utf-8", "surrogatepass").decode("utf-8", "replace")
    if isinstance(value, list):
        return [_utf8_safe(item) for item in value]
    if isinstance(value, dict):
        return {_utf8_safe(key): _utf8_safe(item) for key, item in value.items()}
    return value


async def request_validation_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Return FastAPI's default 422 body with every string made UTF-8 encodable.

    The default body echoes the rejected input; a lone surrogate in it makes
    the response itself unencodable, which would surface as a 500.
    """
    return JSONResponse(
        status_code=422,
        content={"detail": _utf8_safe(jsonable_encoder(exc.errors()))},
    )
