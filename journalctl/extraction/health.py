from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route


async def _health(_request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


app = Starlette(routes=[Route("/health", endpoint=_health)])
