"""Unit tests for oauth.disabled module."""

from fastapi import FastAPI
from starlette.routing import Route

from gubbi.oauth.disabled import register


class TestDisabledRegister:
    def test_registers_no_routes(self) -> None:
        app = FastAPI()
        register(app)
        route_paths = [r.path for r in app.routes if isinstance(r, Route)]
        oauth_paths = [
            "/authorize",
            "/token",
            "/register",
            "/login",
            "/.well-known/oauth-protected-resource/mcp",
        ]
        assert not any(
            p in route_paths for p in oauth_paths
        ), f"Expected no OAuth routes; got: {route_paths}"
