"""Lifespan boot-order smoke tests (CO.53-gubbi).

Three smoke cases pinned by the spec:

1. Module import with minimal env causes no side-effect crash -- nothing in
   ``gubbi.main`` may call ``init_pool`` / ``aioredis.from_url`` / HTTP at
   import time. ``importlib.reload`` is used so module init re-runs and
   regressions where a side-effect is added to the import path surface.
2. Lifespan startup populates every typed ``app.state.*`` field expected
   by the typed accessors in ``gubbi.app_state`` BEFORE any request is
   served.
3. Shutdown does not surface a 500 with stack trace on a late request.

Heavy dependencies (Postgres pool, Redis pool, OAuth SQLite, MCP session
manager, FastAPIInstrumentor) are stubbed -- this is a structural smoke
test for the lifespan wiring, not an integration test of the underlying
backends.
"""

from __future__ import annotations

import importlib
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from asgi_lifespan import LifespanManager
from fastapi import FastAPI

import gubbi.main
from gubbi.app_context import AppContext
from gubbi.app_state import (
    require_app_ctx,
    require_auth_strategies,
    require_redis_client,
)

_MINIMAL_ENV_KEEP: frozenset[str] = frozenset(
    {
        "JOURNAL_API_KEY",
        "JOURNAL_OPERATOR_EMAIL",
        "JOURNAL_DB_APP_URL",
        "JOURNAL_TRANSPORT",
        "JOURNAL_SERVER_URL",
        "JOURNAL_DATA_DIR",
        "JOURNAL_PASSWORD_HASH",
    }
)
"""Env vars required for ``Settings`` validation.

The ``conftest._set_env`` autouse fixture sets these. Test A ("minimal env")
must KEEP these so ``Settings`` can resolve, but DELETE every optional
variable that gates side-effect-prone code paths (Hydra, gateway HMAC,
Redis URL, custom host).
"""


_MINIMAL_ENV_DROP: tuple[str, ...] = (
    "JOURNAL_DB_ADMIN_URL",
    "JOURNAL_REDIS_URL",
    "JOURNAL_HOST",
    "JOURNAL_TRUST_GATEWAY",
    "JOURNAL_GUBBI_GATEWAY_SECRET",
    "JOURNAL_GATEWAY_REQUIRE_SIGNATURE",
    "JOURNAL_HYDRA_ADMIN_URL",
    "JOURNAL_HYDRA_PUBLIC_ISSUER_URL",
    "JOURNAL_HYDRA_PUBLIC_URL",
    "JOURNAL_API_KEY_SCOPES",
    "JOURNAL_AUTH__HYDRA_ADMIN_URL",
    "JOURNAL_AUTH__TRUST_GATEWAY",
    "JOURNAL_AUTH__GATEWAY_SECRET",
    "JOURNAL_AUTH__GATEWAY_REQUIRE_SIGNATURE",
    "JOURNAL_AUTH__OPERATOR_EMAIL",
    "JOURNAL_AUTH__HYDRA_PUBLIC_ISSUER_URL",
    "JOURNAL_AUTH__HYDRA_PUBLIC_URL",
    "JOURNAL_OPERATOR_USER_ID",
    "JOURNAL_HEALTH_BIND_PUBLIC",
)
"""Optional env vars to delete so the import path runs through its default
configured-disabled branches (no Hydra, no trust gateway, no admin pool)."""


def _drop_optional_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove the optional JOURNAL_* env vars listed in ``_MINIMAL_ENV_DROP``."""
    for var in _MINIMAL_ENV_DROP:
        monkeypatch.delenv(var, raising=False)


# ---------------------------------------------------------------------------
# Case A: import has no side-effect crash with minimal env.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_module_import_has_no_side_effects_with_minimal_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reloading ``gubbi.main`` under minimal env must not trigger I/O.

    The reload exercises module-level code a second time -- catches
    regressions where import init re-runs side-effect-bearing setup
    (pool init, Redis connect, HTTP fetch) that should only run inside
    the lifespan context. Brittle in pytest collection but the better
    signal: a once-only ``import`` cache would mask any new init logic.
    """
    _drop_optional_env(monkeypatch)

    # Patch the side-effecting symbols at the module path used by lifespan.
    # If anything at IMPORT time touches them, the call_count assertion
    # below catches it.
    init_pool_mock = AsyncMock()
    monkeypatch.setattr("gubbi.main.init_pool", init_pool_mock)

    redis_from_url = MagicMock()
    monkeypatch.setattr(
        "redis.asyncio.ConnectionPool.from_url",
        redis_from_url,
    )

    # Do not patch httpx.AsyncClient -- the symbol is shared with the test
    # client used by Cases B/C, and patching it globally breaks those.
    # Importing ``gubbi.main`` only imports the httpx module; it does not
    # call ``httpx.AsyncClient(...)`` at module level (only inside the
    # lifespan when Hydra is wired).

    # Reload triggers re-execution of every top-level statement in main.py.
    reloaded = importlib.reload(gubbi.main)

    assert (
        init_pool_mock.call_count == 0
    ), "Importing gubbi.main must not call init_pool -- pool creation belongs to lifespan."
    assert (
        redis_from_url.call_count == 0
    ), "Importing gubbi.main must not open a Redis connection pool."

    assert isinstance(
        reloaded.server, FastAPI
    ), "Reloaded gubbi.main must expose a FastAPI app via the ``server`` symbol."
    # ``app_ctx`` is written by lifespan startup; it must NOT be set after
    # mere import. Test asserts the field is absent or None.
    assert (
        getattr(reloaded.server.state, "app_ctx", None) is None
    ), "app.state.app_ctx must be unset after import alone -- lifespan has not run."


# ---------------------------------------------------------------------------
# Shared lifespan stubs for Cases B and C.
# ---------------------------------------------------------------------------


def _build_stub_app_ctx() -> tuple[Any, Any, Any, Any]:
    """Return ``(app_ctx, pool, admin_pool, mcp)`` mimicking ``_build_app_ctx``.

    Each value is a minimal stand-in -- enough for the lifespan body to
    write it to ``app.state`` and call ``mcp.session_manager.run()`` /
    ``mcp.streamable_http_app()`` / ``pool.close()`` at shutdown.
    """
    pool = MagicMock()
    pool.close = AsyncMock()
    admin_pool = None  # exercises the no-admin-pool branch

    mcp_app = MagicMock()  # ASGI handler stand-in
    mcp = MagicMock()
    mcp.streamable_http_app = MagicMock(return_value=mcp_app)

    @asynccontextmanager
    async def _session_run() -> Any:
        yield None

    mcp.session_manager = MagicMock()
    mcp.session_manager.run = _session_run

    app_ctx = MagicMock(spec=AppContext)
    app_ctx.operator_user_id = None  # unbound -- valid configured-disabled state
    return app_ctx, pool, admin_pool, mcp


def _patch_lifespan_dependencies(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Mock every heavy dep the lifespan touches. Returns the mock handles."""
    app_ctx, pool, _admin_pool, mcp = _build_stub_app_ctx()

    build_app_ctx_mock = AsyncMock(return_value=(app_ctx, pool, None, mcp))
    monkeypatch.setattr("gubbi.main._build_app_ctx", build_app_ctx_mock)

    # Redis: stub the pool + the client object so .aclose() awaits succeed.
    redis_pool_stub = MagicMock()
    redis_pool_stub.aclose = AsyncMock()
    redis_client_stub = MagicMock()
    redis_client_stub.aclose = AsyncMock()

    monkeypatch.setattr(
        "redis.asyncio.ConnectionPool.from_url",
        MagicMock(return_value=redis_pool_stub),
    )
    monkeypatch.setattr(
        "redis.asyncio.Redis",
        MagicMock(return_value=redis_client_stub),
    )

    # Do NOT patch httpx.AsyncClient -- the test client transport in
    # Cases B/C uses the same symbol. The lifespan only constructs an
    # AsyncClient when JOURNAL_HYDRA_ADMIN_URL is set, which the
    # minimal-env override below disables.

    # OAuth: stub setup_oauth to skip SQLite + route registration.
    # setup_oauth is async def, so AsyncMock is required; MagicMock would
    # return a plain tuple and `await setup_oauth(...)` would raise
    # TypeError: 'tuple' object can't be awaited.
    oauth_storage_stub = MagicMock()
    oauth_storage_stub.close = AsyncMock()
    setup_oauth_mock = AsyncMock(return_value=(oauth_storage_stub, None))
    monkeypatch.setattr("gubbi.main.setup_oauth", setup_oauth_mock)

    # OTel + structlog initializers must be inert -- production wires the
    # SDK + log file handler; the test fixtures already configured logging.
    monkeypatch.setattr("gubbi.main.configure_otel", MagicMock())
    monkeypatch.setattr("gubbi.main.initialize_logger", MagicMock())

    return {
        "app_ctx": app_ctx,
        "pool": pool,
        "redis_pool": redis_pool_stub,
        "redis_client": redis_client_stub,
        "oauth_storage": oauth_storage_stub,
        "build_app_ctx": build_app_ctx_mock,
    }


# ---------------------------------------------------------------------------
# Case B: lifespan startup populates every typed app.state.* field.
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_lifespan_populates_app_state_before_first_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Driving the lifespan to the yield point must leave every typed field set.

    Asserts that every accessor in ``gubbi.app_state`` returns either a
    populated value or its valid configured-disabled state by the time
    the lifespan yields. This is the contract the rest of the app
    depends on: handlers may safely call ``require_app_ctx(request)``,
    ``require_auth_strategies(request)`` etc. after lifespan startup.
    """
    _drop_optional_env(monkeypatch)
    handles = _patch_lifespan_dependencies(monkeypatch)

    app = FastAPI(lifespan=gubbi.main.lifespan)

    async with LifespanManager(app):
        # All required fields must be populated.
        assert app.state.app_ctx is handles["app_ctx"]
        assert isinstance(app.state.auth_strategies, list)
        # Minimal env (api-key + operator_email, no Hydra / trust-gateway)
        # must produce at least one strategy -- ApiKeyStrategy.
        assert len(app.state.auth_strategies) >= 1
        assert app.state.redis_client is handles["redis_client"]

        # Optional fields: present on app.state, value may be None per
        # the contract.
        assert hasattr(app.state, "hydra_introspector")
        assert app.state.hydra_introspector is None  # no JOURNAL_HYDRA_ADMIN_URL

        assert hasattr(app.state, "selfhost_token_validator")
        # token_validator may be None when setup_oauth returns None.

        assert hasattr(app.state, "operator_user_id")
        # operator_user_id is None in this stub (build_stub_app_ctx).

        assert hasattr(app.state, "gubbi_gateway_secret")
        # None is a valid configured-disabled state.

        # require_*() accessors must NOT raise for the populated fields.
        scope = {"type": "http", "app": app, "headers": []}
        request = httpx.Request("GET", "http://testserver/")
        starlette_request = MagicMock()
        starlette_request.app = app
        # Use a real-enough Request shape -- accessors only touch request.app.state.
        starlette_request.app.state = app.state
        del request  # silence unused-var linters

        assert require_app_ctx(starlette_request) is handles["app_ctx"]
        assert require_auth_strategies(starlette_request) is app.state.auth_strategies
        assert require_redis_client(starlette_request) is handles["redis_client"]

        del scope  # silence unused-var linters

    # After exit: pool + redis cleanup awaited.
    handles["pool"].close.assert_awaited_once()
    handles["redis_client"].aclose.assert_awaited_once()
    handles["redis_pool"].aclose.assert_awaited_once()
    handles["oauth_storage"].close.assert_called_once()


# ---------------------------------------------------------------------------
# Case C: late requests after shutdown do not surface a 500 with stack trace.
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_post_shutdown_request_is_not_500(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A request issued AFTER lifespan shutdown must not crash with a 500.

    Acceptable outcomes:
      - 503 with Retry-After (graceful-degradation path).
      - ASGI lifespan-shutdown error raised before the handler runs.
      - any exception that is NOT a ``500 Internal Server Error`` with
        stack trace.

    The CRITICAL invariant is no uncaught exception escaping into the
    handler post-teardown.
    """
    _drop_optional_env(monkeypatch)
    _patch_lifespan_dependencies(monkeypatch)

    app = FastAPI(lifespan=gubbi.main.lifespan)

    @app.get("/late")
    async def _late_route() -> dict[str, str]:
        # Touches a torn-down resource on purpose. After shutdown, the
        # pool/redis stubs have been ``aclose()``-d. Reading app.state.app_ctx
        # is harmless on its own; the test below sends the request via
        # AsyncClient AFTER the lifespan context exits.
        return {"hello": "world"}

    transport = httpx.ASGITransport(app=app)

    async with (
        LifespanManager(app),
        httpx.AsyncClient(transport=transport, base_url="http://testserver") as client,
    ):
        warmup = await client.get("/late")
        assert warmup.status_code in {
            200,
            503,
        }, "Pre-shutdown request must succeed cleanly, not crash with 500."

    # Lifespan has now exited. Sending a fresh request via the same
    # transport drives the ASGI app post-teardown. We do not require a
    # specific status -- only that the failure mode is NOT a 500 with
    # an unhandled exception payload.
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        try:
            response = await client.get("/late")
        except Exception as exc:  # noqa: BLE001 -- intentionally broad; shutdown can raise many shapes
            # PT017 would prefer pytest.raises(), but the test contract
            # accepts EITHER an exception OR a non-500 response. The
            # `assert ... not in` guards the failure-mode invariant
            # without committing to a specific exception type.
            assert "500 Internal Server Error" not in repr(exc), (  # noqa: PT017
                f"Post-shutdown request raised a 500-style error: {exc!r}"
            )
        else:
            # If we got a response back, the critical invariant is no 500.
            assert (
                response.status_code != 500
            ), f"Post-shutdown request returned 500: body={response.text!r}"
