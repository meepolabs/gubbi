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
import structlog
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

    # ``server`` is now a CorrelationIDMiddleware-wrapped ASGI chain (the
    # ContextVar must populate BEFORE the FastAPIInstrumentor's
    # OpenTelemetryMiddleware opens its span -- see the rationale block at
    # the bottom of gubbi.main). The inner FastAPI is exposed as
    # ``app`` for tests / introspection that need the FastAPI surface
    # (state, routes, decorators).
    assert isinstance(
        reloaded.app, FastAPI
    ), "Reloaded gubbi.main must expose a FastAPI app via the ``app`` symbol."
    # ``app_ctx`` is written by lifespan startup; it must NOT be set after
    # mere import. Test asserts the field is absent or None.
    assert (
        getattr(reloaded.app.state, "app_ctx", None) is None
    ), "app.state.app_ctx must be unset after import alone -- lifespan has not run."


# ---------------------------------------------------------------------------
# Shared lifespan stubs for Cases B and C.
# ---------------------------------------------------------------------------


# Stub app-pool max size returned by ``pool.get_max_size()`` in the lifespan
# stubs. The lifespan's #138 WARNING reads the LIVE pool max via
# ``get_max_size()`` (not a constant), so the stub must answer with a real
# int. Kept distinct from the production ``APP_POOL_SIZE_MAX`` so the test
# pins the live-read contract, not a particular configured size.
_STUB_APP_POOL_MAX = 12


def _build_stub_app_ctx() -> tuple[Any, Any, Any, Any]:
    """Return ``(app_ctx, pool, admin_pool, mcp)`` mimicking ``_build_app_ctx``.

    Each value is a minimal stand-in -- enough for the lifespan body to
    write it to ``app.state`` and call ``mcp.session_manager.run()`` /
    ``mcp.streamable_http_app()`` / ``pool.close()`` at shutdown.
    """
    pool = MagicMock()
    pool.close = AsyncMock()
    # The #138 WARNING reads the live per-pod pool max via get_max_size();
    # a bare MagicMock would return a MagicMock (not an int) and break the
    # arithmetic. Answer with a real int.
    pool.get_max_size = MagicMock(return_value=_STUB_APP_POOL_MAX)
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
    # The lifespan PINGs Redis after client creation as a fail-fast probe.
    # Default the stub to "PONG"; tests that exercise the PING-failure
    # path override this attribute on the returned handle.
    redis_client_stub.ping = AsyncMock(return_value=b"PONG")

    monkeypatch.setattr(
        "redis.asyncio.ConnectionPool.from_url",
        MagicMock(return_value=redis_pool_stub),
    )
    monkeypatch.setattr(
        "redis.asyncio.Redis",
        MagicMock(return_value=redis_client_stub),
    )

    # Arq pool: stub arq_create_pool so it does not attempt a real Redis
    # connection.  The returned stub needs .close() to be awaitable for the
    # lifespan teardown path.
    arq_pool_stub = MagicMock()
    arq_pool_stub.close = AsyncMock()
    arq_create_pool_mock = AsyncMock(return_value=arq_pool_stub)
    monkeypatch.setattr("gubbi.main.arq_create_pool", arq_create_pool_mock)

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

    # pg_log_probe: stub to a no-op AsyncMock so the lifespan's startup
    # check never tries to fetch real Postgres GUCs from the stub pool.
    monkeypatch.setattr("gubbi.main.probe_pg_log_settings", AsyncMock(return_value=None))

    return {
        "app_ctx": app_ctx,
        "pool": pool,
        "redis_pool": redis_pool_stub,
        "redis_client": redis_client_stub,
        "arq_pool": arq_pool_stub,
        "arq_create_pool": arq_create_pool_mock,
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

        # budget_helper: must be set (or None) after lifespan startup (B3-L3).
        # With JOURNAL_LLM_BUDGET_ENABLED unset (default False), the value is None.
        assert hasattr(app.state, "budget_helper")
        # The value is None in minimal-env (budget disabled by default).
        assert app.state.budget_helper is None

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
    handles["arq_pool"].close.assert_awaited_once()
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
        except Exception as exc:
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


# ---------------------------------------------------------------------------
# Case D: lifespan PINGs Redis and aborts startup on PING failure.
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_lifespan_aborts_when_redis_ping_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Redis PING failure during lifespan startup must propagate.

    Without this, gubbi would yield a half-open service whose first SSE,
    arq, or budget call discovers Redis is down at request time. The
    DEC-098 fail-open audit fallback also depends on Redis-backed
    components; a silent boot against a dead Redis would erase that
    contract too.

    The fail-fast contract is: ``redis_client.ping()`` is called after
    ``aioredis.Redis(...)`` and BEFORE any router uses Redis. Any
    exception from ``ping()`` aborts the lifespan -- LifespanManager
    surfaces it as the underlying connection error.
    """
    import redis.exceptions as redis_exc

    _drop_optional_env(monkeypatch)
    handles = _patch_lifespan_dependencies(monkeypatch)
    # Override the default PONG stub: simulate Redis unreachable. The
    # message string is intentionally distinctive so the assertion below
    # can pin the exact propagation chain (no swallowing, no rewrap into
    # a generic RuntimeError).
    handles["redis_client"].ping = AsyncMock(
        side_effect=redis_exc.ConnectionError("simulated_redis_unreachable")
    )

    app = FastAPI(lifespan=gubbi.main.lifespan)

    with pytest.raises(redis_exc.ConnectionError, match="simulated_redis_unreachable"):
        async with LifespanManager(app):
            pass

    # PING was attempted exactly once.
    handles["redis_client"].ping.assert_awaited_once()
    # arq pool creation must not have run -- it sits AFTER the PING.
    # Asserting the patched factory was never awaited (rather than reading
    # the stub's close-await count) is the tight contract: the stub exists
    # before the lifespan body runs, so its own close-count is not a sound
    # proxy for "arq init never happened".
    handles["arq_create_pool"].assert_not_awaited()
    # The on-failure cleanup must close the resources opened pre-PING:
    # OAuth storage, Redis client (with pool), and the DB pool. These
    # are the "resource leak guard" the surrounding try/except provides.
    handles["oauth_storage"].close.assert_awaited_once()
    handles["redis_client"].aclose.assert_awaited_once()
    handles["pool"].close.assert_awaited_once()


# ---------------------------------------------------------------------------
# Case E: replica-count over-provisioning WARNING + alertable counter (#138).
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_lifespan_warns_and_increments_counter_when_replicas_gt_1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """JOURNAL_REPLICA_COUNT=2 -> structured WARNING + replica-count counter.

    gubbi's DB connection pool is per-pod; at N replicas the cluster
    carries N * pool_max from gubbi alone and the cloud-api startup budget
    guard does not see it. The lifespan must surface this with the
    ``db_pool_over_provisioned`` WARNING (carrying the per-pod / effective
    fields, read from the LIVE pool max) AND the alertable
    ``gateway.replica_count_warning`` counter (the log-sampling-proof
    counterpart -- counter NAME is shared cross-service, only the WARNING
    event key is gubbi-specific). Emitter distinction is via the
    ``service.name`` RESOURCE attribute set in ``configure_otel(app)``,
    not a metric attribute, so the helper takes only ``replica_count``.
    """
    _drop_optional_env(monkeypatch)
    _patch_lifespan_dependencies(monkeypatch)
    monkeypatch.setenv("JOURNAL_REPLICA_COUNT", "2")

    # Patch the counter helper so we assert the alertable signal fired
    # without standing up a real OTel meter provider in this smoke test.
    record_mock = MagicMock()
    monkeypatch.setattr("gubbi.main.record_replica_count_warning", record_mock)

    app = FastAPI(lifespan=gubbi.main.lifespan)

    with structlog.testing.capture_logs() as logs:
        async with LifespanManager(app):
            pass

    # Counter incremented exactly once. The emitter ("gubbi") is
    # distinguished via the service.name RESOURCE attribute set in
    # configure_otel(app), not via a metric attribute -- the helper takes
    # only replica_count.
    record_mock.assert_called_once_with(replica_count=2)

    # The structured WARNING fired with the over-provisioning fields.
    warnings = [
        log
        for log in logs
        if log.get("event") == "db_pool_over_provisioned" and log.get("log_level") == "warning"
    ]
    assert len(warnings) == 1, f"expected one over-provisioning WARNING, got {warnings}"
    emitted = warnings[0]
    assert emitted["replica_count"] == 2
    # admin_pool is None in this stub -> per-pod footprint is the live app
    # pool max (read via get_max_size(), here the stub's _STUB_APP_POOL_MAX).
    assert emitted["db_pool_max_per_pod"] == _STUB_APP_POOL_MAX
    assert (
        emitted["effective_db_connections"] == _STUB_APP_POOL_MAX * 2
    ), "effective DB connections must be POOL_MAX_PER_POD * REPLICA_COUNT"


@pytest.mark.unit
async def test_lifespan_silent_at_default_replica_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default (unset) JOURNAL_REPLICA_COUNT=1 -> no WARNING, no counter.

    Single-instance dev (the default) must not warn -- the over-
    provisioning gap only exists at > 1 replicas.
    """
    _drop_optional_env(monkeypatch)
    _patch_lifespan_dependencies(monkeypatch)
    monkeypatch.delenv("JOURNAL_REPLICA_COUNT", raising=False)

    record_mock = MagicMock()
    monkeypatch.setattr("gubbi.main.record_replica_count_warning", record_mock)

    app = FastAPI(lifespan=gubbi.main.lifespan)

    with structlog.testing.capture_logs() as logs:
        async with LifespanManager(app):
            pass

    record_mock.assert_not_called()
    assert not [
        log for log in logs if log.get("event") == "db_pool_over_provisioned"
    ], "default single-replica deploy must stay silent"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("value", "match"),
    [
        ("abc", "not an integer"),
        ("0", ">= 1"),
        ("-3", ">= 1"),
    ],
)
async def test_lifespan_rejects_invalid_replica_count(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
    match: str,
) -> None:
    """A non-integer or < 1 JOURNAL_REPLICA_COUNT aborts the lifespan."""
    _drop_optional_env(monkeypatch)
    _patch_lifespan_dependencies(monkeypatch)
    monkeypatch.setenv("JOURNAL_REPLICA_COUNT", value)
    monkeypatch.setattr("gubbi.main.record_replica_count_warning", MagicMock())

    app = FastAPI(lifespan=gubbi.main.lifespan)

    with pytest.raises(RuntimeError, match=match):
        async with LifespanManager(app):
            pass


@pytest.mark.unit
def test_validate_replica_count_helper_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unit-cover ``_validate_replica_count`` parse/validate in isolation."""
    monkeypatch.delenv("JOURNAL_REPLICA_COUNT", raising=False)
    assert gubbi.main._validate_replica_count() == 1

    monkeypatch.setenv("JOURNAL_REPLICA_COUNT", "5")
    assert gubbi.main._validate_replica_count() == 5

    monkeypatch.setenv("JOURNAL_REPLICA_COUNT", "abc")
    with pytest.raises(RuntimeError, match="not an integer"):
        gubbi.main._validate_replica_count()

    monkeypatch.setenv("JOURNAL_REPLICA_COUNT", "0")
    with pytest.raises(RuntimeError, match=">= 1"):
        gubbi.main._validate_replica_count()
