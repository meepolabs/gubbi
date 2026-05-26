"""Tests for the gubbi ``/health/ready`` readiness endpoint.

Symmetric with cloud-api's ``/health/ready`` (T2 follow-up): liveness
(``/health``) returns 200 as soon as uvicorn is bound, readiness only
returns 200 once the lifespan has populated ``app.state.app_ctx`` AND
the underlying app pool round-trips a ``SELECT 1``.

The contract pinned here:

* Pre-lifespan / cold-start: 503 with ``status=not_ready`` (no app_ctx).
* Healthy: 200 with ``status=ok`` (pool acquire + SELECT 1 succeed).
* DB unreachable (driver / OS / timeout error): 503 with
  ``status=db_unreachable`` -- a generic failure does NOT leak the
  underlying exception text into the body.
* Liveness (``/health``) stays 200 regardless of lifespan state -- it
  is the kubelet's "process is alive" signal, not "app is ready".
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import asyncpg
import httpx
import pytest
from asgi_lifespan import LifespanManager
from fastapi import FastAPI

import gubbi.main
from tests.unit.test_lifespan_boot import (
    _drop_optional_env,
    _patch_lifespan_dependencies,
)


@pytest.mark.unit
async def test_health_ready_returns_503_before_lifespan_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A request issued BEFORE LifespanManager runs gets ``not_ready``.

    The bare FastAPI app (no lifespan driven) exposes the route, but
    ``app.state.app_ctx`` has not been written yet -- the readiness
    handler must short-circuit to 503 with ``status=not_ready``
    rather than crashing on a missing ``pool`` attribute.
    """
    _drop_optional_env(monkeypatch)
    _patch_lifespan_dependencies(monkeypatch)

    app = FastAPI(lifespan=gubbi.main.lifespan)

    # Re-register the readiness handler on the local FastAPI we built --
    # the production /health/ready route is bound to ``gubbi.main.app``,
    # not to this freshly constructed app.  Mirroring the lifespan tests
    # which build a fresh FastAPI for in-process drive.
    app.add_api_route("/health/ready", gubbi.main.mcp_health_ready, methods=["GET"])

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/health/ready")

    assert response.status_code == 503
    assert "not_ready" in response.text


@pytest.mark.unit
async def test_health_ready_returns_200_when_pool_round_trips(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inside an active lifespan with a healthy pool, /health/ready returns 200.

    The smoke fixture stubs ``pool.acquire().__aenter__()`` to return a
    connection whose ``fetchval`` returns ``None`` (the safe default for
    the PgLogProbe test path).  ``SELECT 1`` returns whatever the stub
    yields -- the readiness handler does not inspect the value, just
    that the call did not raise.

    The shared lifespan fixture builds a ``MagicMock(spec=AppContext)``
    where ``.pool`` is unspec'd (the dataclass field has no default, so
    ``Mock(spec=...)`` does not pre-populate the attribute).  The
    readiness handler reads ``app_ctx.pool`` so we attach the same pool
    handle the lifespan otherwise wires up via the
    ``_build_app_ctx`` mock return tuple.
    """
    _drop_optional_env(monkeypatch)
    handles = _patch_lifespan_dependencies(monkeypatch)
    handles["app_ctx"].pool = handles["pool"]

    app = FastAPI(lifespan=gubbi.main.lifespan)
    app.add_api_route("/health/ready", gubbi.main.mcp_health_ready, methods=["GET"])

    transport = httpx.ASGITransport(app=app)
    async with (
        LifespanManager(app),
        httpx.AsyncClient(transport=transport, base_url="http://testserver") as client,
    ):
        response = await client.get("/health/ready")

    assert response.status_code == 200
    assert response.text == '{"status":"ok"}'


@pytest.mark.unit
async def test_health_ready_returns_503_when_select_1_raises_db_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Driver errors during ``SELECT 1`` map to 503 ``db_unreachable``.

    The handler catches ``asyncpg.PostgresError``, ``OSError``, and
    ``TimeoutError`` so a saturated pool, a socket reset, or a query
    timeout all surface as the same drained-readiness shape -- the
    on-call dashboard sees one signal, not three.
    """
    _drop_optional_env(monkeypatch)
    handles = _patch_lifespan_dependencies(monkeypatch)
    handles["app_ctx"].pool = handles["pool"]

    # Override the shared pool's connection so SELECT 1 raises a
    # ``ConnectionFailureError`` (a ``PostgresError`` subclass).
    # The lifespan smoke fixture already mocks acquire().__aenter__ to
    # return a conn stub whose fetchval is an AsyncMock; here we
    # replace the side_effect so the readiness path exercises the
    # except branch.
    conn_stub = handles["pool"].acquire.return_value.__aenter__.return_value
    conn_stub.fetchval = AsyncMock(
        side_effect=asyncpg.exceptions.ConnectionFailureError("boom"),
    )

    app = FastAPI(lifespan=gubbi.main.lifespan)
    app.add_api_route("/health/ready", gubbi.main.mcp_health_ready, methods=["GET"])

    transport = httpx.ASGITransport(app=app)
    async with (
        LifespanManager(app),
        httpx.AsyncClient(transport=transport, base_url="http://testserver") as client,
    ):
        response = await client.get("/health/ready")

    assert response.status_code == 503
    assert "db_unreachable" in response.text
    # Underlying exception text must NOT bleed into the body -- the
    # response is a fixed envelope, no driver detail.
    assert "boom" not in response.text


@pytest.mark.unit
async def test_health_liveness_stays_200_regardless_of_lifespan_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``/health`` (liveness) must not depend on the pool / app_ctx.

    Pinned because a load balancer that uses the LIVENESS probe to drain
    traffic during cold-start would never bring the instance into
    rotation -- exactly the bug ``/health/ready`` was added to fix.
    Liveness must remain a pure 200.
    """
    _drop_optional_env(monkeypatch)
    _patch_lifespan_dependencies(monkeypatch)

    app = FastAPI(lifespan=gubbi.main.lifespan)
    app.add_api_route("/health", gubbi.main.mcp_health, methods=["GET"])

    transport = httpx.ASGITransport(app=app)
    # No LifespanManager: we deliberately hit /health BEFORE the lifespan
    # populates app_ctx so the contract "liveness ignores readiness state"
    # is observable.
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    body: dict[str, Any] = response.json()
    assert body == {"status": "ok"}
