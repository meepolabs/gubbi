"""Typed accessors for ``request.app.state`` fields populated by lifespan.

Replaces the legacy ``CustomFastAPI`` subclass attribute pattern (CO.39).
Each application-scoped resource the FastAPI lifespan installs on
``app.state`` is exposed via a paired accessor:

* ``require_<field>(request)`` -- returns the value or raises
  ``RuntimeError`` when it has not been initialised. Use in business
  logic where the lifespan invariant must hold.
* ``get_optional_<field>(request)`` -- returns the value or ``None``.
  Use for graceful-degradation paths (readiness probes, optional
  features, shutdown handling).

Both accessors take ``request: Request`` and read
``request.app.state.<field>``. Reads through ``Starlette.State.__getattr__``
return ``Any``; ``typing.cast`` carries the declared field type into the
return value without runtime cost. Each ``require_*`` raises a clear
``RuntimeError`` when the value is ``None`` (whether absent or
explicitly None), rather than letting a missing lifespan write surface
as a silent ``AttributeError`` from ``State``'s default ``KeyError``.

The ``gateway_secret`` accessor is intentionally optional-only -- ``None``
is a valid configured-disabled state for HMAC signing, not a lifespan
failure. See the NOTE block above ``get_optional_gateway_secret``.

Note: ``_run_stdio`` does NOT use these accessors -- stdio mode builds
``AppContext`` directly without a FastAPI request lifecycle. These
accessors are FastAPI-only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from fastapi import Request

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from uuid import UUID

    from arq.connections import ArqRedis
    from redis.asyncio import Redis as RedisClient

    from gubbi.app_context import AppContext
    from gubbi.auth.hydra import HydraIntrospector
    from gubbi.auth.strategies import AuthStrategy


__all__ = [
    "get_optional_app_ctx",
    "get_optional_arq_pool",
    "get_optional_auth_strategies",
    "get_optional_gateway_secret",
    "get_optional_hydra_introspector",
    "get_optional_operator_user_id",
    "get_optional_redis_client",
    "get_optional_selfhost_token_validator",
    "require_app_ctx",
    "require_arq_pool",
    "require_auth_strategies",
    "require_hydra_introspector",
    "require_operator_user_id",
    "require_redis_client",
    "require_selfhost_token_validator",
]


# ---------------------------------------------------------------------------
# app_ctx -- the AppContext bag (pool, embedding_service, settings, logger,
#            admin_pool, operator_user_id, cipher).
# ---------------------------------------------------------------------------


def require_app_ctx(request: Request) -> AppContext:
    """Return ``request.app.state.app_ctx`` or raise if not initialised."""
    value = getattr(request.app.state, "app_ctx", None)
    if value is None:
        raise RuntimeError("app_ctx not initialised")
    return cast("AppContext", value)


def get_optional_app_ctx(request: Request) -> AppContext | None:
    """Return ``request.app.state.app_ctx`` or ``None``."""
    return cast("AppContext | None", getattr(request.app.state, "app_ctx", None))


# ---------------------------------------------------------------------------
# auth_strategies -- list of authentication strategies tried in order.
# ---------------------------------------------------------------------------


def require_auth_strategies(request: Request) -> list[AuthStrategy]:
    """Return ``request.app.state.auth_strategies`` or raise if not initialised."""
    value = getattr(request.app.state, "auth_strategies", None)
    if value is None:
        raise RuntimeError("auth_strategies not initialised")
    return cast("list[AuthStrategy]", value)


def get_optional_auth_strategies(request: Request) -> list[AuthStrategy] | None:
    """Return ``request.app.state.auth_strategies`` or ``None``."""
    return cast(
        "list[AuthStrategy] | None",
        getattr(request.app.state, "auth_strategies", None),
    )


# ---------------------------------------------------------------------------
# gubbi_gateway_secret -- HMAC secret for X-Auth-* envelope verification.
#
# Field name on app.state is ``gubbi_gateway_secret``; accessor name uses the
# shorter ``gateway_secret`` for symmetry with the cloud-side accessor.
#
# NOTE: gubbi_gateway_secret is intentionally optional-only.
# ``None`` is a valid configured-disabled state (gateway HMAC signing
# turned off), not a "lifespan never ran" failure. The lifespan reads
# app.state.gubbi_gateway_secret directly; no production code consumes
# the accessor via a Request object. Mirrors gubbi-cloud/app_state.py.
# ---------------------------------------------------------------------------


def get_optional_gateway_secret(request: Request) -> bytes | None:
    """Return ``request.app.state.gubbi_gateway_secret`` or ``None``."""
    return cast(
        "bytes | None",
        getattr(request.app.state, "gubbi_gateway_secret", None),
    )


# ---------------------------------------------------------------------------
# hydra_introspector -- optional Hydra OAuth token introspector.
# ---------------------------------------------------------------------------


def require_hydra_introspector(request: Request) -> HydraIntrospector:
    """Return ``request.app.state.hydra_introspector`` or raise if not configured.

    Raises when Hydra is not enabled in this deployment. Use
    ``get_optional_hydra_introspector`` for graceful-degradation paths.
    """
    value = getattr(request.app.state, "hydra_introspector", None)
    if value is None:
        raise RuntimeError("hydra_introspector not initialised")
    return cast("HydraIntrospector", value)


def get_optional_hydra_introspector(request: Request) -> HydraIntrospector | None:
    """Return ``request.app.state.hydra_introspector`` or ``None``."""
    return cast(
        "HydraIntrospector | None",
        getattr(request.app.state, "hydra_introspector", None),
    )


# ---------------------------------------------------------------------------
# selfhost_token_validator -- optional callable for self-host OAuth token check.
# ---------------------------------------------------------------------------


def require_selfhost_token_validator(
    request: Request,
) -> Callable[[str], Awaitable[frozenset[str] | None]]:
    """Return ``request.app.state.selfhost_token_validator`` or raise if absent."""
    value = getattr(request.app.state, "selfhost_token_validator", None)
    if value is None:
        raise RuntimeError("selfhost_token_validator not initialised")
    return cast("Callable[[str], Awaitable[frozenset[str] | None]]", value)


def get_optional_selfhost_token_validator(
    request: Request,
) -> Callable[[str], Awaitable[frozenset[str] | None]] | None:
    """Return ``request.app.state.selfhost_token_validator`` or ``None``."""
    return cast(
        "Callable[[str], Awaitable[frozenset[str] | None]] | None",
        getattr(request.app.state, "selfhost_token_validator", None),
    )


# ---------------------------------------------------------------------------
# operator_user_id -- UUID bound for operator-identity auth (None disables).
# ---------------------------------------------------------------------------


def require_operator_user_id(request: Request) -> UUID:
    """Return ``request.app.state.operator_user_id`` or raise if unbound."""
    value = getattr(request.app.state, "operator_user_id", None)
    if value is None:
        raise RuntimeError("operator_user_id not initialised")
    return cast("UUID", value)


def get_optional_operator_user_id(request: Request) -> UUID | None:
    """Return ``request.app.state.operator_user_id`` or ``None``."""
    return cast(
        "UUID | None",
        getattr(request.app.state, "operator_user_id", None),
    )


# ---------------------------------------------------------------------------
# redis_client -- shared Redis client (SSE pub/sub, connection caps).
# ---------------------------------------------------------------------------


def require_redis_client(request: Request) -> RedisClient:
    """Return ``request.app.state.redis_client`` or raise if not initialised."""
    value = getattr(request.app.state, "redis_client", None)
    if value is None:
        raise RuntimeError("redis_client not initialised")
    return cast("RedisClient", value)


def get_optional_redis_client(request: Request) -> RedisClient | None:
    """Return ``request.app.state.redis_client`` or ``None``."""
    return cast(
        "RedisClient | None",
        getattr(request.app.state, "redis_client", None),
    )


# ---------------------------------------------------------------------------
# arq_pool -- Arq Redis connection pool for background job enqueue.
#
# Populated during lifespan startup alongside the aioredis SSE client.
# None is a valid state for self-host deployments that do not run the
# extraction worker, so get_optional is the safe accessor; require is
# provided for request-path code that cannot proceed without the pool.
# ---------------------------------------------------------------------------


def require_arq_pool(request: Request) -> ArqRedis:
    """Return ``request.app.state.arq_pool`` or raise if not initialised."""
    value = getattr(request.app.state, "arq_pool", None)
    if value is None:
        raise RuntimeError("arq_pool not initialised")
    return cast("ArqRedis", value)


def get_optional_arq_pool(request: Request) -> ArqRedis | None:
    """Return ``request.app.state.arq_pool`` or ``None``."""
    return cast(
        "ArqRedis | None",
        getattr(request.app.state, "arq_pool", None),
    )
