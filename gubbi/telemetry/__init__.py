"""OpenTelemetry setup module for gubbi (TASK-03.19).

Provides ``configure_otel(app)`` called during FastAPI app startup.

Design:
    - OTEL_ENABLED env flag (default "true").
    - If enabled: wire real OTel SDK via gubbi-common's configure_otel.
    - If disabled: gubbi-common's configure_otel uses NoOp providers.
    - Auto-instrumentation for FastAPI, httpx, asyncpg, redis is handled
      by gubbi's ``_wire_instrumentors(app)``; gubbi-common stays free
      of FastAPI/instrumentor coupling.
    - Resource attributes from env: service.name, env, version, region
      are parsed by gubbi-common internally.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI

__all__: list[str] = ["configure_otel", "rebind_metrics_after_configure"]

# Renamed from bare `logger` because Python binds the sibling submodule
# `gubbi.telemetry.logger` onto this package's namespace as soon as
# anything (e.g. gubbi/main.py) imports it -- silently overwriting this
# module-level Logger instance and turning `logger.warning(...)` calls
# below into AttributeErrors against the submodule object.
_logger = logging.getLogger(__name__)

_OTEL_ENABLED_ENV = "OTEL_ENABLED"
_OTEL_SERVICE_NAME_ENV = "OTEL_SERVICE_NAME"


def _is_otel_enabled() -> bool:
    """Check the OTEL_ENABLED feature flag. Defaults to true."""
    raw = os.environ.get(_OTEL_ENABLED_ENV, "true")
    return raw.strip().lower() in ("true", "1", "yes")


def configure_otel(app: FastAPI) -> None:
    """Configure OpenTelemetry for the FastAPI application.

    Call during app lifespan startup, before any request handling.
    Idempotent: safe to call multiple times (subsequent calls no-op).

    Resource attributes (S8 M-1) are populated from in-process defaults:

    * ``service.version`` -- ``gubbi.__version__`` (mirrors pyproject).
    * ``deployment.environment`` -- ``settings.app_env``.

    ``OTEL_RESOURCE_ATTRIBUTES`` env var entries OVERLAY these defaults
    (per OTel spec); a deploy-time override always wins.

    Args:
        app: The FastAPI application instance.
    """
    from gubbi import __version__ as _gubbi_version
    from gubbi.config import get_settings

    service_name = os.environ.get(_OTEL_SERVICE_NAME_ENV, "gubbi")
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    enabled = _is_otel_enabled()
    settings = get_settings()
    from gubbi_common.telemetry.correlation_processor import (
        CorrelationSpanProcessor,
    )
    from gubbi_common.telemetry.otel import configure_otel as _common_configure_otel

    _common_configure_otel(
        service_name,
        endpoint,
        enabled=enabled,
        service_version=_gubbi_version,
        deployment_environment=settings.app_env,
        # Auto-inject correlation_id on every span at on_start (reads
        # the request-scoped ContextVar populated by CorrelationIDMiddleware).
        # Background spans without a request scope are left untagged --
        # see CorrelationSpanProcessor docstring for rationale.
        extra_processors=[CorrelationSpanProcessor()],
    )
    _wire_instrumentors(app)
    rebind_metrics_after_configure()


def rebind_metrics_after_configure() -> None:
    """Clear and re-prime ``initialize_metrics()`` against the live provider.

    CRIT-5 H-1: ``initialize_metrics()`` is ``lru_cache``-memoized so it
    only resolves a meter once. If any helper (e.g.
    ``record_audit_persistence_failure``) fired BEFORE ``configure_otel``
    completed -- for example from an early ``_build_app_ctx`` failure
    during ``scaffold_operator`` -- the cache would be sealed against the
    NoOp provider permanently and the DEC-098 ``audit.persistence_failure``
    alarm sensor would be dead. Clearing and re-priming here guarantees
    the cached instruments are bound to whatever meter provider is
    current at this call site (the SDK provider configured above in
    production; whatever the test installed in unit tests).

    Factored out of ``configure_otel`` so unit tests can exercise the
    lifespan re-bind contract without pulling in the FastAPI /
    instrumentor / OTLP exporter wiring.
    """
    from gubbi.telemetry.metrics import initialize_metrics

    initialize_metrics.cache_clear()
    initialize_metrics()


def _wire_instrumentors(app: FastAPI) -> None:
    """Register auto-instrumentation for FastAPI, httpx, asyncpg, redis.

    Safe to call even when the real SDK is NoOp -- instrumentors will
    use whatever tracer/meter provider is currently set.
    """
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app)
        _logger.debug("FastAPI auto-instrumentation wired")
    except Exception as exc:
        _logger.warning("FastAPIInstrumentor failed: %s", exc)

    try:
        from opentelemetry.instrumentation.httpx import (
            HTTPXClientInstrumentor,
        )

        HTTPXClientInstrumentor().instrument()
        _logger.debug("httpx auto-instrumentation wired")
    except Exception as exc:
        _logger.warning("HTTPXClientInstrumentor failed: %s", exc)

    try:
        from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor

        AsyncPGInstrumentor().instrument()  # type: ignore[no-untyped-call]  # opentelemetry-instrumentation-asyncpg ships no py.typed marker
        _logger.debug("asyncpg auto-instrumentation wired")
    except Exception as exc:
        _logger.warning("AsyncPGInstrumentor failed: %s", exc)

    try:
        from opentelemetry.instrumentation.redis import RedisInstrumentor

        RedisInstrumentor().instrument()
        _logger.debug("redis auto-instrumentation wired")
    except Exception as exc:
        _logger.warning("RedisInstrumentor failed: %s", exc)
