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

__all__: list[str] = ["configure_otel"]

logger = logging.getLogger(__name__)

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

    Args:
        app: The FastAPI application instance.
    """
    service_name = os.environ.get(_OTEL_SERVICE_NAME_ENV, "gubbi")
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    enabled = _is_otel_enabled()
    from gubbi_common.telemetry.otel import configure_otel as _common_configure_otel

    _common_configure_otel(service_name, endpoint, enabled=enabled)
    _wire_instrumentors(app)


def _wire_instrumentors(app: FastAPI) -> None:
    """Register auto-instrumentation for FastAPI, httpx, asyncpg, redis.

    Safe to call even when the real SDK is NoOp -- instrumentors will
    use whatever tracer/meter provider is currently set.
    """
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor  # noqa: PLC0415

        FastAPIInstrumentor.instrument_app(app)
        logger.debug("FastAPI auto-instrumentation wired")
    except Exception as exc:
        logger.warning("FastAPIInstrumentor failed: %s", exc)

    try:
        from opentelemetry.instrumentation.httpx import (  # noqa: PLC0415
            HTTPXClientInstrumentor,
        )

        HTTPXClientInstrumentor().instrument()
        logger.debug("httpx auto-instrumentation wired")
    except Exception as exc:
        logger.warning("HTTPXClientInstrumentor failed: %s", exc)

    try:
        from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor  # noqa: PLC0415

        AsyncPGInstrumentor().instrument()  # type: ignore[no-untyped-call]  # opentelemetry-instrumentation-asyncpg ships no py.typed marker
        logger.debug("asyncpg auto-instrumentation wired")
    except Exception as exc:
        logger.warning("AsyncPGInstrumentor failed: %s", exc)

    try:
        from opentelemetry.instrumentation.redis import RedisInstrumentor  # noqa: PLC0415

        RedisInstrumentor().instrument()
        logger.debug("redis auto-instrumentation wired")
    except Exception as exc:
        logger.warning("RedisInstrumentor failed: %s", exc)
