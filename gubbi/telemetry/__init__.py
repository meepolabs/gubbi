"""OpenTelemetry setup module for gubbi (TASK-03.19).

Provides ``configure_otel(app)`` called during FastAPI app startup.

Design:
    - OTEL_ENABLED env flag (default "true").
    - If enabled: wire real OTel SDK with OTLP exporters.
    - If disabled: wire NoOp providers so instrumentation calls are
      safe no-ops; avoids crash loops on Collector mis-config.
    - Auto-instrumentation for FastAPI, httpx, asyncpg, redis.
    - Resource attributes from env: service.name, env, version, region.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)

_OTEL_ENABLED_ENV = "OTEL_ENABLED"
_OTEL_SERVICE_NAME_ENV = "OTEL_SERVICE_NAME"
_OTEL_RESOURCE_ATTRIBUTES_ENV = "OTEL_RESOURCE_ATTRIBUTES"


def _is_otel_enabled() -> bool:
    """Check the OTEL_ENABLED feature flag. Defaults to true."""
    raw = os.environ.get(_OTEL_ENABLED_ENV, "true")
    return raw.strip().lower() in ("true", "1", "yes")


def _build_resource() -> Any:
    """Build an OTel Resource from env vars.

    Reads OTEL_SERVICE_NAME (default "gubbi") and
    OTEL_RESOURCE_ATTRIBUTES (comma-separated key=value pairs).
    """
    from opentelemetry.sdk.resources import Resource  # noqa: PLC0415

    service_name = os.environ.get(_OTEL_SERVICE_NAME_ENV, "gubbi")
    attrs: dict[str, str] = {"service.name": service_name}

    raw_res_attrs = os.environ.get(_OTEL_RESOURCE_ATTRIBUTES_ENV, "")
    if raw_res_attrs:
        for pair in raw_res_attrs.split(","):
            pair = pair.strip()
            if "=" in pair:
                key, value = pair.split("=", 1)
                attrs[key.strip()] = value.strip()

    return Resource.create(attrs)


def configure_otel(app: FastAPI) -> None:  # noqa: C901
    """Configure OpenTelemetry for the FastAPI application.

    Call during app lifespan startup, before any request handling.
    Idempotent: safe to call multiple times (subsequent calls no-op).

    Args:
        app: The FastAPI application instance.
    """
    if not _is_otel_enabled():
        logger.info("OTel disabled (OTEL_ENABLED=false) — using NoOp providers")
        from opentelemetry import trace  # noqa: PLC0415
        from opentelemetry.metrics import set_meter_provider  # noqa: PLC0415
        from opentelemetry.sdk.metrics import MeterProvider  # noqa: PLC0415
        from opentelemetry.sdk.trace import TracerProvider  # noqa: PLC0415

        noop_tracer = TracerProvider()
        trace.set_tracer_provider(noop_tracer)
        set_meter_provider(MeterProvider())

        # Still wire instrumentors so future code that starts spans does
        # not crash — they'll just be no-ops.
        _wire_instrumentors(app)
        return

    logger.info("Configuring OpenTelemetry for gubbi")

    from opentelemetry import trace  # noqa: PLC0415
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (  # noqa: PLC0415
        OTLPMetricExporter,
    )
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (  # noqa: PLC0415
        OTLPSpanExporter,
    )
    from opentelemetry.metrics import set_meter_provider  # noqa: PLC0415
    from opentelemetry.sdk.metrics import MeterProvider  # noqa: PLC0415
    from opentelemetry.sdk.metrics.export import (  # noqa: PLC0415
        PeriodicExportingMetricReader,
    )
    from opentelemetry.sdk.trace import TracerProvider  # noqa: PLC0415
    from opentelemetry.sdk.trace.export import BatchSpanProcessor  # noqa: PLC0415

    resource = _build_resource()

    # Trace provider with OTLP exporter
    tracer_provider = TracerProvider(resource=resource)
    span_exporter = OTLPSpanExporter()
    span_processor = BatchSpanProcessor(span_exporter)
    tracer_provider.add_span_processor(span_processor)
    trace.set_tracer_provider(tracer_provider)

    # Meter provider with OTLP exporter
    metric_exporter = OTLPMetricExporter()
    metric_reader = PeriodicExportingMetricReader(metric_exporter)
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    set_meter_provider(meter_provider)

    _wire_instrumentors(app)

    logger.info("OpenTelemetry configured: tracer + meter + auto-instrumentation ready")


def _wire_instrumentors(app: FastAPI) -> None:
    """Register auto-instrumentation for FastAPI, httpx, asyncpg, redis.

    Safe to call even when the real SDK is NoOp — instrumentors will
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

        AsyncPGInstrumentor().instrument()
        logger.debug("asyncpg auto-instrumentation wired")
    except Exception as exc:
        logger.warning("AsyncPGInstrumentor failed: %s", exc)

    try:
        from opentelemetry.instrumentation.redis import RedisInstrumentor  # noqa: PLC0415

        RedisInstrumentor().instrument()
        logger.debug("redis auto-instrumentation wired")
    except Exception as exc:
        logger.warning("RedisInstrumentor failed: %s", exc)
