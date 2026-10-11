"""In-memory OTel meter for tests that read what gubbi's metric helpers record."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from opentelemetry.metrics import _internal as otel_metrics_internal
from opentelemetry.metrics import set_meter_provider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import Histogram, InMemoryMetricReader

from gubbi.telemetry import metrics as gubbi_metrics

type GaugePoints = dict[tuple[tuple[str, str], ...], float]


@pytest.fixture
def metric_reader() -> Iterator[InMemoryMetricReader]:
    """Install a fresh SDK meter provider with an in-memory reader; restore the old one after."""
    saved_provider = otel_metrics_internal._METER_PROVIDER  # type: ignore[attr-defined]
    saved_once = otel_metrics_internal._METER_PROVIDER_SET_ONCE  # type: ignore[attr-defined]
    reader = InMemoryMetricReader()
    otel_metrics_internal._METER_PROVIDER_SET_ONCE = otel_metrics_internal.Once()  # type: ignore[attr-defined]
    otel_metrics_internal._METER_PROVIDER = None  # type: ignore[attr-defined]
    set_meter_provider(MeterProvider(metric_readers=[reader]))
    gubbi_metrics.initialize_metrics.cache_clear()
    try:
        yield reader
    finally:
        otel_metrics_internal._METER_PROVIDER = saved_provider  # type: ignore[attr-defined]
        otel_metrics_internal._METER_PROVIDER_SET_ONCE = saved_once  # type: ignore[attr-defined]
        gubbi_metrics.initialize_metrics.cache_clear()


def collect_gauges(reader: InMemoryMetricReader) -> dict[str, GaugePoints]:
    """Collect once and return each gauge's or counter's points keyed by name, then attributes.

    A gauge reports its last value only to the first collection after it is
    set, so read every gauge of interest from one call. A counter's point is
    its running total. Histograms are skipped.
    """
    data = reader.get_metrics_data()
    gauges: dict[str, GaugePoints] = {}
    if data is None:
        return gauges
    for resource_metrics in data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                if isinstance(metric.data, Histogram):
                    continue
                points = gauges.setdefault(metric.name, {})
                for point in metric.data.data_points:
                    attrs = tuple(
                        sorted((str(k), str(v)) for k, v in (point.attributes or {}).items())
                    )
                    points[attrs] = float(point.value)  # type: ignore[union-attr]
    return gauges
