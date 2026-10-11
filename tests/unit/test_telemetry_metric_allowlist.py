"""Unit tests for the extraction gauges and their attribute allowlist."""

from __future__ import annotations

import pytest
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from gubbi.telemetry.attrs import GUBBI_METRIC_ALLOWLIST, MetricNames
from gubbi.telemetry.metrics import record_gauge
from tests.fixtures.metrics import collect_gauges, metric_reader  # noqa: F401

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("forbidden", ["user_id", "job_id", "conversation_id"])
def test_a_gauge_drops_an_identifier_attribute_and_keeps_its_allowlisted_one(
    metric_reader: InMemoryMetricReader,  # noqa: F811
    forbidden: str,
) -> None:
    record_gauge(MetricNames.EXTRACTION_QUEUE_DEPTH, 3, {"lane": "live", forbidden: "abc"})

    points = collect_gauges(metric_reader)[MetricNames.EXTRACTION_QUEUE_DEPTH]

    assert points == {(("lane", "live"),): 3.0}


@pytest.mark.parametrize("name", sorted(GUBBI_METRIC_ALLOWLIST))
def test_no_gauge_allowlists_a_user_or_job_id(name: str) -> None:
    assert GUBBI_METRIC_ALLOWLIST[name].isdisjoint({"user_id", "job_id", "conversation_id"})


def test_a_gauge_drops_an_attribute_allowlisted_only_for_another_gauge(
    metric_reader: InMemoryMetricReader,  # noqa: F811
) -> None:
    record_gauge(MetricNames.EXTRACTION_PROVIDER_TOKENS_MONTH, 9, {"kind": "input", "lane": "live"})

    points = collect_gauges(metric_reader)[MetricNames.EXTRACTION_PROVIDER_TOKENS_MONTH]

    assert points == {(("kind", "input"),): 9.0}


def test_recording_a_metric_outside_the_allowlist_raises() -> None:
    with pytest.raises(KeyError):
        record_gauge("extraction.unregistered", 1)
