"""Unit tests for the counters recorded after a settled attempt commits."""

from __future__ import annotations

from unittest.mock import call, patch

import pytest

from gubbi.extraction.charging import SettledAttempt, record_charge, record_unmetered
from gubbi.extraction.wallet import ChargeSplit
from gubbi.storage.repositories.wallet import Charge
from gubbi.telemetry.attrs import MetricNames

pytestmark = pytest.mark.unit

_COUNTER = MetricNames.EXTRACTION_CHARGES_WEIGHTED


def _settled(monthly: int, credit: int, lane: str = "live") -> SettledAttempt:
    return SettledAttempt(
        lane=lane,
        charge=Charge(job_kind=lane, split=ChargeSplit(monthly_part=monthly, credit_part=credit)),
        follow_up_job_id=None,
    )


@pytest.mark.parametrize(
    ("settled", "expected"),
    [
        pytest.param(None, [], id="no-settlement"),
        pytest.param(
            SettledAttempt(lane="live", charge=None, follow_up_job_id=None), [], id="no-charge"
        ),
        pytest.param(
            _settled(700, 0),
            [call(_COUNTER, {"lane": "live", "part": "monthly"}, amount=700)],
            id="monthly-only",
        ),
        pytest.param(
            _settled(0, 300),
            [call(_COUNTER, {"lane": "live", "part": "credit"}, amount=300)],
            id="credit-only",
        ),
        pytest.param(
            _settled(200, 600, lane="backfill"),
            [
                call(_COUNTER, {"lane": "backfill", "part": "monthly"}, amount=200),
                call(_COUNTER, {"lane": "backfill", "part": "credit"}, amount=600),
            ],
            id="split",
        ),
    ],
)
def test_record_charge_counts_each_nonzero_part_by_lane(
    settled: SettledAttempt | None, expected: list[object]
) -> None:
    with patch("gubbi.extraction.charging.record_counter") as record_counter:
        record_charge(settled)

    assert record_counter.call_args_list == expected


_UNMETERED = MetricNames.EXTRACTION_LLM_UNMETERED_CALLS


@pytest.mark.parametrize(
    ("settled", "unmetered", "expected"),
    [
        pytest.param(
            _settled(0, 0, lane="backfill"),
            3,
            [call(_UNMETERED, {"lane": "backfill"}, amount=3)],
            id="unmetered",
        ),
        pytest.param(
            SettledAttempt(lane="live", charge=None, follow_up_job_id=None),
            1,
            [call(_UNMETERED, {"lane": "live"}, amount=1)],
            id="unmetered-without-a-charge",
        ),
        pytest.param(_settled(700, 0), 0, [], id="all-metered"),
        pytest.param(None, 2, [], id="no-settlement"),
    ],
)
def test_record_unmetered_counts_a_settled_attempts_unmetered_calls_by_lane(
    settled: SettledAttempt | None, unmetered: int, expected: list[object]
) -> None:
    with patch("gubbi.extraction.charging.record_counter") as record_counter:
        record_unmetered(settled, unmetered)

    assert record_counter.call_args_list == expected
