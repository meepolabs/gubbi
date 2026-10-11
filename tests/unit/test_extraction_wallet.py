"""Unit tests for wallet arithmetic: weighting usage and splitting a charge."""

from __future__ import annotations

import pytest

from gubbi.extraction.llm.provider import LLMUsage
from gubbi.extraction.wallet import ChargeSplit, UsageWeights, split_charge, weighted

pytestmark = pytest.mark.unit

WEIGHTS = UsageWeights(input=1.0, cached_input=0.1, output=4.0)


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        pytest.param(LLMUsage(), 0, id="nothing"),
        pytest.param(LLMUsage(input=1000), 1000, id="uncached-input"),
        pytest.param(LLMUsage(output=250), 1000, id="output"),
        # Input includes cached: 4 uncached + 17379 cached at 0.1 + 188 output at 4.
        pytest.param(
            LLMUsage(input=17383, cached_input=17379, output=188, reasoning=140),
            4 + 1738 + 752,
            id="recorded-cached-response",
        ),
        pytest.param(LLMUsage(input=100, cached_input=100), 10, id="all-cached"),
        # Reasoning is part of output and is not charged a second time.
        pytest.param(LLMUsage(output=100, reasoning=60), 400, id="reasoning-inside-output"),
        pytest.param(LLMUsage(input=10, cached_input=50), 1, id="cached-capped-at-input"),
        pytest.param(LLMUsage(input=5, cached_input=5), 0, id="rounds-half-to-even"),
        pytest.param(LLMUsage(input=6, cached_input=6), 1, id="rounds-up"),
    ],
)
def test_weighted(usage: LLMUsage, expected: int) -> None:
    assert weighted(usage, WEIGHTS) == expected


def test_weighted_uses_the_given_weights() -> None:
    usage = LLMUsage(input=100, cached_input=40, output=10)

    assert weighted(usage, UsageWeights(input=2.0, cached_input=0.5, output=3.0)) == 120 + 20 + 30


@pytest.mark.parametrize(
    ("monthly_left", "amount", "expected"),
    [
        pytest.param(1000, 400, ChargeSplit(400, 0), id="monthly-covers-all"),
        pytest.param(1000, 1000, ChargeSplit(1000, 0), id="monthly-exactly"),
        pytest.param(300, 1000, ChargeSplit(300, 700), id="monthly-then-credit"),
        pytest.param(0, 1000, ChargeSplit(0, 1000), id="credit-only"),
        pytest.param(-50, 1000, ChargeSplit(0, 1000), id="overdrawn-monthly-is-nothing"),
        pytest.param(500, 0, ChargeSplit(0, 0), id="zero-charge"),
    ],
)
def test_split_charge(monthly_left: int, amount: int, expected: ChargeSplit) -> None:
    split = split_charge(monthly_left, amount)

    assert split == expected
    assert split.monthly_part + split.credit_part == amount
