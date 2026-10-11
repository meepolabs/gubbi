"""Unit tests for wallet arithmetic: weighting usage, splitting a charge, folding a balance."""

from __future__ import annotations

from datetime import date

import pytest

from gubbi.extraction.llm.provider import LLMUsage
from gubbi.extraction.wallet import (
    Balance,
    ChargeSplit,
    MonthTotals,
    UsageWeights,
    balance,
    split_charge,
    weighted,
)

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


_ALLOWANCE = 1_000
_JAN, _FEB, _MAR = date(2026, 1, 1), date(2026, 2, 1), date(2026, 3, 1)


def _month(
    month: date, *, monthly_part: int = 0, credit_part: int = 0, granted: int = 0
) -> MonthTotals:
    return MonthTotals(
        month=month, monthly_part=monthly_part, credit_part=credit_part, granted=granted
    )


@pytest.mark.parametrize(
    ("monthly_amount", "months", "current", "expected"),
    [
        pytest.param(_ALLOWANCE, [], _JAN, Balance(1_000, 0), id="no-history"),
        pytest.param(
            _ALLOWANCE,
            [_month(_JAN, monthly_part=1_000, credit_part=300)],
            _JAN,
            Balance(0, -300),
            id="first-month-debt-is-not-repaid-in-the-month-it-was-made",
        ),
        pytest.param(
            _ALLOWANCE,
            [_month(_JAN, monthly_part=1_000, credit_part=200, granted=500)],
            _FEB,
            Balance(1_000, 300),
            id="no-debt-credit-carries-over",
        ),
        pytest.param(
            _ALLOWANCE,
            [_month(_JAN, credit_part=500)],
            _FEB,
            Balance(500, 0),
            id="debt-repaid-from-next-allowance",
        ),
        pytest.param(
            _ALLOWANCE,
            [_month(_JAN, credit_part=500)],
            _MAR,
            Balance(1_000, 0),
            id="debt-repaid-by-idle-month-leaves-full-allowance-after",
        ),
        pytest.param(
            _ALLOWANCE,
            [_month(_JAN, credit_part=1_500)],
            _FEB,
            Balance(0, -500),
            id="debt-over-one-allowance-first-month-of-repayment",
        ),
        pytest.param(
            _ALLOWANCE,
            [_month(_JAN, credit_part=1_500)],
            _MAR,
            Balance(500, 0),
            id="debt-over-one-allowance-repaid-in-second-month",
        ),
        pytest.param(
            _ALLOWANCE,
            [_month(_JAN, credit_part=500), _month(_FEB, granted=2_000)],
            _FEB,
            Balance(500, 2_000),
            id="grant-mid-month-leaves-the-repayment-fixed",
        ),
        pytest.param(
            _ALLOWANCE,
            [_month(_JAN, credit_part=400), _month(_FEB, monthly_part=600, credit_part=200)],
            _FEB,
            Balance(0, -200),
            id="spending-in-the-repayment-month",
        ),
        pytest.param(
            500,
            [_month(_JAN, monthly_part=800, granted=300)],
            _JAN,
            Balance(0, 300),
            id="lowered-allowance-mid-month-leaves-credit-untouched",
        ),
        pytest.param(
            _ALLOWANCE,
            [_month(_FEB, credit_part=100), _month(_JAN, granted=50)],
            _MAR,
            Balance(950, 0),
            id="months-in-any-order",
        ),
        pytest.param(
            _ALLOWANCE,
            [_month(_MAR, credit_part=700)],
            _FEB,
            Balance(1_000, 0),
            id="months-after-the-current-one-are-ignored",
        ),
    ],
)
def test_balance(
    monthly_amount: int, months: list[MonthTotals], current: date, expected: Balance
) -> None:
    result = balance(monthly_amount, months, current)

    assert result == expected
    assert result.available == expected.monthly_left + expected.credit
