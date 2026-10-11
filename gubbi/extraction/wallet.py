"""Wallet arithmetic: weighting provider usage, splitting a charge, folding a balance.

Pure functions only: no I/O, no logging, no clock reads. Every amount is in
weighted tokens, the unit of the per-user wallet and the global ceiling.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

from gubbi.config import get_settings

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gubbi.extraction.llm.provider import LLMUsage

__all__: list[str] = [
    "Balance",
    "ChargeSplit",
    "MonthTotals",
    "UsageWeights",
    "balance",
    "split_charge",
    "usage_weights",
    "weighted",
]


@dataclass(frozen=True, slots=True)
class UsageWeights:
    """Input-token equivalents of one uncached input, cached input and output token."""

    input: float
    cached_input: float
    output: float


@dataclass(frozen=True, slots=True)
class ChargeSplit:
    """How one charge divides between the monthly allowance and credit."""

    monthly_part: int
    credit_part: int


@dataclass(frozen=True, slots=True)
class MonthTotals:
    """One user's wallet activity in one UTC month (``month`` is its first day)."""

    month: date
    monthly_part: int
    credit_part: int
    granted: int


@dataclass(frozen=True, slots=True)
class Balance:
    """What a user may spend in the current month: allowance left and credit."""

    monthly_left: int
    credit: int

    @property
    def available(self) -> int:
        """Return the allowance left plus credit; negative when debt exceeds the allowance."""
        return self.monthly_left + self.credit


def usage_weights() -> UsageWeights:
    """Return the configured weights."""
    llm = get_settings().llm
    return UsageWeights(
        input=llm.weight_input, cached_input=llm.weight_cached_input, output=llm.weight_output
    )


def weighted(usage: LLMUsage, weights: UsageWeights) -> int:
    """Return ``usage`` in weighted tokens, rounded to the nearest integer.

    The provider's input count includes cached input and its output count
    includes reasoning, so cached tokens are taken out of input before
    weighting and reasoning is not added again. A cached count larger than
    the input count is capped at the input count.
    """
    cached = min(usage.cached_input, usage.input)
    total = (
        (usage.input - cached) * weights.input
        + cached * weights.cached_input
        + usage.output * weights.output
    )
    return round(total)


def split_charge(monthly_left: int, amount: int) -> ChargeSplit:
    """Split ``amount`` into the part the monthly allowance covers and the rest, paid by credit.

    ``monthly_left`` below zero counts as nothing left. Credit takes whatever
    the allowance does not cover, however large; a balance may go negative.
    """
    monthly_part = min(amount, max(monthly_left, 0))
    return ChargeSplit(monthly_part=monthly_part, credit_part=amount - monthly_part)


def _next_month(month: date) -> date:
    return date(month.year + month.month // 12, month.month % 12 + 1, 1)


def balance(monthly_amount: int, months: Sequence[MonthTotals], current_month: date) -> Balance:
    """Fold one user's monthly totals into their balance in ``current_month``.

    Credit may go negative. At the start of each month the allowance first pays
    off the debt carried in, up to the whole allowance; a grant later in the
    month does not change that month's repayment. What the repayment leaves of
    the allowance is spendable that month and expires at its end. Allowance use
    past a lowered allowance is no debt: ``monthly_left`` never goes below zero.

    Every month, from the first with activity through ``current_month``, is
    folded with today's ``monthly_amount``, including months with no activity.
    Totals dated after ``current_month`` are ignored.
    """
    by_month = {m.month: m for m in months}
    month = min([*by_month, current_month])
    credit = 0
    while True:
        totals = by_month.get(month, MonthTotals(month, 0, 0, 0))
        repay = min(max(-credit, 0), monthly_amount)
        credit += repay + totals.granted - totals.credit_part
        if month >= current_month:
            monthly_left = max(monthly_amount - repay - totals.monthly_part, 0)
            return Balance(monthly_left=monthly_left, credit=credit)
        month = _next_month(month)
