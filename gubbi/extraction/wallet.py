"""Wallet arithmetic: weighting provider usage and splitting a charge.

Pure functions only: no I/O, no logging, no clock reads. Every amount is in
weighted tokens, the unit of the per-user wallet and the global ceiling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from gubbi.config import get_settings

if TYPE_CHECKING:
    from gubbi.extraction.llm.provider import LLMUsage

__all__: list[str] = [
    "ChargeSplit",
    "UsageWeights",
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
