"""Spend rules for LLM work and the policy accessors every caller reads.

Each kind of LLM job has one rule in :data:`SPEND_RULES`, keyed by its lane:
when a new job runs (``schedule``), the point at which the user's wallet stops
it (``stop``) and the share of the global ceiling it may run up to
(``ceiling_share``). A lane decides order and timing only; every job is paid
from the same wallet, monthly allowance first, then credit. A new kind of job
is a new row.

Policy is separate from metering: :func:`limits_for` returns a user's limits,
never their balance, which lives in the charge and grant tables. Every figure
is in weighted tokens. Today every user gets the deployment-wide configured
values; per-user limits replace the body of :func:`limits_for` without
changing its callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from gubbi.config import get_settings
from gubbi.extraction.intake import Lane, next_nightly_window

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime
    from uuid import UUID

__all__: list[str] = [
    "SPEND_RULES",
    "CeilingShare",
    "GlobalCeiling",
    "Schedule",
    "SpendRule",
    "StopLine",
    "WalletLimits",
    "global_ceiling",
    "global_stopped",
    "limits_for",
    "max_running_cap",
    "run_after_for",
    "user_stopped",
]


class Schedule(StrEnum):
    """When a newly queued job becomes runnable."""

    ON_SAVE = "on_save"
    NIGHTLY = "nightly"
    SOON_AFTER_UPLOAD = "soon_after_upload"


class StopLine(StrEnum):
    """The wallet balance below which a user's jobs of a kind wait."""

    # Stops once nothing is available.
    EXHAUSTED = "exhausted"
    # Stops while less than the background reserve is available.
    RESERVE = "reserve"


class CeilingShare(StrEnum):
    """How much of the global ceiling a kind of job may run up to."""

    FULL = "full"
    BACKGROUND = "background"


@dataclass(frozen=True, slots=True)
class SpendRule:
    """Schedule, wallet stop line and global-ceiling share of one kind of job."""

    schedule: Schedule
    stop: StopLine
    ceiling_share: CeilingShare


SPEND_RULES: Final[Mapping[Lane, SpendRule]] = MappingProxyType(
    {
        Lane.PASTE: SpendRule(Schedule.ON_SAVE, StopLine.EXHAUSTED, CeilingShare.FULL),
        Lane.LIVE: SpendRule(Schedule.NIGHTLY, StopLine.EXHAUSTED, CeilingShare.FULL),
        Lane.BACKFILL: SpendRule(
            Schedule.SOON_AFTER_UPLOAD, StopLine.RESERVE, CeilingShare.BACKGROUND
        ),
    }
)


@dataclass(frozen=True, slots=True)
class WalletLimits:
    """Wallet policy in effect for one user."""

    monthly_amount: int
    background_reserve: int
    running_cap: int


@dataclass(frozen=True, slots=True)
class GlobalCeiling:
    """Deployment-wide monthly ceiling and the fraction of it background work may use."""

    tokens: int
    background_ratio: float

    def stop_at(self, share: CeilingShare) -> float:
        """Monthly weighted usage at which work with ``share`` stops."""
        if share is CeilingShare.BACKGROUND:
            return self.tokens * self.background_ratio
        return float(self.tokens)


def limits_for(user_id: UUID) -> WalletLimits:
    """Return the wallet policy that applies to ``user_id``."""
    llm = get_settings().llm
    return WalletLimits(
        monthly_amount=llm.wallet_monthly_amount,
        background_reserve=llm.wallet_background_reserve,
        running_cap=llm.running_job_cap,
    )


def max_running_cap() -> int:
    """Return the largest running-job cap any user has; every user's today."""
    return get_settings().llm.running_job_cap


def global_ceiling() -> GlobalCeiling | None:
    """Return the deployment-wide monthly ceiling; None when none is configured."""
    llm = get_settings().llm
    if llm.global_monthly_token_ceiling is None:
        return None
    return GlobalCeiling(
        tokens=llm.global_monthly_token_ceiling, background_ratio=llm.backfill_stop_ratio
    )


def user_stopped(rule: SpendRule, available: int, limits: WalletLimits) -> bool:
    """Whether a user with ``available`` weighted tokens must wait to run a job under ``rule``."""
    if rule.stop is StopLine.RESERVE:
        return available < limits.background_reserve
    return available <= 0


def global_stopped(rule: SpendRule, used: int, ceiling: GlobalCeiling) -> bool:
    """Whether ``used`` weighted tokens this month stop every job under ``rule``."""
    return used >= ceiling.stop_at(rule.ceiling_share)


def run_after_for(lane: Lane, now: datetime, nightly_hour: int) -> datetime:
    """Return when a new job in ``lane`` becomes runnable under its rule's schedule.

    Raises:
        ValueError: as :func:`gubbi.extraction.intake.next_nightly_window`, for
            a nightly schedule.
    """
    if SPEND_RULES[lane].schedule is Schedule.NIGHTLY:
        return next_nightly_window(now, nightly_hour)
    return now
