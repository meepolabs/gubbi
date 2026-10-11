"""Admission policy for queued extraction work.

Pure functions only: no I/O, no logging, no clock reads. The caller passes the
runnable candidates, each user's usage so far, the deployment's token usage for
the current period and the limits; :func:`decide` returns one decision per
candidate. Work is never refused: a candidate that does not fit is deferred
with a reason and evaluated again on a later call.

Budgets, per user, counted in conversations and filtered input tokens:

- Paste and live work draws on the monthly allowance.
- Backfill draws on the first-sync allowance (counted over the user's lifetime)
  first; whatever part of a job exceeds it draws on the monthly allowance,
  which is checked only after the same user's paste and live work in the call.
- A monthly allowance is exhausted when either its conversation or its token
  limit would be exceeded. The exception is a user who has charged nothing to
  the current month: their next job is admitted whatever its size, so a
  conversation larger than the whole allowance does not wait forever. It
  consumes the month and the user's later work waits.

The global ceiling, in filtered tokens per period, stops backfill at
``backfill_stop_ratio`` of the ceiling and all work at the ceiling.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Final
from uuid import UUID

from gubbi.extraction.intake import Lane
from gubbi.extraction.limits import ExtractionLimits

__all__: list[str] = [
    "Candidate",
    "Decision",
    "DeferReason",
    "UserUsage",
    "budget_period",
    "decide",
    "monthly_allowance_exhausted",
    "order_candidates",
    "user_usage_from_totals",
]

_LANE_RANK: Final = MappingProxyType({Lane.PASTE: 0, Lane.LIVE: 1, Lane.BACKFILL: 2})
_NO_ACTIVITY: Final = datetime.min.replace(tzinfo=UTC)


class DeferReason(StrEnum):
    """Why admission held a candidate back."""

    BUDGET = "budget"
    GLOBAL = "global"


@dataclass(frozen=True, slots=True)
class Candidate:
    """One runnable extraction job awaiting admission."""

    job_id: UUID
    user_id: UUID
    lane: Lane
    activity_at: datetime | None
    filtered_tokens: int


@dataclass(frozen=True, slots=True)
class UserUsage:
    """A user's budget usage before the current call.

    ``monthly_*`` is what the current period has charged to the monthly
    allowance: every admitted paste and live job, plus the part of admitted
    backfill that exceeded the first-sync allowance (see
    :func:`user_usage_from_totals`). ``backfill_*`` covers every backfill job
    the user has ever had admitted.
    """

    monthly_conversations: int = 0
    monthly_tokens: int = 0
    backfill_conversations: int = 0
    backfill_tokens: int = 0


@dataclass(frozen=True, slots=True)
class Decision:
    """Admission outcome for one candidate; ``defer_reason`` is None when admitted."""

    job_id: UUID
    defer_reason: DeferReason | None

    @property
    def admitted(self) -> bool:
        """Whether the candidate may run now."""
        return self.defer_reason is None


def order_candidates(candidates: Iterable[Candidate]) -> list[Candidate]:
    """Return candidates in admission order.

    Paste, then live, then backfill; within a lane the newest ``activity_at``
    first, with candidates lacking one last. Ties keep their input order.
    """
    newest_first = sorted(
        candidates,
        key=lambda c: _NO_ACTIVITY if c.activity_at is None else c.activity_at,
        reverse=True,
    )
    return sorted(newest_first, key=lambda c: _LANE_RANK[c.lane])


def decide(
    candidates: Iterable[Candidate],
    usage_by_user: Mapping[UUID, UserUsage],
    global_used: int,
    limits: ExtractionLimits | None,
) -> list[Decision]:
    """Decide admission for every candidate, in admission order.

    ``usage_by_user`` may omit users with no usage. ``global_used`` is the
    filtered tokens admitted across all users in the current period. Each
    admission counts toward the usage seen by every later candidate in the
    same call. ``limits=None`` means budgets are disabled: everything is
    admitted. A ``global_monthly_token_ceiling`` of None means no global
    ceiling. When a candidate breaks both a user and a global limit, the
    reason is ``global``.
    """
    ordered = order_candidates(candidates)
    if limits is None:
        return [Decision(c.job_id, None) for c in ordered]

    usage = dict(usage_by_user)
    deferred_ahead_of_backfill: set[UUID] = set()
    decisions: list[Decision] = []
    for candidate in ordered:
        before = usage.get(candidate.user_id, UserUsage())
        after = _usage_after(candidate, before, limits)
        reason = _global_reason(candidate, global_used, limits) or _budget_reason(
            candidate, before, after, limits, deferred_ahead_of_backfill
        )
        decisions.append(Decision(candidate.job_id, reason))
        if reason is None:
            usage[candidate.user_id] = after
            global_used += candidate.filtered_tokens
        elif candidate.lane is not Lane.BACKFILL:
            deferred_ahead_of_backfill.add(candidate.user_id)
    return decisions


def budget_period(now: datetime) -> date:
    """Return the budget period containing ``now``: the first day of its UTC month.

    Raises:
        ValueError: ``now`` is naive.
    """
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    return now.astimezone(UTC).date().replace(day=1)


def monthly_allowance_exhausted(usage: UserUsage, limits: ExtractionLimits) -> bool:
    """Whether the user's next paste or live job would wait for the next period.

    True once either monthly limit is reached, so even a one-token job would
    not fit, except while the user has charged nothing this period:
    :func:`decide` admits that next job whatever its size.
    """
    if _month_unused(usage):
        return False
    return (
        usage.monthly_conversations >= limits.monthly_conversations
        or usage.monthly_tokens >= limits.monthly_tokens
    )


def user_usage_from_totals(
    *,
    period_conversations: int,
    period_tokens: int,
    period_backfill_conversations: int,
    period_backfill_tokens: int,
    lifetime_backfill_conversations: int,
    lifetime_backfill_tokens: int,
    limits: ExtractionLimits,
) -> UserUsage:
    """Build a user's :class:`UserUsage` from admitted-job totals.

    ``period_*`` count every job of the user admitted in the current period
    (all lanes), ``period_backfill_*`` the backfill subset of those, and
    ``lifetime_backfill_*`` every backfill job ever admitted, this period
    included. The first-sync allowance absorbs backfill in admission order, so
    the monthly charge is the period's non-backfill work plus the part of the
    period's backfill that lay beyond the allowance.
    """
    return UserUsage(
        monthly_conversations=period_conversations
        - period_backfill_conversations
        + _period_overflow(
            period_backfill_conversations,
            lifetime_backfill_conversations,
            limits.first_sync_conversations,
        ),
        monthly_tokens=period_tokens
        - period_backfill_tokens
        + _period_overflow(
            period_backfill_tokens, lifetime_backfill_tokens, limits.first_sync_tokens
        ),
        backfill_conversations=lifetime_backfill_conversations,
        backfill_tokens=lifetime_backfill_tokens,
    )


def _period_overflow(period: int, lifetime: int, allowance: int) -> int:
    return _overflow(period, lifetime - period, allowance)


def _overflow(amount: int, used: int, allowance: int) -> int:
    """Part of ``amount`` beyond what remains of ``allowance`` after ``used``."""
    return amount - min(amount, max(0, allowance - used))


def _usage_after(candidate: Candidate, before: UserUsage, limits: ExtractionLimits) -> UserUsage:
    tokens = candidate.filtered_tokens
    if candidate.lane is not Lane.BACKFILL:
        return UserUsage(
            monthly_conversations=before.monthly_conversations + 1,
            monthly_tokens=before.monthly_tokens + tokens,
            backfill_conversations=before.backfill_conversations,
            backfill_tokens=before.backfill_tokens,
        )
    return UserUsage(
        monthly_conversations=before.monthly_conversations
        + _overflow(1, before.backfill_conversations, limits.first_sync_conversations),
        monthly_tokens=before.monthly_tokens
        + _overflow(tokens, before.backfill_tokens, limits.first_sync_tokens),
        backfill_conversations=before.backfill_conversations + 1,
        backfill_tokens=before.backfill_tokens + tokens,
    )


def _global_reason(
    candidate: Candidate, global_used: int, limits: ExtractionLimits
) -> DeferReason | None:
    ceiling = limits.global_monthly_token_ceiling
    if ceiling is None:
        return None
    stop = ceiling * limits.backfill_stop_ratio if candidate.lane is Lane.BACKFILL else ceiling
    if global_used + candidate.filtered_tokens > stop:
        return DeferReason.GLOBAL
    return None


def _budget_reason(
    candidate: Candidate,
    before: UserUsage,
    after: UserUsage,
    limits: ExtractionLimits,
    deferred_ahead_of_backfill: set[UUID],
) -> DeferReason | None:
    draws_monthly = (
        after.monthly_conversations > before.monthly_conversations
        or after.monthly_tokens > before.monthly_tokens
    )
    if candidate.lane is Lane.BACKFILL:
        if not draws_monthly:
            return None
        if candidate.user_id in deferred_ahead_of_backfill:
            return DeferReason.BUDGET
    if _month_unused(before):
        return None
    within = (
        after.monthly_conversations <= limits.monthly_conversations
        and after.monthly_tokens <= limits.monthly_tokens
    )
    return None if within else DeferReason.BUDGET


def _month_unused(usage: UserUsage) -> bool:
    return usage.monthly_conversations == 0 and usage.monthly_tokens == 0
