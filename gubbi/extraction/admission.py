"""Admission policy for queued LLM work.

Pure functions only: no I/O, no logging, no clock reads. The caller passes the
runnable candidates, each user's available wallet balance, limits and jobs in
flight, the deployment's weighted usage this month and the global ceiling;
:func:`decide` returns one decision per candidate. Work is never refused: a
candidate held back is either deferred with a reason or left queued for a
later tick.

Each candidate is checked against, in order:

1. the global ceiling: its spend rule's share of the ceiling (``global``);
2. its user's wallet: its spend rule's stop line (``budget``);
3. its user's running-job cap: jobs in flight, counting jobs dispatched but not
   yet started and jobs dispatched earlier in the same call (left queued);
4. the dispatch capacity of the call (left queued).

Budget checks apply only when budgets are enabled; the running-job cap and the
capacity always apply. Charges are written after the work, so admission adds
nothing to any balance; the only state carried from one candidate to the next
is the per-user dispatch count and the capacity used.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from gubbi.extraction.intake import Lane
from gubbi.extraction.spend_rules import SPEND_RULES, global_stopped, user_stopped

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from uuid import UUID

    from gubbi.extraction.spend_rules import GlobalCeiling, WalletLimits

__all__: list[str] = [
    "Budget",
    "Candidate",
    "Decision",
    "DeferReason",
    "UserState",
    "decide",
    "order_candidates",
]

_LANE_RANK: Final = MappingProxyType({Lane.PASTE: 0, Lane.LIVE: 1, Lane.BACKFILL: 2})
_NO_ACTIVITY: Final = datetime.min.replace(tzinfo=UTC)


class DeferReason(StrEnum):
    """Why admission deferred a candidate."""

    BUDGET = "budget"
    GLOBAL = "global"


@dataclass(frozen=True, slots=True)
class Candidate:
    """One runnable job awaiting admission."""

    job_id: UUID
    user_id: UUID
    lane: Lane
    activity_at: datetime | None


@dataclass(frozen=True, slots=True)
class UserState:
    """A user's limits, available balance and jobs in flight before the call.

    ``available`` is ignored while budgets are disabled.
    """

    limits: WalletLimits
    available: int
    inflight: int


@dataclass(frozen=True, slots=True)
class Budget:
    """Deployment-wide budget state; ``ceiling`` None means no global ceiling."""

    global_used: int
    ceiling: GlobalCeiling | None


@dataclass(frozen=True, slots=True)
class Decision:
    """Admission outcome for one candidate.

    ``defer_reason`` set: deferred until a recheck. ``defer_reason`` None and
    ``admitted`` False: left queued for a later call (running-job cap or
    capacity).
    """

    job_id: UUID
    user_id: UUID
    lane: Lane
    admitted: bool
    defer_reason: DeferReason | None = None


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
    users: Mapping[UUID, UserState],
    budget: Budget | None,
    capacity: int,
) -> list[Decision]:
    """Decide admission for every candidate, in admission order.

    ``users`` must hold every candidate's user. ``budget`` None means budgets
    are disabled. At most ``capacity`` candidates are admitted. When a
    candidate breaks both the global ceiling and its user's stop line, the
    reason is ``global``.
    """
    dispatched: dict[UUID, int] = {}
    decisions: list[Decision] = []
    for candidate in order_candidates(candidates):
        state = users[candidate.user_id]
        reason = _defer_reason(candidate, state, budget)
        running = state.inflight + dispatched.get(candidate.user_id, 0)
        admitted = (
            reason is None
            and running < state.limits.running_cap
            and sum(dispatched.values()) < capacity
        )
        if admitted:
            dispatched[candidate.user_id] = dispatched.get(candidate.user_id, 0) + 1
        decisions.append(
            Decision(candidate.job_id, candidate.user_id, candidate.lane, admitted, reason)
        )
    return decisions


def _defer_reason(
    candidate: Candidate, state: UserState, budget: Budget | None
) -> DeferReason | None:
    if budget is None:
        return None
    rule = SPEND_RULES[candidate.lane]
    if budget.ceiling is not None and global_stopped(rule, budget.global_used, budget.ceiling):
        return DeferReason.GLOBAL
    if user_stopped(rule, state.available, state.limits):
        return DeferReason.BUDGET
    return None
