"""Behavior tests for the extraction admission policy."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest

from gubbi.extraction.admission import (
    Budget,
    Candidate,
    Decision,
    DeferReason,
    UserState,
    decide,
    order_candidates,
)
from gubbi.extraction.intake import Lane
from gubbi.extraction.spend_rules import GlobalCeiling, WalletLimits

pytestmark = pytest.mark.unit

T0 = datetime(2026, 1, 1, tzinfo=UTC)
ALICE = UUID(int=1)
BOB = UUID(int=2)

LIMITS = WalletLimits(monthly_amount=5_000, background_reserve=1_000, running_cap=3)
CEILING = GlobalCeiling(tokens=10_000, background_ratio=0.8)
ROOMY = 100


def _job(lane: Lane, *, user: UUID = ALICE, minute: int | None = 0) -> Candidate:
    activity = None if minute is None else T0 + timedelta(minutes=minute)
    return Candidate(job_id=uuid4(), user_id=user, lane=lane, activity_at=activity)


def _state(available: int = 5_000, inflight: int = 0, cap: int = 3) -> UserState:
    return UserState(
        limits=replace(LIMITS, running_cap=cap), available=available, inflight=inflight
    )


def _budget(global_used: int = 0) -> Budget:
    return Budget(global_used=global_used, ceiling=CEILING)


def _outcome(decision: Decision) -> str:
    if decision.defer_reason is not None:
        return decision.defer_reason.value
    return "admitted" if decision.admitted else "queued"


def test_order_is_paste_live_backfill_newest_first_within_lane() -> None:
    old_live = _job(Lane.LIVE, minute=1)
    new_live = _job(Lane.LIVE, minute=5)
    undated_live = _job(Lane.LIVE, minute=None)
    backfill = _job(Lane.BACKFILL, minute=9)
    paste = _job(Lane.PASTE, minute=0)

    ordered = order_candidates([backfill, old_live, undated_live, paste, new_live])

    assert ordered == [paste, new_live, old_live, undated_live, backfill]


def test_decisions_follow_admission_order() -> None:
    backfill, live, paste = _job(Lane.BACKFILL), _job(Lane.LIVE), _job(Lane.PASTE)

    decisions = decide([backfill, live, paste], {ALICE: _state()}, _budget(), ROOMY)

    assert [d.job_id for d in decisions] == [paste.job_id, live.job_id, backfill.job_id]


@pytest.mark.parametrize(
    ("lane", "available", "global_used", "inflight", "expected"),
    [
        # Global share: paste and live run to the whole ceiling, backfill to 80%.
        pytest.param(Lane.PASTE, 5_000, 9_999, 0, "admitted", id="paste-just-under-ceiling"),
        pytest.param(Lane.PASTE, 5_000, 10_000, 0, "global", id="paste-at-ceiling"),
        pytest.param(Lane.LIVE, 5_000, 10_000, 0, "global", id="live-at-ceiling"),
        pytest.param(Lane.LIVE, 5_000, 12_000, 0, "global", id="live-over-ceiling"),
        pytest.param(Lane.BACKFILL, 5_000, 7_999, 0, "admitted", id="backfill-under-80"),
        pytest.param(Lane.BACKFILL, 5_000, 8_000, 0, "global", id="backfill-at-80"),
        pytest.param(Lane.LIVE, 5_000, 8_000, 0, "admitted", id="live-past-backfill-share"),
        # Stop line: paste and live stop at nothing available, backfill below the reserve.
        pytest.param(Lane.PASTE, 1, 0, 0, "admitted", id="paste-one-left"),
        pytest.param(Lane.PASTE, 0, 0, 0, "budget", id="paste-nothing-left"),
        pytest.param(Lane.LIVE, -50, 0, 0, "budget", id="live-negative-balance"),
        pytest.param(Lane.BACKFILL, 1_000, 0, 0, "admitted", id="backfill-at-reserve"),
        pytest.param(Lane.BACKFILL, 999, 0, 0, "budget", id="backfill-below-reserve"),
        pytest.param(Lane.LIVE, 999, 0, 0, "admitted", id="live-below-reserve"),
        # Running cap: in flight counts dispatched-not-started jobs too.
        pytest.param(Lane.LIVE, 5_000, 0, 2, "admitted", id="live-under-cap"),
        pytest.param(Lane.LIVE, 5_000, 0, 3, "queued", id="live-at-cap"),
        pytest.param(Lane.BACKFILL, 5_000, 0, 3, "queued", id="backfill-at-cap"),
        # Order of checks: global first, then the stop line, then the cap.
        pytest.param(Lane.LIVE, 0, 10_000, 0, "global", id="global-wins-over-budget"),
        pytest.param(Lane.LIVE, 0, 0, 3, "budget", id="budget-wins-over-cap"),
    ],
)
def test_decide(lane: Lane, available: int, global_used: int, inflight: int, expected: str) -> None:
    job = _job(lane)

    [decision] = decide([job], {ALICE: _state(available, inflight)}, _budget(global_used), ROOMY)

    assert _outcome(decision) == expected


@pytest.mark.parametrize("lane", list(Lane))
def test_budgets_disabled_ignores_wallet_and_ceiling(lane: Lane) -> None:
    [decision] = decide([_job(lane)], {ALICE: _state(available=-1_000_000)}, None, ROOMY)

    assert _outcome(decision) == "admitted"


def test_budgets_disabled_still_applies_the_running_cap() -> None:
    [decision] = decide([_job(Lane.PASTE)], {ALICE: _state(inflight=3)}, None, ROOMY)

    assert _outcome(decision) == "queued"


def test_no_global_ceiling_never_defers_global() -> None:
    budget = Budget(global_used=10**12, ceiling=None)

    [decision] = decide([_job(Lane.BACKFILL)], {ALICE: _state()}, budget, ROOMY)

    assert _outcome(decision) == "admitted"


def test_dispatches_in_one_call_count_toward_the_running_cap() -> None:
    jobs = [_job(Lane.LIVE, minute=m) for m in range(4)]

    decisions = decide(jobs, {ALICE: _state(inflight=1)}, _budget(), ROOMY)

    assert [_outcome(d) for d in decisions] == ["admitted", "admitted", "queued", "queued"]


def test_one_users_cap_does_not_hold_back_another_user() -> None:
    alice = [_job(Lane.PASTE, minute=m) for m in range(5)]
    bob = _job(Lane.BACKFILL, user=BOB)

    decisions = decide([*alice, bob], {ALICE: _state(), BOB: _state()}, _budget(), ROOMY)

    by_id = {d.job_id: _outcome(d) for d in decisions}
    assert sorted(by_id[j.job_id] for j in alice) == ["admitted"] * 3 + ["queued"] * 2
    assert by_id[bob.job_id] == "admitted"


def test_capacity_bounds_admissions_across_users() -> None:
    jobs = [_job(Lane.LIVE, user=ALICE), _job(Lane.LIVE, user=BOB), _job(Lane.BACKFILL)]

    decisions = decide(jobs, {ALICE: _state(), BOB: _state()}, _budget(), capacity=2)

    assert [_outcome(d) for d in decisions] == ["admitted", "admitted", "queued"]


def test_admission_adds_nothing_to_balance_or_global_usage() -> None:
    jobs = [_job(Lane.LIVE, minute=m) for m in range(3)]

    decisions = decide(jobs, {ALICE: _state(available=1)}, _budget(global_used=9_999), ROOMY)

    assert all(d.admitted for d in decisions)


def test_a_deferred_user_does_not_affect_another_user() -> None:
    broke = _job(Lane.LIVE, user=ALICE)
    funded = _job(Lane.LIVE, user=BOB)

    decisions = decide(
        [broke, funded], {ALICE: _state(available=0), BOB: _state()}, _budget(), ROOMY
    )

    assert {d.job_id: _outcome(d) for d in decisions} == {
        broke.job_id: "budget",
        funded.job_id: "admitted",
    }


def test_deferred_decisions_carry_their_user_and_lane() -> None:
    job = _job(Lane.BACKFILL, user=BOB)

    [decision] = decide([job], {BOB: _state(available=0)}, _budget(), ROOMY)

    assert (decision.user_id, decision.lane, decision.defer_reason) == (
        BOB,
        Lane.BACKFILL,
        DeferReason.BUDGET,
    )
