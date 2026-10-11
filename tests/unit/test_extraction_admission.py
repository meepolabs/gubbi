"""Behavior tests for the extraction admission policy."""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest

from gubbi.extraction.admission import (
    Candidate,
    Decision,
    DeferReason,
    UserUsage,
    decide,
    order_candidates,
    user_usage_from_totals,
)
from gubbi.extraction.intake import Lane
from gubbi.extraction.limits import ExtractionLimits
from gubbi.storage.repositories.extraction_queue import UserPeriodUsage

T0 = datetime(2026, 1, 1, tzinfo=UTC)
ALICE = UUID(int=1)
BOB = UUID(int=2)

LIMITS = ExtractionLimits(
    monthly_conversations=3,
    monthly_tokens=1_000,
    first_sync_conversations=2,
    first_sync_tokens=500,
    global_monthly_token_ceiling=10_000,
    backfill_stop_ratio=0.8,
    nightly_window_hour=3,
    onboarding_window=timedelta(days=7),
)


def _job(
    lane: Lane,
    tokens: int = 100,
    *,
    user: UUID = ALICE,
    minute: int | None = 0,
) -> Candidate:
    activity = None if minute is None else T0 + timedelta(minutes=minute)
    return Candidate(
        job_id=uuid4(), user_id=user, lane=lane, activity_at=activity, filtered_tokens=tokens
    )


def _reasons(decisions: list[Decision]) -> list[DeferReason | None]:
    return [d.defer_reason for d in decisions]


@pytest.mark.unit
def test_order_is_paste_live_backfill_newest_first_within_lane() -> None:
    backfill_old = _job(Lane.BACKFILL, minute=1)
    live_undated = _job(Lane.LIVE, minute=None)
    live_new = _job(Lane.LIVE, minute=9)
    paste_old = _job(Lane.PASTE, minute=0)
    backfill_new = _job(Lane.BACKFILL, minute=5)
    paste_new = _job(Lane.PASTE, minute=3)

    ordered = order_candidates(
        [backfill_old, live_undated, live_new, paste_old, backfill_new, paste_new]
    )

    assert ordered == [paste_new, paste_old, live_new, live_undated, backfill_new, backfill_old]


@pytest.mark.unit
def test_decisions_follow_admission_order() -> None:
    backfill = _job(Lane.BACKFILL)
    paste = _job(Lane.PASTE)

    decisions = decide([backfill, paste], {}, 0, LIMITS)

    assert [d.job_id for d in decisions] == [paste.job_id, backfill.job_id]


@pytest.mark.unit
def test_every_candidate_gets_a_decision_and_none_is_refused() -> None:
    candidates = [_job(lane, 400, user=user) for lane in Lane for user in (ALICE, BOB)] * 3
    usage = {ALICE: UserUsage(monthly_conversations=2, monthly_tokens=900)}

    decisions = decide(candidates, usage, 9_000, LIMITS)

    assert sorted(d.job_id for d in decisions) == sorted(c.job_id for c in candidates)
    assert {d.defer_reason for d in decisions} <= {None, DeferReason.BUDGET, DeferReason.GLOBAL}


@pytest.mark.unit
@pytest.mark.parametrize("lane", [Lane.PASTE, Lane.LIVE, Lane.BACKFILL])
def test_budgets_disabled_admits_everything(lane: Lane) -> None:
    candidates = [_job(lane, 10**9) for _ in range(5)]
    usage = {ALICE: UserUsage(10**6, 10**12, 10**6, 10**12)}

    decisions = decide(candidates, usage, 10**15, None)

    assert all(d.admitted for d in decisions)


@pytest.mark.unit
def test_no_global_ceiling_never_defers_global() -> None:
    limits = replace(LIMITS, global_monthly_token_ceiling=None)

    decisions = decide([_job(Lane.BACKFILL, 100)], {}, 10**15, limits)

    assert _reasons(decisions) == [None]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("lane", "usage", "tokens", "expected"),
    [
        pytest.param(Lane.LIVE, UserUsage(2, 900), 100, None, id="live-lands-on-both-caps"),
        pytest.param(Lane.LIVE, UserUsage(3, 0), 1, DeferReason.BUDGET, id="live-conversation-cap"),
        pytest.param(Lane.PASTE, UserUsage(0, 900), 101, DeferReason.BUDGET, id="paste-token-cap"),
        pytest.param(
            Lane.BACKFILL,
            UserUsage(3, 1_000, 0, 0),
            500,
            None,
            id="backfill-inside-first-sync-ignores-monthly-cap",
        ),
        pytest.param(
            Lane.BACKFILL,
            UserUsage(3, 0, 2, 0),
            1,
            DeferReason.BUDGET,
            id="backfill-past-first-sync-conversations-hits-monthly-cap",
        ),
        pytest.param(
            Lane.BACKFILL,
            UserUsage(0, 990, 0, 450),
            60,
            None,
            id="backfill-token-overflow-fits-monthly",
        ),
        pytest.param(
            Lane.BACKFILL,
            UserUsage(0, 990, 0, 450),
            61,
            DeferReason.BUDGET,
            id="backfill-token-overflow-exceeds-monthly",
        ),
    ],
)
def test_user_budget(
    lane: Lane, usage: UserUsage, tokens: int, expected: DeferReason | None
) -> None:
    decisions = decide([_job(lane, tokens)], {ALICE: usage}, 0, LIMITS)

    assert _reasons(decisions) == [expected]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("lane", "global_used", "tokens", "expected"),
    [
        pytest.param(Lane.BACKFILL, 7_900, 100, None, id="backfill-lands-on-stop-ratio"),
        pytest.param(
            Lane.BACKFILL, 7_900, 101, DeferReason.GLOBAL, id="backfill-crosses-stop-ratio"
        ),
        pytest.param(Lane.BACKFILL, 8_000, 1, DeferReason.GLOBAL, id="backfill-at-stop-ratio"),
        pytest.param(Lane.LIVE, 8_000, 100, None, id="live-past-stop-ratio"),
        pytest.param(Lane.PASTE, 9_900, 100, None, id="paste-lands-on-ceiling"),
        pytest.param(Lane.LIVE, 9_900, 101, DeferReason.GLOBAL, id="live-crosses-ceiling"),
        pytest.param(Lane.PASTE, 10_000, 1, DeferReason.GLOBAL, id="paste-at-ceiling"),
    ],
)
def test_global_ceiling(
    lane: Lane, global_used: int, tokens: int, expected: DeferReason | None
) -> None:
    decisions = decide([_job(lane, tokens)], {}, global_used, LIMITS)

    assert _reasons(decisions) == [expected]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("lane", "usage", "global_used", "expected"),
    [
        pytest.param(Lane.LIVE, UserUsage(0, 0), 0, [None], id="live-on-empty-month"),
        pytest.param(Lane.LIVE, UserUsage(0, 1), 0, [DeferReason.BUDGET], id="live-on-used-month"),
        pytest.param(
            Lane.LIVE, UserUsage(0, 0), 8_000, [DeferReason.GLOBAL], id="live-past-global-headroom"
        ),
        pytest.param(
            Lane.BACKFILL,
            UserUsage(0, 0, 2, 500),
            0,
            [None],
            id="backfill-past-first-sync-on-empty-month",
        ),
        pytest.param(
            Lane.BACKFILL,
            UserUsage(1, 0, 2, 500),
            0,
            [DeferReason.BUDGET],
            id="backfill-past-first-sync-on-used-month",
        ),
    ],
)
def test_oversized_job(
    lane: Lane, usage: UserUsage, global_used: int, expected: list[DeferReason | None]
) -> None:
    decisions = decide([_job(lane, 2_500)], {ALICE: usage}, global_used, LIMITS)

    assert _reasons(decisions) == expected


@pytest.mark.unit
def test_oversized_job_consumes_the_month_for_later_work_in_the_same_call() -> None:
    oversized, small = _job(Lane.LIVE, 2_500, minute=2), _job(Lane.LIVE, 1, minute=1)

    decisions = decide([oversized, small], {}, 0, LIMITS)

    assert _reasons(decisions) == [None, DeferReason.BUDGET]


@pytest.mark.unit
def test_global_reason_wins_over_budget() -> None:
    usage = {ALICE: UserUsage(3, 1_000)}

    decisions = decide([_job(Lane.LIVE, 100)], usage, 10_000, LIMITS)

    assert _reasons(decisions) == [DeferReason.GLOBAL]


@pytest.mark.unit
def test_user_usage_accumulates_within_one_call() -> None:
    first, second, other_user = (
        _job(Lane.LIVE, 600, minute=2),
        _job(Lane.LIVE, 600, minute=1),
        _job(Lane.LIVE, 600, user=BOB, minute=0),
    )

    decisions = decide([first, second, other_user], {}, 0, LIMITS)

    assert _reasons(decisions) == [None, DeferReason.BUDGET, None]


@pytest.mark.unit
def test_global_usage_accumulates_within_one_call() -> None:
    candidates = [_job(Lane.LIVE, 600, user=uuid4(), minute=m) for m in (3, 2, 1)]

    decisions = decide(candidates, {}, 8_900, LIMITS)

    assert _reasons(decisions) == [None, DeferReason.GLOBAL, DeferReason.GLOBAL]


@pytest.mark.unit
def test_first_sync_allowance_accumulates_within_one_call() -> None:
    candidates = [_job(Lane.BACKFILL, 10, minute=m) for m in (5, 4, 3, 2, 1, 0)]

    decisions = decide(candidates, {ALICE: UserUsage(2, 0)}, 0, LIMITS)

    assert _reasons(decisions) == [None, None, None, *[DeferReason.BUDGET] * 3]


@pytest.mark.unit
def test_live_is_admitted_before_backfill_draws_spare_monthly() -> None:
    backfill = _job(Lane.BACKFILL, 401, minute=9)
    live = _job(Lane.LIVE, 400, minute=0)
    usage = {ALICE: UserUsage(0, 200, 2, 500)}

    decisions = decide([backfill, live], usage, 0, LIMITS)

    assert decisions == [Decision(live.job_id, None), Decision(backfill.job_id, DeferReason.BUDGET)]


@pytest.mark.unit
def test_backfill_spare_monthly_waits_behind_deferred_live_work() -> None:
    big_live = _job(Lane.LIVE, 900)
    small_backfill = _job(Lane.BACKFILL, 50)
    usage = {ALICE: UserUsage(0, 200, 2, 500)}

    decisions = decide([big_live, small_backfill], usage, 0, LIMITS)

    assert _reasons(decisions) == [DeferReason.BUDGET, DeferReason.BUDGET]


@pytest.mark.unit
def test_deferred_live_does_not_block_backfill_inside_first_sync() -> None:
    big_live = _job(Lane.LIVE, 900)
    small_backfill = _job(Lane.BACKFILL, 50)
    usage = {ALICE: UserUsage(0, 200, 0, 0)}

    decisions = decide([big_live, small_backfill], usage, 0, LIMITS)

    assert _reasons(decisions) == [DeferReason.BUDGET, None]


@pytest.mark.unit
def test_deferred_live_of_another_user_does_not_block_backfill() -> None:
    bob_live = _job(Lane.LIVE, 900, user=BOB)
    alice_backfill = _job(Lane.BACKFILL, 50)
    usage = {BOB: UserUsage(0, 200), ALICE: UserUsage(0, 200, 2, 500)}

    decisions = decide([bob_live, alice_backfill], usage, 0, LIMITS)

    assert _reasons(decisions) == [DeferReason.BUDGET, None]


@pytest.mark.unit
def test_input_usage_mapping_is_not_mutated() -> None:
    usage = {ALICE: UserUsage(0, 0)}

    decide([_job(Lane.LIVE, 100)], usage, 0, LIMITS)

    assert usage == {ALICE: UserUsage(0, 0)}


@pytest.mark.unit
@pytest.mark.parametrize(
    ("totals", "expected"),
    [
        pytest.param(
            (5, 900, 0, 0, 0, 0), UserUsage(5, 900, 0, 0), id="no-backfill-is-all-monthly"
        ),
        pytest.param(
            (3, 600, 2, 500, 2, 500), UserUsage(1, 100, 2, 500), id="backfill-inside-allowance"
        ),
        pytest.param(
            (4, 800, 4, 800, 4, 800), UserUsage(2, 300, 4, 800), id="period-crosses-allowance"
        ),
        pytest.param(
            (3, 300, 3, 300, 9, 900), UserUsage(3, 300, 9, 900), id="allowance-used-earlier"
        ),
        pytest.param(
            (3, 300, 3, 300, 4, 600), UserUsage(2, 100, 4, 600), id="allowance-split-across-periods"
        ),
    ],
)
def test_user_usage_from_totals(totals: tuple[int, ...], expected: UserUsage) -> None:
    period_conv, period_tok, period_bf_conv, period_bf_tok, life_bf_conv, life_bf_tok = totals

    usage = user_usage_from_totals(
        period_conversations=period_conv,
        period_tokens=period_tok,
        period_backfill_conversations=period_bf_conv,
        period_backfill_tokens=period_bf_tok,
        lifetime_backfill_conversations=life_bf_conv,
        lifetime_backfill_tokens=life_bf_tok,
        limits=LIMITS,
    )

    assert usage == expected


@pytest.mark.unit
def test_usage_from_totals_matches_charges_made_by_decide() -> None:
    backfill = [_job(Lane.BACKFILL, 200, minute=m) for m in (5, 4, 3)]
    live = _job(Lane.LIVE, 100)
    limits = replace(LIMITS, monthly_conversations=10)
    decisions = decide([*backfill, live], {}, 0, limits)
    assert all(d.admitted for d in decisions)

    rebuilt = user_usage_from_totals(
        period_conversations=4,
        period_tokens=700,
        period_backfill_conversations=3,
        period_backfill_tokens=600,
        lifetime_backfill_conversations=3,
        lifetime_backfill_tokens=600,
        limits=limits,
    )
    follow_up = decide(
        [_job(Lane.LIVE, 1_000 - rebuilt.monthly_tokens)], {ALICE: rebuilt}, 0, limits
    )

    assert rebuilt == UserUsage(2, 200, 3, 600)
    assert _reasons(follow_up) == [None]


@pytest.mark.unit
def test_repository_usage_feeds_usage_from_totals_by_name() -> None:
    totals = UserPeriodUsage(
        period_conversations=4,
        period_tokens=700,
        period_backfill_conversations=3,
        period_backfill_tokens=600,
        lifetime_backfill_conversations=3,
        lifetime_backfill_tokens=600,
    )

    usage = user_usage_from_totals(**asdict(totals), limits=LIMITS)

    assert usage == UserUsage(2, 200, 3, 600)
