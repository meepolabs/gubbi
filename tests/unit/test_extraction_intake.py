"""Behavior tests for intake classification: lanes, nightly window, token count."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone

import pytest

from gubbi.extraction.chunking import estimate_tokens
from gubbi.extraction.intake import (
    Lane,
    assign_lane,
    count_filtered_tokens,
    next_nightly_window,
)

WINDOW = timedelta(days=7)
SHORT_WINDOW = timedelta(days=2)
DAY0 = datetime(2026, 1, 1, 12, tzinfo=UTC)


def _day(n: int) -> datetime:
    return DAY0 + timedelta(days=n)


@dataclass(frozen=True)
class _Msg:
    role: str
    content: str


@pytest.mark.parametrize(
    ("source", "activity_at", "now", "first_platform", "first_any", "window", "expected"),
    [
        pytest.param(
            "paste_memories", _day(30), _day(30), None, None, WINDOW, Lane.PASTE, id="paste"
        ),
        pytest.param(
            "extension_chatgpt",
            _day(-5),
            _day(0),
            None,
            None,
            WINDOW,
            Lane.BACKFILL,
            id="first-ever",
        ),
        pytest.param(
            "extension_claude",
            _day(1),
            _day(3),
            None,
            _day(0),
            SHORT_WINDOW,
            Lane.BACKFILL,
            id="second-platform-first-seen-day-3-after-window",
        ),
        pytest.param(
            "extension_chatgpt",
            _day(2),
            _day(3),
            _day(0),
            _day(0),
            WINDOW,
            Lane.BACKFILL,
            id="inside-onboarding-window",
        ),
        pytest.param(
            "extension_claude",
            _day(-40),
            _day(40),
            _day(5),
            _day(0),
            WINDOW,
            Lane.BACKFILL,
            id="after-window-history-older-than-platform-first-upload",
        ),
        pytest.param(
            "extension_chatgpt",
            _day(39),
            _day(40),
            _day(0),
            _day(0),
            WINDOW,
            Lane.LIVE,
            id="after-window-continued-old-thread",
        ),
        pytest.param(
            "zip_upload",
            _day(-1),
            _day(40),
            _day(0),
            _day(0),
            WINDOW,
            Lane.BACKFILL,
            id="zip-history",
        ),
        pytest.param(
            "zip_upload", _day(39), _day(40), _day(0), _day(0), WINDOW, Lane.LIVE, id="zip-live"
        ),
        pytest.param(
            "extension_claude",
            _day(10),
            _day(40),
            _day(10),
            _day(0),
            WINDOW,
            Lane.LIVE,
            id="activity-equal-to-platform-first-upload-is-live",
        ),
        pytest.param(
            "extension_chatgpt",
            _day(7),
            _day(7),
            _day(0),
            _day(0),
            WINDOW,
            Lane.LIVE,
            id="now-at-onboarding-window-end-is-live",
        ),
    ],
)
def test_assign_lane_classifies_by_source_platform_and_onboarding(
    source: str,
    activity_at: datetime,
    now: datetime,
    first_platform: datetime | None,
    first_any: datetime | None,
    window: timedelta,
    expected: Lane,
) -> None:
    lane = assign_lane(
        source,
        activity_at=activity_at,
        now=now,
        first_upload_at_platform=first_platform,
        first_upload_at_any=first_any,
        onboarding_window=window,
    )

    assert lane is expected


def test_assign_lane_rejects_unknown_source() -> None:
    with pytest.raises(ValueError, match="unknown ingest source"):
        assign_lane(
            "agent",
            activity_at=_day(0),
            now=_day(0),
            first_upload_at_platform=None,
            first_upload_at_any=None,
            onboarding_window=WINDOW,
        )


@pytest.mark.parametrize(
    ("now", "expected"),
    [
        pytest.param(
            datetime(2026, 3, 10, 1, 30, tzinfo=UTC),
            datetime(2026, 3, 10, 3, tzinfo=UTC),
            id="before-hour-today",
        ),
        pytest.param(
            datetime(2026, 3, 10, 3, tzinfo=UTC),
            datetime(2026, 3, 10, 3, tzinfo=UTC),
            id="exactly-on-hour-is-now",
        ),
        pytest.param(
            datetime(2026, 3, 10, 3, 0, 1, tzinfo=UTC),
            datetime(2026, 3, 11, 3, tzinfo=UTC),
            id="after-hour-tomorrow",
        ),
        pytest.param(
            datetime(2026, 3, 10, 23, 30, tzinfo=timezone(timedelta(hours=-5))),
            datetime(2026, 3, 12, 3, tzinfo=UTC),
            id="non-utc-input-uses-utc-day",
        ),
    ],
)
def test_next_nightly_window_returns_next_hour_in_utc(now: datetime, expected: datetime) -> None:
    assert next_nightly_window(now, 3) == expected


@pytest.mark.parametrize(
    ("now", "hour"),
    [
        pytest.param(datetime(2026, 3, 10, tzinfo=UTC), 24, id="hour-too-large"),
        pytest.param(datetime(2026, 3, 10, tzinfo=UTC), -1, id="hour-negative"),
        pytest.param(datetime(2026, 3, 10), 3, id="naive-now"),
    ],
)
def test_next_nightly_window_rejects_invalid_input(now: datetime, hour: int) -> None:
    with pytest.raises(ValueError):
        next_nightly_window(now, hour)


def test_count_filtered_tokens_ignores_system_and_tool_turns() -> None:
    kept = [_Msg("user", "u" * 40), _Msg("assistant", "a" * 25)]
    messages = [
        _Msg("system", "s" * 4000),
        kept[0],
        _Msg("tool", "t" * 4000),
        kept[1],
    ]

    tokens = count_filtered_tokens(messages)

    assert tokens == sum(estimate_tokens(m.content) for m in kept)
    assert tokens > 0
