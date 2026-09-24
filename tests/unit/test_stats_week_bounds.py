"""Unit tests for the stats repo's pure week-boundary helper.

``user_week_bounds`` derives the current Monday-start week window in a given
IANA timezone with no database access, so it is exercised here directly.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, tzinfo
from zoneinfo import ZoneInfo

import pytest

import gubbi.validation
from gubbi.storage.repositories.stats import user_week_bounds

# Sunday 12:00 UTC is already Monday 02:00 in Pacific/Kiritimati (UTC+14), so
# the UTC date and the local date fall in different Monday-start weeks.
_SPLIT_INSTANT = datetime(2026, 9, 20, 12, 0, tzinfo=UTC)
_SPLIT_TZ = "Pacific/Kiritimati"


class _FrozenDatetime(datetime):
    @classmethod
    def now(cls, tz: tzinfo | None = None) -> _FrozenDatetime:
        frozen = _SPLIT_INSTANT.astimezone(tz)
        return cls.combine(frozen.date(), frozen.timetz())


def _today_in(timezone: str, now: datetime | None = None) -> date:
    instant = now if now is not None else datetime.now(UTC)
    return instant.astimezone(ZoneInfo(timezone)).date()


class TestUserWeekBounds:
    """Monday-start week window in the user's timezone."""

    def test_returns_monday_to_sunday_span(self) -> None:
        # Act
        week_from, week_to = user_week_bounds("UTC")

        # Assert
        assert week_from.weekday() == 0  # Monday
        assert week_to.weekday() == 6  # Sunday
        assert (week_to - week_from).days == 6

    @pytest.mark.parametrize("timezone", ["UTC", "Pacific/Kiritimati", "Pacific/Pago_Pago"])
    def test_today_falls_within_the_window(self, timezone: str) -> None:
        # Act
        week_from, week_to = user_week_bounds(timezone)

        # Assert
        today = _today_in(timezone)
        assert week_from <= today <= week_to

    def test_window_follows_local_date_when_utc_date_differs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        monkeypatch.setattr(gubbi.validation, "datetime", _FrozenDatetime)

        # Act
        week_from, week_to = user_week_bounds(_SPLIT_TZ)

        # Assert
        today = _today_in(_SPLIT_TZ, now=_SPLIT_INSTANT)
        assert week_from <= today <= week_to
        assert (week_from, week_to) == (date(2026, 9, 21), date(2026, 9, 27))

    def test_unknown_timezone_falls_back_to_utc(self) -> None:
        # Act
        fallback = user_week_bounds("Not/AReal_Zone")
        utc = user_week_bounds("UTC")

        # Assert: fallback path resolves to a valid Monday-start window.
        assert fallback[0].weekday() == 0
        assert (fallback[1] - fallback[0]).days == 6
        # On the same call day both resolve identically (UTC fallback).
        assert fallback == utc
