"""Unit tests for the stats repo's pure week-boundary helper.

``user_week_bounds`` derives the current Monday-start week window in a given
IANA timezone with no database access, so it is exercised here directly.
"""

from __future__ import annotations

from datetime import date

from gubbi.storage.repositories.stats import user_week_bounds


class TestUserWeekBounds:
    """Monday-start week window in the user's timezone."""

    def test_returns_monday_to_sunday_span(self) -> None:
        # Act
        week_from, week_to = user_week_bounds("UTC")

        # Assert
        assert week_from.weekday() == 0  # Monday
        assert week_to.weekday() == 6  # Sunday
        assert (week_to - week_from).days == 6

    def test_today_falls_within_the_window(self) -> None:
        # Act
        week_from, week_to = user_week_bounds("UTC")

        # Assert
        today = date.today()
        assert week_from <= today <= week_to

    def test_unknown_timezone_falls_back_to_utc(self) -> None:
        # Act
        fallback = user_week_bounds("Not/AReal_Zone")
        utc = user_week_bounds("UTC")

        # Assert: fallback path resolves to a valid Monday-start window.
        assert fallback[0].weekday() == 0
        assert (fallback[1] - fallback[0]).days == 6
        # On the same call day both resolve identically (UTC fallback).
        assert fallback == utc
