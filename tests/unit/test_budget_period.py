"""Unit tests for gubbi.budget.period.current_period_start."""

from __future__ import annotations

from datetime import UTC, date, datetime
from unittest.mock import patch

from gubbi.budget.period import current_period_start


def test_returns_first_of_current_month() -> None:
    # Arrange: freeze time at 2026-05-15 14:30:00 UTC
    frozen = datetime(2026, 5, 15, 14, 30, 0, tzinfo=UTC)
    with patch("gubbi.budget.period.datetime") as mock_dt:
        mock_dt.now.return_value = frozen
        mock_dt.now.side_effect = None

        # Act
        result = current_period_start()

    # Assert
    assert result == date(2026, 5, 1)


def test_returns_date_not_datetime() -> None:
    # The return type must be a date, not a datetime, so Redis keys are stable strings.
    frozen = datetime(2026, 3, 20, 9, 0, 0, tzinfo=UTC)
    with patch("gubbi.budget.period.datetime") as mock_dt:
        mock_dt.now.return_value = frozen
        mock_dt.now.side_effect = None
        result = current_period_start()
    assert isinstance(result, date)
    assert not isinstance(result, datetime)


def test_idempotent_within_same_month() -> None:
    # Two calls on different days in the same month must return the same value.
    day5 = datetime(2026, 7, 5, 0, 0, 0, tzinfo=UTC)
    day28 = datetime(2026, 7, 28, 23, 59, 59, tzinfo=UTC)
    with patch("gubbi.budget.period.datetime") as mock_dt:
        mock_dt.now.return_value = day5
        mock_dt.now.side_effect = None
        r1 = current_period_start()
    with patch("gubbi.budget.period.datetime") as mock_dt:
        mock_dt.now.return_value = day28
        mock_dt.now.side_effect = None
        r2 = current_period_start()
    assert r1 == r2 == date(2026, 7, 1)


def test_first_of_month_is_still_first() -> None:
    # Called on the first of the month: result is still the first of that month.
    first = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
    with patch("gubbi.budget.period.datetime") as mock_dt:
        mock_dt.now.return_value = first
        mock_dt.now.side_effect = None
        result = current_period_start()
    assert result == date(2026, 1, 1)


def test_last_day_of_month() -> None:
    last = datetime(2026, 4, 30, 23, 59, 59, 999999, tzinfo=UTC)
    with patch("gubbi.budget.period.datetime") as mock_dt:
        mock_dt.now.return_value = last
        mock_dt.now.side_effect = None
        result = current_period_start()
    assert result == date(2026, 4, 1)
