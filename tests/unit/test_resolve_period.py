"""Unit tests for the year-ceiling boundaries of ``resolve_period``.

The fixture round-trip in ``test_period_fixtures.py`` pins the resolver's full
output matrix, but these four inputs sit at the edge of the four-digit civil
year range and are asserted directly here so a regression is caught even if
the fixture matrix itself changes shape.
"""

from __future__ import annotations

from datetime import date

import pytest

from gubbi.tools.context import resolve_period

pytestmark = pytest.mark.unit

_ANY_TODAY = date(2026, 6, 11)


def test_resolves_last_representable_december() -> None:
    date_from, date_to, label = resolve_period("9999-12", today=_ANY_TODAY)

    assert date_from == "9999-12-01"
    assert date_to == "9999-12-31"
    assert label == "December 9999"


def test_resolves_last_representable_iso_week() -> None:
    date_from, date_to, label = resolve_period("9999-W51", today=_ANY_TODAY)

    assert date_from == "9999-12-20"
    assert date_to == "9999-12-26"
    assert label == "Week 51, 9999"


def test_rejects_iso_week_whose_range_crosses_year_10000() -> None:
    with pytest.raises(ValueError, match="Invalid period"):
        resolve_period("9999-W52", today=_ANY_TODAY)


def test_resolves_iso_week_crossing_into_previous_year() -> None:
    """Regression guard: the year-10000 fix must not touch the low end.

    Week 1 of year 1000 starts in year 999 -- this must keep resolving after
    the ``_month_end``/ISO-week fixes, since the fix is one-sided (upper bound
    only).
    """
    date_from, date_to, label = resolve_period("1000-w01", today=_ANY_TODAY)

    assert date_from == "0999-12-30"
    assert date_to == "1000-01-05"
    assert label == "Week 1, 1000"
