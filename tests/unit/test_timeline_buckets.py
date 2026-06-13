"""Unit tests for the timeline count-row mapper and the SQL bucket expression.

``count_rows_to_buckets`` maps already-aggregated Postgres ``GROUP BY`` rows
into ``TimelineBucket`` models -- it preserves the repository's ascending order
and skips malformed rows. ``_timeline_bucket_expr`` is the pure SQL-fragment
builder that selects day vs month bucketing in Postgres; both are tested here
without a database.
"""

from __future__ import annotations

import pytest

from gubbi.api.v1.web.timeline import count_rows_to_buckets
from gubbi.storage.repositories.entries import _timeline_bucket_expr


def test_empty_input_returns_empty() -> None:
    # Arrange / Act
    result = count_rows_to_buckets([])

    # Assert
    assert result == []


def test_rows_mapped_to_buckets_in_order() -> None:
    # Arrange -- repository returns rows pre-aggregated and pre-sorted
    rows = [
        {"bucket": "2026-06-10", "entry_count": 2, "conversation_count": 1},
        {"bucket": "2026-06-11", "entry_count": 1, "conversation_count": 0},
    ]

    # Act
    result = count_rows_to_buckets(rows)

    # Assert
    assert [b.model_dump() for b in result] == [
        {"date": "2026-06-10", "entry_count": 2, "conversation_count": 1},
        {"date": "2026-06-11", "entry_count": 1, "conversation_count": 0},
    ]


def test_month_buckets_passed_through() -> None:
    # Arrange -- month grouping already done in SQL; mapper just relays keys
    rows = [
        {"bucket": "2026-05", "entry_count": 1, "conversation_count": 0},
        {"bucket": "2026-06", "entry_count": 1, "conversation_count": 1},
    ]

    # Act
    result = count_rows_to_buckets(rows)

    # Assert
    assert [b.date for b in result] == ["2026-05", "2026-06"]


def test_order_preserved_from_repository() -> None:
    # Arrange -- mapper must not re-sort; it trusts the repo's ORDER BY
    rows = [
        {"bucket": "2026-06-09", "entry_count": 1, "conversation_count": 0},
        {"bucket": "2026-06-11", "entry_count": 1, "conversation_count": 0},
        {"bucket": "2026-06-12", "entry_count": 1, "conversation_count": 0},
    ]

    # Act
    result = count_rows_to_buckets(rows)

    # Assert
    assert [b.date for b in result] == ["2026-06-09", "2026-06-11", "2026-06-12"]


def test_malformed_bucket_rows_skipped() -> None:
    # Arrange -- a missing/blank bucket key must not crash the mapper
    rows = [
        {"bucket": "2026-06-10", "entry_count": 1, "conversation_count": 0},
        {"bucket": "", "entry_count": 5, "conversation_count": 5},
        {"entry_count": 9, "conversation_count": 9},
    ]

    # Act
    result = count_rows_to_buckets(rows)

    # Assert
    assert [b.model_dump() for b in result] == [
        {"date": "2026-06-10", "entry_count": 1, "conversation_count": 0},
    ]


def test_bucket_expr_day_keeps_full_date() -> None:
    # Act
    expr = _timeline_bucket_expr("day", "e.date")

    # Assert
    assert expr == "e.date::text"


def test_bucket_expr_month_truncates() -> None:
    # Act
    expr = _timeline_bucket_expr("month", "c.created_at::date")

    # Assert
    assert expr == "to_char(c.created_at::date, 'YYYY-MM')"


def test_bucket_expr_rejects_unknown_unit() -> None:
    # Act / Assert
    with pytest.raises(ValueError, match="Invalid bucket"):
        _timeline_bucket_expr("week", "e.date")
