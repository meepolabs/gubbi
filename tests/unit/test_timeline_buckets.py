"""Unit tests for the timeline bucket-aggregation helper.

``aggregate_buckets`` is a pure function over title-only date-range rows, so it
is tested here without a database: day vs month grouping, the
entry/conversation split, ascending order, and empty input.
"""

from __future__ import annotations

from gubbi.api.v1.web.timeline import aggregate_buckets


def test_empty_input_returns_empty() -> None:
    # Arrange / Act
    result = aggregate_buckets([], "day")

    # Assert
    assert result == []


def test_day_grouping_counts_entries_and_conversations() -> None:
    # Arrange
    rows = [
        {"doc_type": "entry", "updated": "2026-06-10"},
        {"doc_type": "entry", "updated": "2026-06-10"},
        {"doc_type": "conversation", "updated": "2026-06-10"},
        {"doc_type": "entry", "updated": "2026-06-11"},
    ]

    # Act
    result = aggregate_buckets(rows, "day")

    # Assert
    assert [b.model_dump() for b in result] == [
        {"date": "2026-06-10", "entry_count": 2, "conversation_count": 1},
        {"date": "2026-06-11", "entry_count": 1, "conversation_count": 0},
    ]


def test_month_grouping_collapses_days() -> None:
    # Arrange
    rows = [
        {"doc_type": "entry", "updated": "2026-05-31"},
        {"doc_type": "entry", "updated": "2026-06-01"},
        {"doc_type": "conversation", "updated": "2026-06-20"},
    ]

    # Act
    result = aggregate_buckets(rows, "month")

    # Assert
    assert [b.model_dump() for b in result] == [
        {"date": "2026-05", "entry_count": 1, "conversation_count": 0},
        {"date": "2026-06", "entry_count": 1, "conversation_count": 1},
    ]


def test_buckets_sorted_ascending() -> None:
    # Arrange -- out-of-order input
    rows = [
        {"doc_type": "entry", "updated": "2026-06-12"},
        {"doc_type": "entry", "updated": "2026-06-09"},
        {"doc_type": "entry", "updated": "2026-06-11"},
    ]

    # Act
    result = aggregate_buckets(rows, "day")

    # Assert
    assert [b.date for b in result] == ["2026-06-09", "2026-06-11", "2026-06-12"]


def test_unparseable_date_rows_skipped() -> None:
    # Arrange -- a malformed/missing date should not crash aggregation
    rows = [
        {"doc_type": "entry", "updated": "2026-06-10"},
        {"doc_type": "entry", "updated": ""},
        {"doc_type": "entry"},
    ]

    # Act
    result = aggregate_buckets(rows, "day")

    # Assert
    assert [b.model_dump() for b in result] == [
        {"date": "2026-06-10", "entry_count": 1, "conversation_count": 0},
    ]
