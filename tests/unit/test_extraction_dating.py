"""Tests for event-date resolution of extracted entries."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import pytest

from gubbi.extraction.dating import resolve_event_date
from gubbi.models.conversation import Message

TODAY = date(2026, 5, 20)
CREATED = date(2026, 1, 2)


@dataclass(frozen=True)
class _Msg:
    timestamp: datetime | str | None


MESSAGES = (
    _Msg("2026-03-01T09:00:00"),
    _Msg(None),
    _Msg(datetime(2026, 4, 15, 23, 30)),
    _Msg("2026-02-10T08:00:00Z"),
)


@pytest.mark.parametrize(
    ("model_date", "source_index", "messages", "expected"),
    [
        pytest.param("2026-02-14", 0, MESSAGES, date(2026, 2, 14), id="valid-model-date-wins"),
        pytest.param(
            " 2026-05-20 ", 0, MESSAGES, date(2026, 3, 1), id="backfill-sync-day-falls-to-source"
        ),
        pytest.param(
            "2026-04-16", None, MESSAGES, date(2026, 4, 15), id="after-latest-message-uses-latest"
        ),
        pytest.param(
            "2026-04-15", 0, MESSAGES, date(2026, 4, 15), id="equal-to-latest-message-kept"
        ),
        pytest.param(
            "2026-05-20",
            0,
            (*MESSAGES, _Msg("2026-05-20T07:00:00")),
            TODAY,
            id="latest-message-today-allows-today",
        ),
        pytest.param(
            "2026-03-01", 0, (_Msg(None),), CREATED, id="undated-after-created-uses-created"
        ),
        pytest.param(
            "2026-05-25",
            0,
            (_Msg("2026-03-01T09:00:00"), _Msg("2026-06-01T09:00:00")),
            date(2026, 3, 1),
            id="model-date-after-today-rejected-despite-later-message",
        ),
        pytest.param(
            "2025-12-24", 0, (_Msg(None),), date(2025, 12, 24), id="undated-before-created-kept"
        ),
        pytest.param(
            "2026-05-21", 0, MESSAGES, date(2026, 3, 1), id="future-model-date-falls-to-source"
        ),
        pytest.param(
            "last tuesday",
            2,
            MESSAGES,
            date(2026, 4, 15),
            id="unparseable-model-date-falls-to-source",
        ),
        pytest.param(None, 0, MESSAGES, date(2026, 3, 1), id="no-model-date-uses-source-str"),
        pytest.param(None, 2, MESSAGES, date(2026, 4, 15), id="no-model-date-uses-source-datetime"),
        pytest.param(
            None, 1, MESSAGES, date(2026, 4, 15), id="source-without-timestamp-uses-latest"
        ),
        pytest.param(None, 4, MESSAGES, date(2026, 4, 15), id="index-past-end-uses-latest"),
        pytest.param(None, -1, MESSAGES, date(2026, 4, 15), id="negative-index-uses-latest"),
        pytest.param(None, None, MESSAGES, date(2026, 4, 15), id="no-index-uses-latest"),
        pytest.param(
            None,
            0,
            (_Msg("not a timestamp"), _Msg("2026-03-05T00:00:00")),
            date(2026, 3, 5),
            id="unparseable-source-timestamp-uses-latest",
        ),
        pytest.param(None, 0, (_Msg(None), _Msg("")), CREATED, id="no-timestamps-uses-created"),
        pytest.param("2026-09-01", 0, (), CREATED, id="future-model-date-no-messages-uses-created"),
    ],
)
def test_resolve_event_date_fallback_order(
    model_date: str | None,
    source_index: int | None,
    messages: tuple[_Msg, ...],
    expected: date,
) -> None:
    result = resolve_event_date(model_date, source_index, messages, CREATED, TODAY)

    assert result == expected


def test_no_dates_anywhere_returns_created_not_today() -> None:
    created = date(2025, 11, 3)
    today = date(2026, 10, 10)

    result = resolve_event_date(None, None, [_Msg(None), _Msg(None)], created, today)

    assert result == created
    assert result != today


def test_accepts_conversation_message_model() -> None:
    messages = [
        Message(role="user", content="hi", timestamp="2026-03-07T10:00:00+00:00"),
        Message(role="assistant", content="hello"),
    ]

    result = resolve_event_date(None, 1, messages, CREATED, TODAY)

    assert result == date(2026, 3, 7)
