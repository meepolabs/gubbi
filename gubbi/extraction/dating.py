"""Event-date resolution for extracted entries.

An extracted entry's date is when the event happened, not when the worker ran.
Resolution order: the model-supplied date (if it is a valid ISO date and not
in the future), then the date of the message the entry was drawn from, then
the conversation's latest message, then the conversation's created date.
The worker's own clock is never a fallback; ``today`` only bounds the model
date.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence


class TimestampedMessage(Protocol):
    """Any message carrying an optional timestamp (datetime or ISO string)."""

    @property
    def timestamp(self) -> datetime | str | None:
        """When the message was sent, if known."""
        ...


def resolve_event_date(
    model_date: str | None,
    source_message_index: int | None,
    messages: Sequence[TimestampedMessage],
    conversation_created: date,
    today: date,
) -> date:
    """Return the event date for an extracted entry."""
    from_model = _parse_model_date(model_date)
    if from_model is not None and from_model <= today:
        return from_model

    from_source = _source_message_date(source_message_index, messages)
    if from_source is not None:
        return from_source

    message_dates = [d for d in (_timestamp_date(m.timestamp) for m in messages) if d is not None]
    if message_dates:
        return max(message_dates)

    return conversation_created


def _parse_model_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        return None


def _source_message_date(index: int | None, messages: Sequence[TimestampedMessage]) -> date | None:
    # Negative indexes are invalid model output, not Python-style offsets from the end.
    if index is None or not 0 <= index < len(messages):
        return None
    return _timestamp_date(messages[index].timestamp)


def _timestamp_date(value: datetime | str | None) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.strip()).date()
    except ValueError:
        return None
