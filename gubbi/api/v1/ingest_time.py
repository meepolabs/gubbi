"""Timestamp field type for ingest payloads.

Accepts one canonical RFC 3339 layout, the same one the extension's schema
emits, and normalizes every value to UTC. Pydantic's own datetime parsing is
far laxer (date-only, epoch numbers, a space separator, lowercase ``t``/``z``,
``+HHMM`` offsets), so a client could otherwise send values the extension
would never produce and that parity checks between the two never see.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Annotated, Any

from pydantic import AfterValidator, BeforeValidator

__all__: list[str] = ["IngestTimestamp"]

_CANONICAL = re.compile(
    r"(?!0000)[0-9]{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])"
    r"T(?:[01][0-9]|2[0-3]):[0-5][0-9](?::[0-5][0-9](?:\.[0-9]+)?)?"
    r"(?:Z|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])?"
)


def _require_canonical(value: Any) -> Any:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and _CANONICAL.fullmatch(value):
        return value
    raise ValueError(
        "timestamp must be an RFC 3339 date-time: YYYY-MM-DDTHH:MM[:SS[.fraction]][Z|+HH:MM|-HH:MM]"
    )


def _to_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    try:
        return value.astimezone(UTC)
    except OverflowError:
        raise ValueError("timestamp is out of range once converted to UTC") from None


IngestTimestamp = Annotated[datetime, BeforeValidator(_require_canonical), AfterValidator(_to_utc)]
