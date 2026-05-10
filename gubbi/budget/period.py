"""Shared period_start derivation for budget keying."""

from __future__ import annotations

from datetime import UTC, date, datetime


def current_period_start() -> date:
    """Return UTC first-of-month for the current instant."""
    return datetime.now(UTC).replace(day=1, hour=0, minute=0, second=0, microsecond=0).date()
