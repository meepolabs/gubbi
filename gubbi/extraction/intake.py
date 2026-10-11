"""Intake classification for queued extraction work.

Pure functions only: no I/O, no logging, no clock reads. Callers pass ``now``
and the user's upload history so every decision is reproducible in tests.
"""

from collections.abc import Iterable
from datetime import UTC, datetime, time, timedelta
from enum import StrEnum
from typing import Final

from gubbi.extraction.chunking import estimate_tokens
from gubbi.models.conversation import ConversationTurn, keep_conversation_roles

TOKEN_COUNT_VERSION: Final = 1

PASTE_SOURCES: Final = frozenset({"paste_memories"})
SYNC_SOURCES: Final = frozenset({"extension_chatgpt", "extension_claude", "zip_upload"})


class Lane(StrEnum):
    """Scheduling lane of an extraction job."""

    PASTE = "paste"
    BACKFILL = "backfill"
    LIVE = "live"


def assign_lane(
    request_source: str,
    *,
    activity_at: datetime,
    now: datetime,
    first_upload_at_platform: datetime | None,
    first_upload_at_any: datetime | None,
    onboarding_window: timedelta,
) -> Lane:
    """Return the lane for one conversation of an ingest request.

    ``request_source`` is the ingest request's ``source``. ``activity_at`` is
    the conversation's latest message time. The two ``first_upload_at``
    arguments are the user's earliest prior upload from this conversation's
    platform and from any platform; ``None`` means no such upload exists yet,
    so the current upload is the first one and all of it is history.

    A synced conversation is backfill when its latest message predates the
    user's first upload from the same platform, or when ``now`` is within
    ``onboarding_window`` of the user's first upload of any platform;
    otherwise it is live.

    Raises:
        ValueError: ``request_source`` is not a known ingest source.
    """
    if request_source in PASTE_SOURCES:
        return Lane.PASTE
    if request_source not in SYNC_SOURCES:
        raise ValueError(f"unknown ingest source: {request_source!r}")

    if first_upload_at_platform is None or activity_at < first_upload_at_platform:
        return Lane.BACKFILL
    onboarding_start = now if first_upload_at_any is None else first_upload_at_any
    if now < onboarding_start + onboarding_window:
        return Lane.BACKFILL
    return Lane.LIVE


def next_nightly_window(now: datetime, hour: int) -> datetime:
    """Return the next ``hour``:00 UTC at or after ``now``.

    Exactly on the hour returns ``now`` itself, so work scheduled then is
    runnable immediately rather than a day later.

    Raises:
        ValueError: ``hour`` is outside 0-23 or ``now`` is naive.
    """
    if not 0 <= hour <= 23:
        raise ValueError(f"nightly hour must be within 0-23, got {hour}")
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    now_utc = now.astimezone(UTC)
    today_window = datetime.combine(now_utc.date(), time(hour), tzinfo=UTC)
    if now_utc <= today_window:
        return today_window
    return today_window + timedelta(days=1)


def run_after_for(lane: Lane, now: datetime, nightly_hour: int) -> datetime:
    """Return when a new job in ``lane`` becomes runnable.

    Live work waits for the next nightly window; paste and backfill run now.

    Raises:
        ValueError: as :func:`next_nightly_window`, for a live lane.
    """
    if lane is Lane.LIVE:
        return next_nightly_window(now, nightly_hour)
    return now


def count_filtered_tokens(messages: Iterable[ConversationTurn]) -> int:
    """Estimate the input tokens of the role-filtered transcript.

    Only the turns kept for extraction count, so system prompts and tool
    output never draw on a user's budget. The result is stable for a given
    :data:`TOKEN_COUNT_VERSION`.
    """
    return sum(estimate_tokens(m.content) for m in keep_conversation_roles(messages))
