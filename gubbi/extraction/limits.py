"""Extraction budget limits and the single accessor every caller reads them through.

Limits are counted in conversations and role-filtered input tokens. Today every
user gets the deployment-wide configured values; per-user limits replace the body
of ``limits_for`` without changing its callers.
"""

from dataclasses import dataclass
from datetime import timedelta
from uuid import UUID

from gubbi.config import get_settings

__all__: list[str] = ["ExtractionLimits", "global_monthly_token_ceiling", "limits_for"]


@dataclass(frozen=True, slots=True)
class ExtractionLimits:
    """Extraction allowances and scheduling windows in effect for one user."""

    monthly_conversations: int
    monthly_tokens: int
    first_sync_conversations: int
    first_sync_tokens: int
    # None only when budgets are disabled; settings require it otherwise.
    global_monthly_token_ceiling: int | None
    backfill_stop_ratio: float
    nightly_window_hour: int
    onboarding_window: timedelta


def limits_for(user_id: UUID) -> ExtractionLimits:
    """Return the extraction limits that apply to ``user_id``."""
    llm = get_settings().llm
    return ExtractionLimits(
        monthly_conversations=llm.monthly_conversation_limit,
        monthly_tokens=llm.monthly_token_limit,
        first_sync_conversations=llm.first_sync_conversation_limit,
        first_sync_tokens=llm.first_sync_token_limit,
        global_monthly_token_ceiling=global_monthly_token_ceiling(),
        backfill_stop_ratio=llm.backfill_stop_ratio,
        nightly_window_hour=llm.nightly_window_hour,
        onboarding_window=timedelta(days=llm.onboarding_window_days),
    )


def global_monthly_token_ceiling() -> int | None:
    """Return the deployment-wide monthly filtered-token ceiling; None when unset."""
    return get_settings().llm.global_monthly_token_ceiling
