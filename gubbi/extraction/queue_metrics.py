"""Queue and budget gauges the extraction scheduler records once per tick.

Every gauge carries bounded attributes only: lane, defer reason or token kind.
A lane, defer reason or token kind with no rows reports 0, so a drained queue
reads as empty rather than as a missing series.

``extraction.budget.global_used_ratio`` is filtered tokens admitted this budget
period over the global ceiling. It is not recorded while budgets are disabled
or no ceiling is set: there is no limit to be near, and a 0 would read as
headroom.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gubbi.extraction.admission import DeferReason
from gubbi.extraction.intake import Lane
from gubbi.extraction.limits import global_monthly_token_ceiling
from gubbi.storage.repositories import extraction_queue as queue_repo
from gubbi.telemetry.attrs import MetricNames
from gubbi.telemetry.metrics import record_gauge

if TYPE_CHECKING:
    from datetime import date

    import asyncpg

__all__: list[str] = ["record_queue_metrics"]


async def record_queue_metrics(
    conn: asyncpg.Connection, period: date, *, budget_enabled: bool
) -> None:
    """Read the queue figures for budget period ``period`` and record every gauge."""
    snapshot = await queue_repo.queue_snapshot(conn, period)
    record_gauge(
        MetricNames.EXTRACTION_QUEUE_OLDEST_RUNNABLE_AGE, snapshot.oldest_runnable_age_seconds
    )
    for lane in Lane:
        record_gauge(
            MetricNames.EXTRACTION_QUEUE_DEPTH,
            snapshot.depth_by_lane.get(lane, 0),
            {"lane": lane.value},
        )
    for reason in sorted({*DeferReason, *snapshot.deferred_by_reason}):
        record_gauge(
            MetricNames.EXTRACTION_QUEUE_DEFERRED,
            snapshot.deferred_by_reason.get(reason, 0),
            {"defer_reason": reason},
        )
    for kind, tokens in snapshot.period_provider_tokens.items():
        record_gauge(MetricNames.EXTRACTION_PROVIDER_TOKENS_MONTH, tokens, {"kind": kind})
    ceiling = global_monthly_token_ceiling() if budget_enabled else None
    if ceiling is not None:
        record_gauge(
            MetricNames.EXTRACTION_BUDGET_GLOBAL_USED_RATIO,
            snapshot.period_filtered_tokens / ceiling,
        )
