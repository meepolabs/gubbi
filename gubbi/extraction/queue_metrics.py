"""Queue and budget gauges the extraction scheduler records once per tick.

Every gauge carries bounded attributes only: lane, defer reason or token kind.
A lane, defer reason or token kind with no rows reports 0, so a drained queue
reads as empty rather than as a missing series.

``extraction.budget.global_used_ratio`` is the weighted tokens charged this
month over the global ceiling, unclamped: above 1 means the month overran the
ceiling. It is not recorded while budgets are disabled or no ceiling is set:
there is no limit to be near, and a 0 would read as headroom.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gubbi.extraction.admission import DeferReason
from gubbi.extraction.intake import Lane
from gubbi.extraction.spend_rules import global_ceiling
from gubbi.storage.repositories import extraction_queue as queue_repo
from gubbi.storage.repositories import wallet as wallet_repo
from gubbi.telemetry.attrs import MetricNames
from gubbi.telemetry.metrics import record_gauge

if TYPE_CHECKING:
    import asyncpg

__all__: list[str] = ["record_queue_metrics"]


async def record_queue_metrics(conn: asyncpg.Connection, *, budget_enabled: bool) -> None:
    """Read the queue figures and this month's charges, then record every gauge."""
    snapshot = await queue_repo.queue_snapshot(conn)
    usage = await wallet_repo.month_usage(conn)
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
    for kind, tokens in (
        ("input", usage.input_tokens),
        ("output", usage.output_tokens),
        ("cached", usage.cached_input_tokens),
    ):
        record_gauge(MetricNames.EXTRACTION_PROVIDER_TOKENS_MONTH, tokens, {"kind": kind})
    ceiling = global_ceiling() if budget_enabled else None
    if ceiling is not None:
        record_gauge(
            MetricNames.EXTRACTION_BUDGET_GLOBAL_USED_RATIO, usage.weighted / ceiling.tokens
        )
