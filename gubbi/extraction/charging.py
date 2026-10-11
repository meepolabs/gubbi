"""Charging a finished job attempt to its user's wallet.

Called in the transaction that moves a job out of ``running``, and only when
that transition changed the row, so each attempt is charged at most once
whatever the retries, revivals or duplicate deliveries; the charge table's
unique ``(job_id, attempt)`` is the backstop. The charge is the attempt's
provider-reported usage in weighted tokens. An attempt that reported no usage
is not charged.

A completed attempt whose conversation was saved again while it ran read
older content, so a follow-up job is queued for the conversation in the same
transaction, in the same lane, scheduled by the lane's spend rule.

The ``extraction.charges.weighted_total`` counter is recorded by the caller
after the transaction commits, through :func:`record_charge`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from gubbi.config import get_settings
from gubbi.extraction.intake import Lane
from gubbi.extraction.spend_rules import limits_for, run_after_for
from gubbi.extraction.wallet import usage_weights, weighted
from gubbi.storage.repositories import extraction_jobs
from gubbi.storage.repositories import wallet as wallet_repo
from gubbi.telemetry.attrs import MetricNames
from gubbi.telemetry.metrics import record_counter

if TYPE_CHECKING:
    from uuid import UUID

    import asyncpg

    from gubbi.extraction.llm.provider import LLMUsage

__all__: list[str] = ["SettledAttempt", "record_charge", "settle_attempt"]


@dataclass(frozen=True, slots=True)
class SettledAttempt:
    """What settling one attempt wrote: its charge (None when none) and any follow-up job."""

    lane: str
    charge: wallet_repo.Charge | None
    follow_up_job_id: UUID | None


async def settle_attempt(
    conn: asyncpg.Connection, job_id: UUID, usage: LLMUsage, *, completed: bool
) -> SettledAttempt | None:
    """Charge the attempt that just ended and, for a completion, queue any follow-up.

    Must run in the transaction that ended the attempt, after the terminal
    transition changed the row. Returns None when the job row is not visible.
    """
    attempt = await extraction_jobs.finished_attempt(conn, job_id)
    if attempt is None:
        return None
    charge = await wallet_repo.charge(
        conn,
        wallet_repo.AttemptUsage(
            user_id=attempt.user_id,
            job_kind=attempt.lane,
            job_id=job_id,
            attempt=attempt.attempt,
            input_tokens=usage.input,
            cached_input_tokens=usage.cached_input,
            output_tokens=usage.output,
            weighted=weighted(usage, usage_weights()),
        ),
        limits_for(attempt.user_id).monthly_amount,
    )
    follow_up = None
    if completed and attempt.resaved_during_run:
        run_after = run_after_for(
            Lane(attempt.lane), datetime.now(UTC), get_settings().llm.nightly_window_hour
        )
        follow_up = await extraction_jobs.create_follow_up(conn, attempt, run_after)
    return SettledAttempt(lane=attempt.lane, charge=charge, follow_up_job_id=follow_up)


def record_charge(settled: SettledAttempt | None) -> None:
    """Count a committed charge's weighted tokens by lane and paying balance."""
    if settled is None or settled.charge is None:
        return
    split = settled.charge.split
    for part, amount in (("monthly", split.monthly_part), ("credit", split.credit_part)):
        if amount > 0:
            record_counter(
                MetricNames.EXTRACTION_CHARGES_WEIGHTED,
                {"lane": settled.lane, "part": part},
                amount=amount,
            )
