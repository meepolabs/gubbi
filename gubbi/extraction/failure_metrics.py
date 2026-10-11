"""The ``extraction.jobs.failed_total`` counter.

Recorded once per failure written to a job row, after the write commits, by
the code's retry class and whether the row runs again. Never carries a user,
job or conversation id.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gubbi.extraction.retry_policy import retry_class
from gubbi.telemetry.attrs import MetricNames
from gubbi.telemetry.metrics import record_counter

if TYPE_CHECKING:
    from gubbi.extraction.retry_policy import ErrorCode, FailureOutcome

__all__: list[str] = ["record_job_failures"]


def record_job_failures(code: ErrorCode, outcome: FailureOutcome, count: int = 1) -> None:
    """Count ``count`` job failures written with ``code`` and ``outcome``."""
    record_counter(
        MetricNames.EXTRACTION_JOBS_FAILED,
        {"error_class": retry_class(code).value, "outcome": outcome.value},
        amount=count,
    )
