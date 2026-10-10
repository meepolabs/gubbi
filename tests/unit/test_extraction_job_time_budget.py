"""Time budget of the extraction job: Arq timeout, in-job deadline, orphan sweep."""

from __future__ import annotations

from arq.worker import Function

from gubbi.constants import (
    ARQ_JOB_TIMEOUT_SECS,
    EXTRACTION_JOB_DEADLINE_SECS,
    EXTRACTION_JOB_TIMEOUT_SECS,
)
from gubbi.extraction.jobs.extract_conversation import extract_conversation
from gubbi.extraction.orphan_cleanup import _RUNNING_THRESHOLD_SECS
from gubbi.extraction.worker import WorkerSettings


def _registered(name: str) -> Function:
    matches = [
        fn for fn in WorkerSettings.functions if isinstance(fn, Function) and fn.name == name
    ]
    assert len(matches) == 1, f"expected exactly one registered Function named {name!r}"
    return matches[0]


def test_extract_conversation_registered_with_extraction_timeout() -> None:
    registered = _registered("extract_conversation")

    assert registered.coroutine is extract_conversation
    assert registered.timeout_s == EXTRACTION_JOB_TIMEOUT_SECS


def test_shared_job_timeout_stays_at_default() -> None:
    assert WorkerSettings.job_timeout == ARQ_JOB_TIMEOUT_SECS


def test_deadline_falls_before_arq_timeout() -> None:
    assert EXTRACTION_JOB_DEADLINE_SECS < EXTRACTION_JOB_TIMEOUT_SECS


def test_running_sweep_threshold_exceeds_extraction_timeout() -> None:
    assert _RUNNING_THRESHOLD_SECS > EXTRACTION_JOB_TIMEOUT_SECS
