"""Unit tests for the pure parts of ``gubbi.scripts.release_parked_extraction_jobs``."""

from __future__ import annotations

import io
import json

import pytest

from gubbi.extraction.retry_policy import ErrorCode, RetryClass, retry_class
from gubbi.scripts.release_parked_extraction_jobs import (
    TERMINAL_ERROR_CODES,
    GroupCount,
    ReleaseResult,
    _build_parser,
    build_logger,
    error_class_of,
    group_counts,
    log_dry_run,
    log_release,
)


@pytest.mark.parametrize(
    ("error_code", "expected"),
    [
        pytest.param(None, "none", id="no-code"),
        pytest.param("llm_rate_limited", "transient", id="transient"),
        pytest.param("llm_schema", "deterministic", id="deterministic"),
        pytest.param("llm_refused", "terminal", id="terminal"),
        pytest.param("legacy_code", "unknown", id="unrecognised"),
    ],
)
def test_error_class_of_maps_codes_to_retry_classes(error_code: str | None, expected: str) -> None:
    assert error_class_of(error_code) == expected


def test_terminal_error_codes_are_exactly_the_terminal_retry_class() -> None:
    expected = {code.value for code in ErrorCode if retry_class(code) is RetryClass.TERMINAL}

    assert set(TERMINAL_ERROR_CODES) == expected
    assert "llm_refused" in TERMINAL_ERROR_CODES


def test_group_counts_sums_codes_sharing_an_error_class() -> None:
    rows = [
        {"status": "failed", "error_code": "llm_rate_limited", "n": 2},
        {"status": "failed", "error_code": "worker_lost", "n": 3},
        {"status": "pending", "error_code": None, "n": 4},
    ]

    assert group_counts(rows) == (
        GroupCount("failed", "transient", 5),
        GroupCount("pending", "none", 4),
    )


@pytest.mark.parametrize(
    ("group", "expected"),
    [
        pytest.param(GroupCount("failed", "terminal", 1), True, id="failed-terminal"),
        pytest.param(GroupCount("failed", "transient", 1), False, id="failed-transient"),
        pytest.param(GroupCount("pending", "terminal", 1), False, id="pending-terminal"),
    ],
)
def test_is_terminal_only_for_failed_terminal_rows(group: GroupCount, expected: bool) -> None:
    assert group.is_terminal is expected


def _lines(stream: io.StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in stream.getvalue().splitlines()]


def test_dry_run_log_reports_counts_per_group_and_summary() -> None:
    stream = io.StringIO()
    groups = (GroupCount("failed", "terminal", 2), GroupCount("pending", "none", 5))

    log_dry_run(build_logger(stream), groups)

    lines = _lines(stream)
    assert [(line["event"], line.get("count")) for line in lines] == [
        ("parked_jobs", 2),
        ("parked_jobs", 5),
        ("dry_run_summary", None),
    ]
    assert (lines[-1]["parked"], lines[-1]["releasable"], lines[-1]["terminal"]) == (7, 5, 2)


def test_release_log_reports_released_total_and_held_terminal() -> None:
    stream = io.StringIO()
    result = ReleaseResult(
        released=(GroupCount("failed", "transient", 3), GroupCount("pending", "none", 1)),
        held_terminal=2,
    )

    log_release(build_logger(stream), result)

    summary = _lines(stream)[-1]
    assert (summary["event"], summary["released"], summary["held_terminal"]) == (
        "release_summary",
        4,
        2,
    )


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param([], id="no-mode"),
        pytest.param(["--dry-run", "--release"], id="both-modes"),
    ],
)
def test_parser_requires_exactly_one_mode(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        _build_parser().parse_args(argv)
