"""Unit tests for the pure parts of ``gubbi.scripts.release_parked_extraction_jobs``."""

from __future__ import annotations

import io
import json

import pytest

from gubbi.scripts.release_parked_extraction_jobs import (
    DryRunResult,
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
    ("status", "error_code", "expected"),
    [
        pytest.param("pending", None, "none", id="pending-no-code"),
        pytest.param("failed", "llm_rate_limited", "transient", id="transient"),
        pytest.param("failed", "llm_schema", "deterministic", id="deterministic"),
        pytest.param("failed", "llm_refused", "terminal", id="terminal"),
        pytest.param("failed", None, "unclassified", id="failed-no-code"),
        pytest.param("failed", "legacy_code", "unclassified", id="unrecognised"),
    ],
)
def test_error_class_of_maps_codes_to_retry_classes(
    status: str, error_code: str | None, expected: str
) -> None:
    assert error_class_of(status, error_code) == expected


def test_group_counts_sums_codes_sharing_an_error_class_and_verdict() -> None:
    rows = [
        {"status": "failed", "error_code": "llm_rate_limited", "terminal": False, "n": 2},
        {"status": "failed", "error_code": "worker_lost", "terminal": False, "n": 3},
        {"status": "failed", "error_code": "llm_schema", "terminal": False, "n": 6},
        {"status": "failed", "error_code": "internal_error", "terminal": True, "n": 7},
        {"status": "pending", "error_code": None, "terminal": False, "n": 4},
    ]

    assert group_counts(rows) == (
        GroupCount("failed", "deterministic", 6),
        GroupCount("failed", "deterministic", 7, terminal=True),
        GroupCount("failed", "transient", 5),
        GroupCount("pending", "none", 4),
    )


def test_group_counts_rejects_rows_of_unexpected_types() -> None:
    row = {"status": "failed", "error_code": "llm_schema", "terminal": "yes", "n": 1}

    with pytest.raises(TypeError):
        group_counts([row])


def _lines(stream: io.StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in stream.getvalue().splitlines()]


def test_dry_run_log_reports_counts_per_group_and_summary() -> None:
    stream = io.StringIO()
    groups = (
        GroupCount("failed", "terminal", 2, terminal=True),
        GroupCount("pending", "none", 5),
    )

    log_dry_run(build_logger(stream), DryRunResult(groups, would_hydrate=3, held_unhydrated=1))

    lines = _lines(stream)
    assert [(line["event"], line.get("count")) for line in lines] == [
        ("parked_jobs", 2),
        ("parked_jobs", 5),
        ("dry_run_summary", None),
    ]
    summary = lines[-1]
    assert (
        summary["parked"],
        summary["releasable"],
        summary["terminal"],
        summary["would_hydrate"],
        summary["held_unhydrated"],
    ) == (7, 4, 2, 3, 1)


def test_release_log_reports_released_total_and_held_terminal() -> None:
    stream = io.StringIO()
    result = ReleaseResult(
        released=(GroupCount("failed", "transient", 3), GroupCount("pending", "none", 1)),
        held_terminal=2,
        hydrated=3,
        held_unhydrated=1,
    )

    log_release(build_logger(stream), result)

    summary = _lines(stream)[-1]
    assert (
        summary["event"],
        summary["released"],
        summary["held_terminal"],
        summary["hydrated"],
        summary["held_unhydrated"],
    ) == ("release_summary", 4, 2, 3, 1)


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
