"""Tests for the extraction retry classification table and backoff."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from itertools import pairwise

import pytest

from gubbi.extraction import retry_policy
from gubbi.extraction.jobs import extract_conversation
from gubbi.extraction.retry_policy import (
    DETERMINISTIC_DELAY,
    DETERMINISTIC_MAX_ATTEMPTS,
    JITTER_RATIO,
    TRANSIENT_BASE_DELAY,
    TRANSIENT_MAX_DELAY,
    ErrorCode,
    RetryClass,
    is_terminal,
    next_run_after,
    retry_class,
)

NOW = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)

# orphan_cleanup writes these two codes from inline SQL, so no Python
# constant exists to import.
_ORPHAN_SWEEP_CODES = ("enqueue_lost", "worker_lost")

_PRODUCED_CODES = sorted(
    {
        *(code for _, code in extract_conversation._ERROR_CODES),
        extract_conversation._CANCELLED_ERROR_CODE,
        extract_conversation._classify_error(Exception("unclassified")),
        *_ORPHAN_SWEEP_CODES,
    }
)


def _codes_of(cls: RetryClass) -> list[ErrorCode]:
    return [code for code, c in retry_policy.RETRY_CLASSES.items() if c is cls]


_TRANSIENT_CODES = _codes_of(RetryClass.TRANSIENT)
_DETERMINISTIC_CODES = _codes_of(RetryClass.DETERMINISTIC)


def _no_jitter() -> float:
    return 0.0


def _max_jitter() -> float:
    return 0.999999


def _delay(code: ErrorCode, attempts: int, rng: Callable[[], float]) -> timedelta:
    run_after = next_run_after(code, attempts, NOW, rng=rng)
    assert run_after is not None
    return run_after - NOW


@pytest.mark.parametrize("code", list(ErrorCode), ids=str)
def test_every_error_code_has_a_retry_class(code: ErrorCode) -> None:
    assert code in retry_policy.RETRY_CLASSES
    assert isinstance(retry_class(code), RetryClass)


@pytest.mark.parametrize("code", _PRODUCED_CODES)
def test_every_produced_error_code_is_an_error_code_member(code: str) -> None:
    assert code in {member.value for member in ErrorCode}


def test_classes_match_the_settled_partition() -> None:
    assert {c.value for c in _TRANSIENT_CODES} == {
        "llm_rate_limited",
        "llm_provider_error",
        "worker_lost",
        "enqueue_lost",
    }
    assert {c.value for c in _DETERMINISTIC_CODES} == {
        "llm_schema",
        "llm_truncated",
        "deadline_exceeded",
        "internal_error",
    }


@pytest.mark.parametrize("attempts", [1, 2, 5, 100])
def test_llm_refused_never_runs_again(attempts: int) -> None:
    assert next_run_after(ErrorCode.LLM_REFUSED, attempts, NOW) is None


@pytest.mark.parametrize("code", _TRANSIENT_CODES, ids=str)
def test_transient_backoff_doubles_from_base_until_cap(code: ErrorCode) -> None:
    delays = [_delay(code, n, _no_jitter) for n in range(1, 40)]

    assert delays[0] == TRANSIENT_BASE_DELAY
    assert delays[1] == TRANSIENT_BASE_DELAY * 2
    assert all(a <= b for a, b in pairwise(delays))
    assert max(delays) == TRANSIENT_MAX_DELAY
    assert delays[-1] == TRANSIENT_MAX_DELAY


@pytest.mark.parametrize("attempts", [1, 4, 39])
def test_jitter_shortens_delay_within_ratio(attempts: int) -> None:
    code = ErrorCode.LLM_RATE_LIMITED
    unjittered = _delay(code, attempts, _no_jitter)

    jittered = _delay(code, attempts, _max_jitter)

    assert unjittered * (1 - JITTER_RATIO) <= jittered < unjittered


@pytest.mark.parametrize("code", _DETERMINISTIC_CODES, ids=str)
def test_deterministic_waits_long_delay_before_max_attempts(code: ErrorCode) -> None:
    for attempts in range(1, DETERMINISTIC_MAX_ATTEMPTS):
        assert next_run_after(code, attempts, NOW, rng=_no_jitter) == NOW + DETERMINISTIC_DELAY


@pytest.mark.parametrize("code", _DETERMINISTIC_CODES, ids=str)
def test_deterministic_is_terminal_at_max_attempts(code: ErrorCode) -> None:
    for attempts in (DETERMINISTIC_MAX_ATTEMPTS, DETERMINISTIC_MAX_ATTEMPTS + 5):
        assert next_run_after(code, attempts, NOW) is None


def test_rejects_attempts_below_one() -> None:
    with pytest.raises(ValueError, match="attempts"):
        next_run_after(ErrorCode.WORKER_LOST, 0, NOW)


def test_rejects_naive_now() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        next_run_after(ErrorCode.WORKER_LOST, 1, datetime(2026, 10, 10, 12, 0))


@pytest.mark.parametrize(
    ("code", "attempts", "expected"),
    [
        pytest.param("llm_refused", 1, True, id="terminal-first-attempt"),
        pytest.param("llm_schema", DETERMINISTIC_MAX_ATTEMPTS - 1, False, id="deterministic-below"),
        pytest.param("llm_schema", DETERMINISTIC_MAX_ATTEMPTS, True, id="deterministic-at-max"),
        pytest.param(
            "internal_error", DETERMINISTIC_MAX_ATTEMPTS + 4, True, id="deterministic-past"
        ),
        pytest.param("worker_lost", 100, False, id="transient-many-attempts"),
        pytest.param(None, 100, False, id="no-code"),
        pytest.param("legacy_code", 100, False, id="unknown-code"),
    ],
)
def test_is_terminal_matches_terminal_class_or_exhausted_deterministic(
    code: str | None, attempts: int, expected: bool
) -> None:
    assert is_terminal(code, attempts) is expected


@pytest.mark.parametrize("code", list(ErrorCode), ids=str)
def test_is_terminal_agrees_with_next_run_after(code: ErrorCode) -> None:
    for attempts in range(1, DETERMINISTIC_MAX_ATTEMPTS + 3):
        assert is_terminal(code, attempts) is (next_run_after(code, attempts, NOW) is None)


def test_code_sets_partition_the_error_codes() -> None:
    sets = (
        retry_policy.TERMINAL_CODES,
        retry_policy.DETERMINISTIC_CODES,
        retry_policy.TRANSIENT_CODES,
    )

    assert sum(len(codes) for codes in sets) == len(ErrorCode)
    assert set().union(*sets) == {code.value for code in ErrorCode}
    assert {"llm_refused", "superseded", "conversation_unreadable"} == retry_policy.TERMINAL_CODES
