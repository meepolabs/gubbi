"""Wiring tests for ``tools/check_required_needs.py`` against this repo's workflow.

The checker's full behavior suite lives with its canonical copy in gubbi-common
(``tests/tools/test_check_required_needs.py``). This file pins this repo's copy
to those bytes, welds the checker's line parse of the real workflow to a PyYAML
parse, and pins the step that runs it inside ``required``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import shlex
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import yaml

if TYPE_CHECKING:
    from types import ModuleType

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW = ".github/workflows/ci.yml"

_TOOL = _ROOT / "tools" / "check_required_needs.py"
_TOOL_SHA256 = "0a783a673dc82818ff16e0eb2098536f5532c536a71a7f19d97f6e6d899c2159"
_NEEDS_STEP = "needs lists every other job"
_VERDICT_STEP = "every required job succeeded"
_COMMAND = ["python3", "tools/check_required_needs.py", "--self-test", _WORKFLOW]


def _load_checker() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_required_needs", _TOOL)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checker = _load_checker()


def _names(entries: list[Any]) -> list[str]:
    return [str(entry.name) for entry in entries]


# -- this copy matches the canonical bytes --------------------------------------


def test_the_checker_is_byte_identical_to_the_canonical_copy() -> None:
    """The behavior suite runs only against the canonical bytes, so a drifted copy is untested."""
    digest = hashlib.sha256(_TOOL.read_bytes()).hexdigest()

    assert digest == _TOOL_SHA256, (
        "tools/check_required_needs.py differs from the canonical copy. Edit "
        "gubbi-common tools/check_required_needs.py first (its full suite is "
        "tests/tools/test_check_required_needs.py there), copy the new bytes into "
        "every repo that carries the checker, then update _TOOL_SHA256 in each copy "
        "of this test."
    )


# -- what the parse reads -------------------------------------------------------


def test_the_line_parse_agrees_with_pyyaml_on_the_real_workflow() -> None:
    source = (_ROOT / _WORKFLOW).read_text(encoding="utf-8")
    parsed = yaml.safe_load(source)["jobs"]

    workflow = checker.parse_workflow(source)

    assert len(parsed) > 1
    assert parsed["required"]["needs"]
    assert _names(workflow.jobs) == list(parsed)
    assert _names(workflow.needs) == parsed["required"]["needs"]


def test_the_real_workflow_is_complete() -> None:
    assert checker.find_problems((_ROOT / _WORKFLOW).read_text(encoding="utf-8")) == []


def test_the_script_runs_standalone_on_the_real_workflow() -> None:
    result = subprocess.run(  # noqa: S603 -- this repo's own tool
        [sys.executable, "-I", str(_TOOL), "--self-test", str(_ROOT / _WORKFLOW)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


# -- the step that runs it inside `required` -----------------------------------


def _required_job() -> dict[str, Any]:
    job: dict[str, Any] = yaml.safe_load((_ROOT / _WORKFLOW).read_text(encoding="utf-8"))["jobs"][
        "required"
    ]
    return job


def _step_index(steps: list[dict[str, Any]], name: str) -> int:
    matches = [i for i, step in enumerate(steps) if step.get("name") == name]
    assert len(matches) == 1, f"expected one {name!r} step in `required`; found {len(matches)}"
    return matches[0]


def test_required_runs_the_check_with_exactly_this_command() -> None:
    steps = _required_job()["steps"]
    step = steps[_step_index(steps, _NEEDS_STEP)]

    assert shlex.split(step["run"]) == _COMMAND


def test_the_check_cannot_be_skipped_or_have_its_failure_ignored() -> None:
    steps = _required_job()["steps"]
    step = steps[_step_index(steps, _NEEDS_STEP)]

    assert "if" not in step
    assert "continue-on-error" not in step


def test_the_check_runs_after_checkout_and_before_the_verdict() -> None:
    steps = _required_job()["steps"]
    checkouts = [
        i
        for i, step in enumerate(steps)
        if str(step.get("uses", "")).startswith("actions/checkout@")
    ]

    assert len(checkouts) == 1
    assert checkouts[0] < _step_index(steps, _NEEDS_STEP) < _step_index(steps, _VERDICT_STEP)


def test_required_checks_out_without_credentials_and_has_a_timeout() -> None:
    job = _required_job()
    checkout = next(
        step for step in job["steps"] if str(step.get("uses", "")).startswith("actions/checkout@")
    )

    assert checkout.get("with", {}).get("persist-credentials") is False
    assert job.get("timeout-minutes") == 5
