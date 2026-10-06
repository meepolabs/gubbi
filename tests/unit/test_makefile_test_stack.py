"""The Makefile's test-stack targets match the runner and the CI job.

``make -n`` prints each recipe without running it, so no docker, stack or
suite is touched. The stack profile must be the one tools/run_db_suites.py
resets, or ``prepush`` would check one stack and reset another; the suite env
must be the one security-tests.yml gives the same suites.
"""

from __future__ import annotations

import importlib.util
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import yaml

if TYPE_CHECKING:
    from types import ModuleType

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_RUNNER = _ROOT / "tools" / "run_db_suites.py"
_WORKFLOW = _ROOT / ".github" / "workflows" / "security-tests.yml"
_JOB = "security-tests"
_TESTDB = ["python3", "tools/testdb/testdb.py"]


def _load_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_db_suites_for_makefile", _RUNNER)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _load_runner()


def _dry_run(target: str, *assignments: str) -> list[list[str]]:
    make = shutil.which("make")
    assert make is not None, "make is required to read the Makefile"
    env = {k: v for k, v in os.environ.items() if not k.startswith("MAKE")}
    completed = subprocess.run(  # noqa: S603
        [make, "-s", "-n", "-C", str(_ROOT), target, *assignments],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return [shlex.split(line) for line in completed.stdout.splitlines() if line.strip()]


@pytest.mark.parametrize(
    ("target", "assignments", "expected"),
    [
        pytest.param("test-stack-up", (), [[*_TESTDB, "up", *RUNNER.PROFILE]], id="up"),
        pytest.param("test-stack-down", (), [[*_TESTDB, "down", *RUNNER.PROFILE]], id="down"),
        pytest.param("test-stack-status", (), [[*_TESTDB, "status", *RUNNER.PROFILE]], id="status"),
        pytest.param(
            "test-stack-status",
            ("REQUIRE_READY=1",),
            [[*_TESTDB, "status", *RUNNER.PROFILE, "--require-ready"]],
            id="status-require-ready",
        ),
    ],
)
def test_stack_targets_use_the_runner_profile(
    target: str, assignments: tuple[str, ...], expected: list[list[str]]
) -> None:
    assert _dry_run(target, *assignments) == expected


def test_prepush_checks_the_stack_before_running_the_suites() -> None:
    check, run = _dry_run("prepush")

    assert check == [*_TESTDB, "status", *RUNNER.PROFILE, "--require-ready"]
    assert run[0] == "env"
    assert run[-2:] == ["python3", "tools/run_db_suites.py"]


def test_prepush_gives_the_suites_the_ci_job_env() -> None:
    (_, run) = _dry_run("prepush")
    workflow = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    job_env = workflow["jobs"][_JOB]["env"]

    assignments = dict(word.split("=", 1) for word in run[1:-2])

    assert assignments == {name: job_env[name] for name in RUNNER.REQUIRED_ENV}


def test_a_command_line_profile_cannot_change_the_stack() -> None:
    (check, _) = _dry_run("prepush", "PROFILE=--repo other", "TESTDB=elsewhere.py")

    assert check == [*_TESTDB, "status", *RUNNER.PROFILE, "--require-ready"]
