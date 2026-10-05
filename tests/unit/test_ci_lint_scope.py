"""Pin which paths the ``lint`` job in ``.github/workflows/ci.yml`` checks.

``tools/`` holds the scripts CI and the Makefile run (the env-contract and
required-needs checks, the DB-suite runner, the test-database wrapper), so it
is linted, format-checked and type-checked alongside ``gubbi/``. A path dropped
from one of those commands stops being checked without any job failing, which
is what these tests catch. Offline: they read the committed workflow.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

CI_WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "ci.yml"
LINT_JOB = "lint"
TOOLS_PATH = "tools/"


def _lint_step_command(step_name: str) -> list[str]:
    workflow = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))
    steps: list[dict[str, Any]] = workflow["jobs"][LINT_JOB]["steps"]
    matches = [step for step in steps if step.get("name") == step_name]
    assert len(matches) == 1, f"expected exactly one {step_name!r} step in job {LINT_JOB!r}"
    return shlex.split(matches[0]["run"])


@pytest.mark.parametrize(
    ("step_name", "tool_args"),
    [
        pytest.param("ruff check", ["ruff", "check"], id="ruff-check"),
        pytest.param("ruff format --check", ["ruff", "format", "--check"], id="ruff-format"),
        pytest.param("mypy", ["mypy"], id="mypy"),
    ],
)
def test_lint_step_checks_tools(step_name: str, tool_args: list[str]) -> None:
    command = _lint_step_command(step_name)

    assert command[: 2 + len(tool_args)] == ["poetry", "run", *tool_args]
    paths = command[2 + len(tool_args) :]
    assert TOOLS_PATH in paths, f"{step_name!r} does not check {TOOLS_PATH}: {command}"
