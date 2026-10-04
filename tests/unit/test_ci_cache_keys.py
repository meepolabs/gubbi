"""Contract test: every venv cache key stays visible to Renovate and is complete.

Renovate bumps the Poetry pin in cache keys through a regex that only matches the
literal ``-poetry<version>-`` segment; an expression in its place silently drops
the key from the Poetry group. A key without the resolved Python version would
restore a venv built for a different interpreter.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOWS = _ROOT / ".github" / "workflows"
_PYTHON_VERSION_SEGMENT = "-py${{ steps.setup-python.outputs.python-version }}-"
_CACHE_ACTION = "actions/cache@"


def _renovate_cache_key_regex() -> re.Pattern[str]:
    """The cache-key matchString from renovate.json, translated to Python syntax."""
    config = json.loads((_ROOT / "renovate.json").read_text(encoding="utf-8"))
    strings = [
        s
        for manager in config.get("customManagers", [])
        for s in manager.get("matchStrings", [])
        if s.startswith("key:")
    ]
    assert len(strings) == 1, f"expected one cache-key matchString, found {strings}"
    # Renovate uses JavaScript named groups, `(?<name>...)`.
    return re.compile(strings[0].replace("(?<", "(?P<"))


def _venv_cache_steps() -> list[Any]:
    found = []
    for path in sorted(_WORKFLOWS.glob("*.yml")):
        workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
        for job_id, job in (workflow.get("jobs") or {}).items():
            for step in job.get("steps", []):
                if str(step.get("uses", "")).startswith(_CACHE_ACTION) and str(
                    step.get("with", {}).get("key", "")
                ).startswith("venv-"):
                    where = f"{path.name}:{job_id}"
                    found.append(pytest.param(where, step, id=where))
    return found


def _poetry_version_in_installs() -> set[str]:
    versions = set()
    for path in _WORKFLOWS.glob("*.yml"):
        workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
        for job in (workflow.get("jobs") or {}).values():
            for step in job.get("steps", []):
                if str(step.get("uses", "")).startswith("snok/install-poetry@"):
                    versions.add(str(step["with"]["version"]))
    return versions


def test_every_lockfile_venv_cache_is_found() -> None:
    """Positive control: an empty scan would pass every assertion below."""
    assert len(_venv_cache_steps()) == 3


@pytest.mark.parametrize(("where", "step"), _venv_cache_steps())
def test_cache_key_carries_the_resolved_python_version(where: str, step: dict[str, Any]) -> None:
    assert _PYTHON_VERSION_SEGMENT in step["with"]["key"], where


@pytest.mark.parametrize(("where", "step"), _venv_cache_steps())
def test_cache_key_matches_renovates_poetry_regex(where: str, step: dict[str, Any]) -> None:
    match = _renovate_cache_key_regex().search(f"key: {step['with']['key']}\n")

    assert match is not None, f"{where}: Renovate cannot see the Poetry pin in this key"
    assert {match.group("currentValue")} == _poetry_version_in_installs(), where
