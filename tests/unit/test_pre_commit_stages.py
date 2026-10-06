"""Unit tests for the hook stages in ``.pre-commit-config.yaml``.

The pre-push hook runs ``make prepush`` and nothing else. A hook without an
explicit ``stages:`` takes its stages from the upstream manifest at the
pinned rev, which this config does not control, so such a hook counts as
able to run at pre-push. Offline: reads the committed config only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

CONFIG = Path(__file__).resolve().parents[2] / ".pre-commit-config.yaml"
PUSH_HOOK = "prepush"
PUSH_STAGE = "pre-push"

pytestmark = pytest.mark.unit


def _load_config() -> dict[str, Any]:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert isinstance(config, dict)
    return config


def _hooks() -> list[dict[str, Any]]:
    return [hook for repo in _load_config()["repos"] for hook in repo["hooks"]]


def _may_run_at_push(hook: dict[str, Any]) -> bool:
    stages = hook.get("stages")
    return stages is None or PUSH_STAGE in stages


def test_install_wires_the_pre_push_hook() -> None:
    assert PUSH_STAGE in _load_config()["default_install_hook_types"]


def test_prepush_is_the_only_hook_that_can_run_at_push() -> None:
    hooks = _hooks()

    push_hooks = [hook["id"] for hook in hooks if _may_run_at_push(hook)]

    assert len(hooks) > 1, "fixture must contain hooks besides prepush"
    assert push_hooks == [PUSH_HOOK], (
        f"hooks able to run at {PUSH_STAGE}: {push_hooks}; set "
        "`stages: [pre-commit]` on every hook other than prepush"
    )


def test_prepush_hook_runs_make_prepush_on_every_push() -> None:
    (hook,) = [hook for hook in _hooks() if hook["id"] == PUSH_HOOK]

    assert hook["entry"] == "make prepush"
    assert hook["stages"] == [PUSH_STAGE]
    assert hook["language"] == "system"
    assert hook["always_run"] is True
    assert hook["pass_filenames"] is False
