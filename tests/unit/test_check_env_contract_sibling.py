"""Tests for the sibling gubbi-cloud config resolver in tools/check_env_contract.py.

The env-contract lint reads gubbi-cloud's ``config.py`` to check Environment
Literal parity. A cwd-relative ``../gubbi-cloud/gubbi_cloud/config.py`` is
correct from the main checkout and wrong from a git worktree nested inside it,
which has no sibling beside it. Both shapes must resolve the SAME absolute
path, and an unresolvable sibling must
fail loudly instead of skipping the parity check.

The CLI cases run a staged copy of the shipped script against real
``git worktree`` state, so what is under test is the shipped tool.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.unit

TOOL_PATH = Path(__file__).resolve().parents[2] / "tools" / "check_env_contract.py"


def _import_tool(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


lint = _import_tool(TOOL_PATH, "check_env_contract_sibling")

CANONICAL_LITERAL = 'Environment = Literal["dev", "ci", "staging", "production"]'
DIVERGENT_LITERAL = 'Environment = Literal["dev", "prod"]'
_WORKSPACE_DIR_NAME = "workspace"

# `git commit` refuses to run without an identity, and the ambient one must not
# leak into these fixtures.
GIT_ENV = {
    "GIT_AUTHOR_NAME": "gubbi-tests",
    "GIT_AUTHOR_EMAIL": "gubbi-tests@invalid",
    "GIT_COMMITTER_NAME": "gubbi-tests",
    "GIT_COMMITTER_EMAIL": "gubbi-tests@invalid",
}

_SETTINGS_SOURCE = (
    "from pydantic_settings import BaseSettings, SettingsConfigDict\n"
    "\n"
    "\n"
    "class Settings(BaseSettings):\n"
    '    model_config = SettingsConfigDict(env_prefix="JOURNAL_")\n'
    "    database_url: str\n"
)

_COMPOSE_SOURCE = "services:\n  gubbi:\n    environment:\n      - JOURNAL_DATABASE_URL\n"


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(  # noqa: S603 -- fixed argv, no shell
        ["git", *args],  # noqa: S607 -- git resolves via PATH by design
        cwd=cwd,
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(cwd), **GIT_ENV},
        capture_output=True,
        text=True,
        check=True,
    )


def _write_cloud_config(workspace: Path, literal: str) -> Path:
    """Create a sibling gubbi-cloud checkout carrying *literal* in its config.py."""
    config = workspace / "gubbi-cloud" / "gubbi_cloud" / "config.py"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(f"from typing import Literal\n\n{literal}\n", encoding="utf-8")
    return config


def _make_gubbi_checkout(workspace: Path) -> Path:
    """Create a git checkout holding a copy of the shipped lint tool."""
    checkout = workspace / "gubbi"
    staged = checkout / "tools" / TOOL_PATH.name
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_text(TOOL_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    (checkout / "gubbi").mkdir(parents=True, exist_ok=True)
    (checkout / "gubbi" / "config.py").write_text(
        f"from typing import Literal\n\n{CANONICAL_LITERAL}\n\n{_SETTINGS_SOURCE}",
        encoding="utf-8",
    )
    (checkout / "docker-compose.yml").write_text(_COMPOSE_SOURCE, encoding="utf-8")

    _git(checkout, "init", "-q", "-b", "develop")
    _git(checkout, "add", "-A")
    _git(checkout, "commit", "-qm", "initial")
    return checkout


def _add_worktree(checkout: Path, destination: Path, name: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    _git(checkout, "worktree", "add", "-q", "-b", f"wt-{name}", str(destination), "HEAD")
    return destination


def _nested_worktree(checkout: Path) -> Path:
    return _add_worktree(checkout, checkout / ".worktrees" / "nested", "nested")


def _add_detached_worktree(checkout: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    _git(checkout, "worktree", "add", "-q", "--detach", str(destination), "HEAD")
    return destination


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A workspace holding a gubbi git checkout and a sibling gubbi-cloud.

    Nested one level below ``tmp_path`` so a worktree placed beside the
    workspace has no sibling gubbi-cloud among its own ancestors.
    """
    root = tmp_path / _WORKSPACE_DIR_NAME
    root.mkdir()
    _write_cloud_config(root, CANONICAL_LITERAL)
    _make_gubbi_checkout(root)
    return root


def _run_lint(cwd: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 -- staged copy of this repo's own tool
        [sys.executable, "tools/check_env_contract.py", *extra],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


# -- resolver ---------------------------------------------------------------


def test_nested_worktree_resolves_the_same_config_as_the_main_checkout(workspace: Path) -> None:
    """The defect: ``../gubbi-cloud/...`` from a worktree points nowhere."""
    checkout = workspace / "gubbi"
    worktree = _nested_worktree(checkout)

    from_main = lint.resolve_cloud_config(checkout / "tools")
    from_worktree = lint.resolve_cloud_config(worktree / "tools")

    assert from_main == workspace / "gubbi-cloud" / "gubbi_cloud" / "config.py"
    assert from_worktree == from_main


def test_worktree_outside_the_workspace_resolves_through_the_git_common_dir(
    workspace: Path, tmp_path: Path
) -> None:
    """Ancestor walking cannot reach the sibling from here; only the common dir can."""
    worktree = _add_worktree(workspace / "gubbi", tmp_path / "external" / "wt", "external")
    for ancestor in [worktree, *worktree.parents]:
        assert not (ancestor / "gubbi-cloud" / "gubbi_cloud" / "config.py").exists()

    resolved = lint.resolve_cloud_config(worktree / "tools")

    assert resolved == workspace / "gubbi-cloud" / "gubbi_cloud" / "config.py"


def test_detached_worktree_outside_the_workspace_resolves_through_the_git_common_dir(
    workspace: Path, tmp_path: Path
) -> None:
    """A detached worktree has no branch but shares the same common dir."""
    worktree = _add_detached_worktree(workspace / "gubbi", tmp_path / "external" / "detached")
    for ancestor in [worktree, *worktree.parents]:
        assert not (ancestor / "gubbi-cloud" / "gubbi_cloud" / "config.py").exists()

    resolved = lint.resolve_cloud_config(worktree / "tools")

    assert resolved == workspace / "gubbi-cloud" / "gubbi_cloud" / "config.py"


def test_cli_passes_from_an_external_detached_worktree(workspace: Path, tmp_path: Path) -> None:
    worktree = _add_detached_worktree(workspace / "gubbi", tmp_path / "external" / "detached")

    result = _run_lint(worktree)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Env-contract check passed." in result.stdout


def test_default_start_anchors_on_the_tool_file_not_the_cwd(
    workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Production passes no ``start``; the default must be the tool's own dir."""
    staged = _import_tool(
        workspace / "gubbi" / "tools" / TOOL_PATH.name, f"staged_env_contract_{id(workspace)}"
    )
    elsewhere = tmp_path / "unrelated"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert lint.resolve_cloud_config(elsewhere) is None

    resolved = staged.resolve_cloud_config()

    assert resolved == workspace / "gubbi-cloud" / "gubbi_cloud" / "config.py"


def test_returns_none_when_no_sibling_checkout_exists(tmp_path: Path) -> None:
    checkout = _make_gubbi_checkout(tmp_path / _WORKSPACE_DIR_NAME)

    assert lint.resolve_cloud_config(checkout / "tools") is None


# -- CLI behaviour ----------------------------------------------------------


def test_cli_passes_from_a_worktree_when_the_sibling_literal_matches(workspace: Path) -> None:
    result = _run_lint(_nested_worktree(workspace / "gubbi"))

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Env-contract check passed." in result.stdout


def test_cli_reports_parity_drift_from_a_worktree(workspace: Path) -> None:
    """Positive control: a `differs` verdict proves the sibling was read."""
    _write_cloud_config(workspace, DIVERGENT_LITERAL)

    result = _run_lint(_nested_worktree(workspace / "gubbi"))

    assert result.returncode == 1
    assert "differs" in result.stderr


def test_cli_fails_with_one_prerequisite_when_the_sibling_is_absent(tmp_path: Path) -> None:
    """A missing sibling is a prerequisite failure, never a silent skip."""
    checkout = _make_gubbi_checkout(tmp_path / _WORKSPACE_DIR_NAME)

    result = _run_lint(_nested_worktree(checkout))

    assert result.returncode == 1
    lines = [ln for ln in result.stderr.splitlines() if "sibling gubbi-cloud checkout" in ln]
    assert len(lines) == 1, result.stderr
    assert "--cloud-config" in result.stderr


def test_cli_honours_an_explicit_cloud_config_path(workspace: Path) -> None:
    """The explicit file diverges while the sibling matches, so only the flag can fail it."""
    elsewhere = workspace / "elsewhere" / "config.py"
    elsewhere.parent.mkdir(parents=True)
    elsewhere.write_text(f"from typing import Literal\n\n{DIVERGENT_LITERAL}\n", encoding="utf-8")

    result = _run_lint(workspace / "gubbi", "--cloud-config", str(elsewhere))

    assert result.returncode == 1
    assert str(elsewhere) in result.stderr
    assert "differs" in result.stderr


def test_cli_skips_parity_on_an_empty_cloud_config_flag(workspace: Path) -> None:
    """The sibling diverges, so exit 0 proves the check was skipped, not satisfied."""
    _write_cloud_config(workspace, DIVERGENT_LITERAL)

    result = _run_lint(workspace / "gubbi", "--cloud-config", "")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "differs" not in result.stderr
