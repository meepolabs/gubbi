"""Tests for ``tools/image_closure_needed.py`` and the ``image-closure`` lane that runs it.

The decision must fail closed: every base it cannot diff against runs the
install. Diffs are read from a throwaway git repository, so the git calls are
the real ones the lane makes.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import yaml

if TYPE_CHECKING:
    from types import ModuleType

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_CI = _ROOT / ".github" / "workflows" / "ci.yml"
_TOOL = _ROOT / "tools" / "image_closure_needed.py"
_JOB = "image-closure"
_DECIDE_STEP = "decide whether the closure can be affected"
_INSTALL_STEP = "Install the Poetry closure in the image's base"
_DOCKERFILE = "deployment/Dockerfile"
_REQUIREMENTS = "deployment/poetry-requirements.txt"
_GIT = ("git", "-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "commit.gpgsign=false")


def _load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("image_closure_needed", _TOOL)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


closure = _load_tool()


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(  # noqa: S603
        [*_GIT, *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def _commit(repo: Path, path: str, content: str) -> str:
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    _git(repo, "add", "--", path)
    _git(repo, "commit", "-q", "-m", f"touch {path}")
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _commit(root, "README.md", "base\n")
    return root


@pytest.mark.parametrize(
    "base",
    [
        pytest.param("", id="empty"),
        pytest.param("0" * 40, id="all-zero-push-before"),
        pytest.param("d" * 40, id="object-not-in-clone"),
        pytest.param("not-a-revision", id="unresolvable"),
    ],
)
def test_an_unknown_base_runs_the_install(repo: Path, base: str) -> None:
    _commit(repo, "src/app.py", "x\n")

    verdict = closure.decide(closure.changed_paths(repo, base))

    assert verdict.run is True, verdict.reason


def test_a_git_failure_runs_the_install(tmp_path: Path) -> None:
    """Outside any repository every git call fails; that must not read as an empty diff."""
    verdict = closure.decide(closure.changed_paths(tmp_path, "a" * 40))

    assert verdict.run is True, verdict.reason


@pytest.mark.parametrize("path", [_DOCKERFILE, _REQUIREMENTS], ids=["dockerfile", "requirements"])
def test_a_diff_touching_an_image_input_runs_the_install(repo: Path, path: str) -> None:
    base = _git(repo, "rev-parse", "HEAD")
    _commit(repo, path, "changed\n")

    verdict = closure.decide(closure.changed_paths(repo, base))

    assert verdict.run is True
    assert path in verdict.reason


def test_an_unrelated_diff_skips_the_install(repo: Path) -> None:
    base = _git(repo, "rev-parse", "HEAD")
    _commit(repo, "gubbi/app.py", "x\n")
    _commit(repo, "deployment/entrypoint.sh", "x\n")

    changed = closure.changed_paths(repo, base)
    verdict = closure.decide(changed)

    assert changed == ["deployment/entrypoint.sh", "gubbi/app.py"]
    assert verdict.run is False


def test_the_watched_paths_name_the_image_inputs_and_the_check_itself() -> None:
    assert {
        _DOCKERFILE,
        _REQUIREMENTS,
        ".github/workflows/ci.yml",
        "tools/image_closure_needed.py",
    } == closure.WATCHED_PATHS


def test_every_watched_path_exists() -> None:
    """A renamed input would leave its old name watched and the new one unwatched."""
    missing = [path for path in closure.WATCHED_PATHS if not (closure.REPO_ROOT / path).is_file()]

    assert missing == []


@pytest.mark.parametrize(
    ("base", "expected"),
    [pytest.param("0" * 40, "run=true", id="unknown"), pytest.param(None, "run=false", id="skip")],
)
def test_main_writes_the_verdict_to_the_github_output(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, base: str | None, expected: str
) -> None:
    head = _git(repo, "rev-parse", "HEAD")
    output = tmp_path / "github_output"
    monkeypatch.setattr(closure, "REPO_ROOT", repo)

    code = closure.main(["--base", head if base is None else base, "--github-output", str(output)])

    assert code == 0
    assert output.read_text(encoding="utf-8") == f"{expected}\n"


def _job() -> dict[str, Any]:
    job: dict[str, Any] = yaml.safe_load(_CI.read_text(encoding="utf-8"))["jobs"][_JOB]
    return job


def _step(name: str) -> dict[str, Any]:
    matches = [step for step in _job()["steps"] if step.get("name") == name]
    assert len(matches) == 1, f"expected one {name!r} step in {_JOB}"
    step: dict[str, Any] = matches[0]
    return step


def test_the_lane_checks_out_full_history() -> None:
    """A shallow clone lacks the base object, which would force the install on every run."""
    checkout = _job()["steps"][0]

    assert str(checkout["uses"]).startswith("actions/checkout@")
    assert checkout["with"]["fetch-depth"] == 0


def test_the_lane_diffs_against_the_pr_base_or_the_push_before() -> None:
    step = _step(_DECIDE_STEP)

    assert step["env"]["BASE_SHA"] == (
        "${{ github.event_name == 'pull_request' && github.event.pull_request.base.sha"
        " || github.event.before }}"
    )
    assert step["run"] == (
        'python3 tools/image_closure_needed.py --base "$BASE_SHA" --github-output "$GITHUB_OUTPUT"'
    )


def test_only_an_explicit_skip_verdict_skips_the_install() -> None:
    """Gating on ``== 'true'`` would skip the install whenever the output went missing."""
    assert _step(_DECIDE_STEP)["id"] == "decide"
    assert _step(_INSTALL_STEP)["if"] == "${{ steps.decide.outputs.run != 'false' }}"


def test_the_lane_installs_from_the_watched_inputs() -> None:
    assert _job()["env"] == {"DOCKERFILE": _DOCKERFILE, "REQUIREMENTS": _REQUIREMENTS}
