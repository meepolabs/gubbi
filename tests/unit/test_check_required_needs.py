"""Tests for ``tools/check_required_needs.py`` and the step that runs it in ``required``.

The checker is the branch-protection guard, so every layout it does not accept
must fail closed, and its line parse of the real workflow is welded to a
PyYAML parse here: a disagreement would mean it checks a different job set
from the one GitHub runs.
"""

from __future__ import annotations

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

# Every accepted variation in one valid workflow: trailing comments, blank and
# comment lines, another job with its own block-list `needs`, and block-scalar
# bodies holding text (a tab-indented line, `needs:`, `- ghost`, `ghost:`) that
# would be a layout error or a phantom entry if it were read as structure.
_COMPLETE = """\
name: ci  # fixture
on: push

jobs:  # every job
  lint:  # first lane
    runs-on: ubuntu-latest
    steps:
      - run: |
          needs:
            - ghost
          \tprintf 'tab-indented shell'
          ghost:
      - run: >-
          folded

  # a comment between jobs
  tests:
    needs:
      - lint
    uses: ./.github/workflows/tests.yml

  required:  # the aggregator
    if: ${{ always() }}
    needs:  # every other job
      - lint  # first lane

      # a comment inside needs
      - tests
    steps:
      - run: "true"
"""

_REQUIRED_TESTS = "      - tests\n"


def _replace(old: str, new: str, source: str = _COMPLETE) -> str:
    assert source.count(old) == 1, f"{old!r} is not unique in the fixture"
    return source.replace(old, new)


def _line_of(source: str, text: str) -> int:
    matches = [no for no, line in enumerate(source.splitlines(), start=1) if line == text]
    assert len(matches) == 1, f"{text!r} is not one line of the fixture"
    return matches[0]


def _names(entries: list[Any]) -> list[str]:
    return [str(entry.name) for entry in entries]


# -- what the parse reads -------------------------------------------------------


def test_a_complete_workflow_has_no_problems() -> None:
    assert checker.find_problems(_COMPLETE) == []


def test_block_scalar_bodies_and_comments_are_never_read_as_structure() -> None:
    workflow = checker.parse_workflow(_COMPLETE)

    assert _names(workflow.jobs) == ["lint", "tests", "required"]
    assert _names(workflow.needs) == ["lint", "tests"]


def test_the_fixture_parses_the_same_with_pyyaml() -> None:
    """The fixture is a valid workflow, so its accepted shapes are ones GitHub accepts."""
    parsed = yaml.safe_load(_COMPLETE)["jobs"]
    workflow = checker.parse_workflow(_COMPLETE)

    assert _names(workflow.jobs) == list(parsed)
    assert _names(workflow.needs) == parsed["required"]["needs"]


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


# -- findings: exit 1 -----------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "line", "message"),
    [
        pytest.param(
            _replace(_REQUIRED_TESTS, ""),
            "  tests:",
            "job `tests` is missing from `required.needs`",
            id="missing",
        ),
        pytest.param(
            _replace(_REQUIRED_TESTS, _REQUIRED_TESTS + "      - deploy\n"),
            "      - deploy",
            "`deploy` in `needs` is not a job in this workflow",
            id="unknown",
        ),
        pytest.param(
            _replace(_REQUIRED_TESTS, _REQUIRED_TESTS + "      - lint  # again\n"),
            "      - lint  # again",
            "`lint` is listed in `needs` more than once",
            id="duplicate",
        ),
        pytest.param(
            _replace(_REQUIRED_TESTS, _REQUIRED_TESTS + "      - required\n"),
            "      - required",
            "`required` lists itself in `needs`",
            id="self-listed",
        ),
        pytest.param(
            "jobs:\n  required:\n    needs:\n    steps:\n      - run: x\n",
            "  required:",
            "`required` has no other job to aggregate",
            id="nothing-to-aggregate",
        ),
    ],
)
def test_each_finding_is_reported_on_its_line(source: str, line: str, message: str) -> None:
    assert checker.find_problems(source) == [f"line {_line_of(source, line)}: {message}"]


# -- layouts that fail closed: exit 2 -------------------------------------------


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            _replace(
                '    steps:\n      - run: "true"\n',
                '    needs:\n      - lint\n    steps:\n      - run: "true"\n',
            ),
            id="second-needs-key-after-a-complete-one",
        ),
        pytest.param(
            _replace("\n  required:", '\n  "ungated":\n    runs-on: ubuntu-latest\n\n  required:'),
            id="double-quoted-job-key-after-a-job",
        ),
        pytest.param(
            _replace("\n  required:", "\n  'ungated':\n    runs-on: ubuntu-latest\n\n  required:"),
            id="single-quoted-job-key-after-a-job",
        ),
        pytest.param(
            _replace(
                "    needs:  # every other job\n      - lint  # first lane\n",
                "    needs: [lint, tests]\n",
            ).replace(_REQUIRED_TESTS, ""),
            id="flow-list-needs",
        ),
        pytest.param(
            _replace(
                "    needs:  # every other job\n      - lint  # first lane\n",
                "    needs: lint\n",
            ),
            id="scalar-needs",
        ),
        pytest.param(_replace(_REQUIRED_TESTS, '      - "tests"\n'), id="double-quoted-entry"),
        pytest.param(_replace(_REQUIRED_TESTS, "      - 'tests'\n"), id="single-quoted-entry"),
        pytest.param(_replace(_REQUIRED_TESTS, "     - tests\n"), id="entry-at-five-spaces"),
        pytest.param(_replace(_REQUIRED_TESTS, "       - tests\n"), id="entry-at-seven-spaces"),
        pytest.param(_replace("  tests:\n", "   tests:\n"), id="job-at-three-spaces"),
        pytest.param(_replace("    if: ${{", "\tif: ${{"), id="tab-indentation"),
        pytest.param(_replace("    if: ${{", "    - if: ${{"), id="non-key-in-required"),
        pytest.param(_replace("jobs:  # every job\n", "workflows:\n"), id="no-jobs-key"),
        pytest.param(_replace("jobs:  # every job\n", '"jobs":\n'), id="quoted-jobs-key"),
        pytest.param(_replace("  required:  # the aggregator", "  verdict:"), id="no-required-job"),
        pytest.param(_replace("    needs:  # every other job", "    depends:"), id="no-needs-key"),
        pytest.param(
            _replace("    needs:  # every other job", '    "needs":'), id="quoted-needs-key"
        ),
        pytest.param(_COMPLETE + "  lint:\n    runs-on: ubuntu-latest\n", id="duplicate-job-id"),
        pytest.param(_COMPLETE + "jobs:\n  extra:\n    runs-on: x\n", id="second-jobs-key"),
        pytest.param(_COMPLETE + "- stray\n", id="non-key-after-jobs"),
        pytest.param(
            _replace(_REQUIRED_TESTS, "      - lint\u2028      - tests\n"), id="foreign-line-break"
        ),
    ],
)
def test_an_unaccepted_layout_fails_closed(source: str) -> None:
    with pytest.raises(checker.LayoutError):
        checker.find_problems(source)


# -- the self-test --------------------------------------------------------------


def test_the_self_test_passes_when_another_job_has_its_own_needs() -> None:
    """Only the ``required`` block is mutated, so `tests`' own `- lint` is never dropped."""
    assert checker.self_test_failures(_COMPLETE) == []


def test_the_self_test_fails_a_check_that_reports_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(checker, "find_problems", lambda _source: [])
    workflow = tmp_path / "ci.yml"
    workflow.write_text(_COMPLETE, encoding="utf-8")

    failures = checker.self_test_failures(_COMPLETE)

    assert failures == [
        f"line {_line_of(_COMPLETE, '      - lint  # first lane')}: "
        "dropping `lint` from `needs` went unreported",
        f"line {_line_of(_COMPLETE, '      - tests')}: dropping `tests` from `needs` went unreported",
    ]
    assert checker.main(["--self-test", str(workflow)]) == 1


# -- exit codes -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        pytest.param(_COMPLETE, 0, id="complete"),
        pytest.param(_replace(_REQUIRED_TESTS, ""), 1, id="finding"),
        pytest.param(
            _replace("\n  required:", '\n  "ungated":\n    runs-on: x\n\n  required:'),
            2,
            id="layout",
        ),
        pytest.param(b"jobs:\n  \xff:\n", 2, id="not-utf8"),
    ],
)
def test_main_exit_code(source: str | bytes, expected: int, tmp_path: Path) -> None:
    workflow = tmp_path / "ci.yml"
    if isinstance(source, bytes):
        workflow.write_bytes(source)
    else:
        workflow.write_text(source, encoding="utf-8")

    assert checker.main(["--self-test", str(workflow)]) == expected


def test_main_reports_an_unreadable_file_as_exit_2(tmp_path: Path) -> None:
    assert checker.main([str(tmp_path / "absent.yml")]) == 2


def test_main_without_a_workflow_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as exc:
        checker.main([])

    assert exc.value.code == 2


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        pytest.param(None, 0, id="real-workflow"),
        pytest.param(_replace(_REQUIRED_TESTS, ""), 1, id="finding"),
    ],
)
def test_the_script_runs_standalone(source: str | None, expected: int, tmp_path: Path) -> None:
    workflow = _ROOT / _WORKFLOW
    if source is not None:
        workflow = tmp_path / "ci.yml"
        workflow.write_text(source, encoding="utf-8")

    result = subprocess.run(  # noqa: S603 -- this repo's own tool
        [sys.executable, "-I", str(_TOOL), "--self-test", str(workflow)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == expected, result.stdout + result.stderr


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
