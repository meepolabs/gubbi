"""Contract tests for the ``required`` aggregator job in ci.yml.

Branch protection names one check, ``required``. It is only meaningful if it
depends on every blocking lane, fails on any result other than success, and is
never itself skipped. The lanes it calls must not double-run (a called workflow
that keeps its own pull_request trigger runs twice per PR) and must not share a
concurrency group with their caller (inside a called workflow
``github.workflow`` is the caller's name, so a copied group cancels the caller).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

_WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"
_CI = _WORKFLOWS / "ci.yml"
_AGGREGATOR = "required"
_AGGREGATOR_STEP = "every required job succeeded"

# Workflows that must never gate `required`: advisory scans, which must not
# block unrelated work.
_NEVER_REQUIRED = ("dependency-scan.yml",)

# Called lanes that are call-only (no triggers of their own).
_CALL_ONLY = ("security-tests.yml",)

# The secret scan keeps its own push, schedule and manual triggers so a pushed
# commit is scanned even when the ci run is cancelled; pull requests reach it
# only through ci.yml.
_SECRET_SCAN = "gitleaks.yml"
_SECRET_SCAN_JOB = "secret-scan"
_SECRET_SCAN_TRIGGERS = {"workflow_call", "push", "schedule", "workflow_dispatch"}


def _load(path: Path) -> dict[str, Any]:
    parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict), f"{path} did not parse as a mapping"
    return parsed


def _triggers(workflow: dict[str, Any]) -> set[str]:
    # PyYAML (YAML 1.1) reads the bare key `on` as boolean True.
    on = workflow.get("on", workflow.get(True))
    if isinstance(on, str):
        return {on}
    if isinstance(on, list):
        return set(on)
    assert isinstance(on, dict), f"unrecognised trigger block: {on!r}"
    return set(on)


def _ci_jobs() -> dict[str, Any]:
    jobs = _load(_CI)["jobs"]
    assert isinstance(jobs, dict)
    return jobs


def _aggregator() -> dict[str, Any]:
    jobs = _ci_jobs()
    assert _AGGREGATOR in jobs, f"ci.yml has no {_AGGREGATOR!r} job: {sorted(jobs)}"
    job = jobs[_AGGREGATOR]
    assert isinstance(job, dict)
    return job


def _needs() -> list[str]:
    needs = _aggregator().get("needs", [])
    return [needs] if isinstance(needs, str) else list(needs)


def _called_workflows() -> dict[str, str]:
    """Map each ci.yml job that calls a local workflow to the called file name."""
    prefix = "./.github/workflows/"
    return {
        job_id: job["uses"].removeprefix(prefix)
        for job_id, job in _ci_jobs().items()
        if isinstance(job, dict) and str(job.get("uses", "")).startswith(prefix)
    }


# -- shape of the aggregator --------------------------------------------------


def test_required_needs_every_other_ci_job() -> None:
    """A job missing from `needs` can fail while `required` stays green."""
    other_jobs = set(_ci_jobs()) - {_AGGREGATOR}

    assert set(_needs()) == other_jobs
    assert len(_needs()) == len(set(_needs())), f"duplicate entries in needs: {_needs()}"


def test_required_needs_the_named_blocking_lanes() -> None:
    """Positive control: an empty ci.yml would satisfy the completeness test."""
    assert {"lint", "security-tests", "secret-scan"} <= set(_needs())
    assert _called_workflows() == {
        "security-tests": "security-tests.yml",
        "secret-scan": _SECRET_SCAN,
    }


def test_required_always_runs_and_is_named_for_branch_protection() -> None:
    """A skipped required check reads as passing, so it must run on failure too."""
    job = _aggregator()

    assert job.get("name") == _AGGREGATOR
    assert job.get("if") == "${{ always() }}"


@pytest.mark.parametrize("workflow", _NEVER_REQUIRED)
def test_advisory_workflows_are_not_lanes(workflow: str) -> None:
    assert workflow not in _called_workflows().values()


# -- aggregator step behaviour ------------------------------------------------


def _aggregator_script() -> str:
    for step in _aggregator()["steps"]:
        if step.get("name") == _AGGREGATOR_STEP:
            assert step["env"]["RESULTS"] == "${{ toJSON(needs.*.result) }}"
            return str(step["run"])
    pytest.fail(f"the {_AGGREGATOR!r} job has no {_AGGREGATOR_STEP!r} step")


def _run_aggregator(results: list[str]) -> subprocess.CompletedProcess[str]:
    bash = shutil.which("bash")
    assert bash is not None, "bash is required to execute the aggregator step"
    return subprocess.run(  # noqa: S603 -- the workflow's own step script
        [bash, "-c", _aggregator_script()],
        env={"PATH": "/usr/bin:/bin", "RESULTS": json.dumps(results, indent=2)},
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    "results",
    [
        pytest.param(["success", "failure"], id="failure"),
        pytest.param(["success", "cancelled"], id="cancelled"),
        pytest.param(["success", "skipped"], id="skipped"),
        pytest.param([], id="no-results"),
    ],
)
def test_aggregator_fails_unless_every_result_is_success(results: list[str]) -> None:
    assert _run_aggregator(results).returncode == 1


def test_aggregator_passes_when_every_result_is_success() -> None:
    result = _run_aggregator(["success", "success", "success"])

    assert result.returncode == 0, result.stdout + result.stderr


# -- called lanes -------------------------------------------------------------


@pytest.mark.parametrize("workflow", _CALL_ONLY)
def test_call_only_lanes_have_no_trigger_of_their_own(workflow: str) -> None:
    """Any extra trigger would run the lane a second time beside ci.yml."""
    assert _triggers(_load(_WORKFLOWS / workflow)) == {"workflow_call"}


def test_secret_scan_keeps_its_own_triggers_except_pull_request() -> None:
    """Pull requests reach the scan through ci.yml; a second PR trigger double-runs it."""
    assert _triggers(_load(_WORKFLOWS / _SECRET_SCAN)) == _SECRET_SCAN_TRIGGERS


@pytest.mark.parametrize("workflow", _CALL_ONLY)
def test_call_only_lanes_declare_no_workflow_concurrency(workflow: str) -> None:
    assert "concurrency" not in _load(_WORKFLOWS / workflow)


def test_secret_scan_concurrency_group_differs_from_the_callers() -> None:
    """Called from ci.yml, an equal group expression would cancel the caller."""
    caller = _load(_CI)["concurrency"]["group"]
    callee = _load(_WORKFLOWS / _SECRET_SCAN)["concurrency"]["group"]

    assert "github.event_name" in callee
    assert "github.event_name" not in caller
    assert callee != caller


def test_secret_scan_push_runs_are_grouped_per_commit() -> None:
    """A ref-keyed push group lets a newer push displace the pending scan of an older one."""
    concurrency = _load(_WORKFLOWS / _SECRET_SCAN)["concurrency"]

    assert concurrency["group"] == (
        "${{ github.workflow }}-${{ github.event_name }}-"
        "${{ github.event_name == 'push' && github.sha || github.ref }}"
    )
    assert concurrency["cancel-in-progress"] == "${{ github.event_name == 'pull_request' }}"


def test_the_secret_scan_lane_is_the_canonical_required_job() -> None:
    """The job id and display name are what branch protection and tooling expect."""
    jobs = _ci_jobs()

    assert _SECRET_SCAN_JOB in _needs()
    assert jobs[_SECRET_SCAN_JOB]["name"] == "secret scan"
    assert _called_workflows()[_SECRET_SCAN_JOB] == _SECRET_SCAN


def test_secret_scan_caller_grants_what_the_callee_declares() -> None:
    """A called workflow cannot exceed the permissions its calling job grants."""
    callee = _load(_WORKFLOWS / _SECRET_SCAN)["permissions"]
    caller = _ci_jobs()[_SECRET_SCAN_JOB]["permissions"]

    assert caller == callee


# -- service image pins ---------------------------------------------------------

# A service image is pinned by an immutable digest, or is an output of the
# lane's own testdb-config job, whose step takes it from tools/testdb/testdb.env
# through that file's one validator; the file's digest pins are its contract.
# Accepted only for the image keys that validator exports, and only when the
# producing job in the same workflow wires that output straight from the
# validating step. Any other expression is refused, and any missing piece fails
# closed.
_CONFIG_JOB = "testdb-config"
_VALIDATED_IMAGE_KEYS = frozenset({"PGVECTOR_IMAGE", "REDIS_IMAGE"})
_CONFIG_IMAGE_OUTPUT = re.compile(r"\$\{\{ needs\.testdb-config\.outputs\.([A-Z][A-Z0-9_]*) \}\}")
_STEP_OUTPUT = re.compile(r"\$\{\{ steps\.([A-Za-z0-9_-]+)\.outputs\.([A-Z][A-Z0-9_]*) \}\}")
_EXPORT_RUN = 'python3 tools/testdb/testdb.py check-env --github-output "$GITHUB_OUTPUT"'


def _job_needs(job: dict[str, Any]) -> list[str]:
    needs = job.get("needs", [])
    return [needs] if isinstance(needs, str) else list(needs)


def _gated_workflows() -> list[tuple[str, dict[str, Any]]]:
    """(filename, workflow) for ci.yml and every lane it calls."""
    called = sorted(set(_called_workflows().values()))
    return [("ci.yml", _load(_CI)), *((name, _load(_WORKFLOWS / name)) for name in called)]


def _exports_validated(workflow: dict[str, Any], key: str) -> bool:
    """Whether the workflow's config job sets output ``key`` from a check-env export step."""
    producer = (workflow.get("jobs") or {}).get(_CONFIG_JOB)
    if not isinstance(producer, dict):
        return False
    step_ref = _STEP_OUTPUT.fullmatch(str((producer.get("outputs") or {}).get(key, "")))
    if step_ref is None or step_ref.group(2) != key:
        return False
    steps = list(producer.get("steps") or [])
    matches = [i for i, step in enumerate(steps) if step.get("id") == step_ref.group(1)]
    if len(matches) != 1 or str(steps[matches[0]].get("run", "")).strip() != _EXPORT_RUN:
        return False
    return not any("GITHUB_OUTPUT" in str(step.get("run", "")) for step in steps[matches[0] + 1 :])


def _is_pinned_image(workflow: dict[str, Any], job: dict[str, Any], image: str) -> bool:
    if re.search(r"@sha256:[0-9a-f]{64}$", image):
        return True
    output = _CONFIG_IMAGE_OUTPUT.fullmatch(image)
    if output is None or output.group(1) not in _VALIDATED_IMAGE_KEYS:
        return False
    return _CONFIG_JOB in _job_needs(job) and _exports_validated(workflow, output.group(1))


def test_every_service_image_in_the_required_graph_is_pinned() -> None:
    images = [
        (f"{name}:{job_id}", workflow, job, str(service["image"]))
        for name, workflow in _gated_workflows()
        for job_id, job in workflow["jobs"].items()
        for service in (job.get("services") or {}).values()
    ]

    assert images, "no service containers found; the assertion below would be vacuous"
    for label, workflow, job, image in images:
        assert _is_pinned_image(workflow, job, image), f"{label} runs {image!r} by tag"


def _config_workflow(
    outputs: dict[str, str] | None = None, steps: list[dict[str, str]] | None = None
) -> dict[str, Any]:
    """A workflow whose config job exports every image key from a check-env step by default."""
    keys = [*_VALIDATED_IMAGE_KEYS, "UNRELATED_IMAGE"]
    default_outputs = {key: f"${{{{ steps.pins.outputs.{key} }}}}" for key in keys}
    default_steps = [{"id": "other", "run": "true"}, {"id": "pins", "run": _EXPORT_RUN}]
    producer = {
        "outputs": default_outputs if outputs is None else outputs,
        "steps": default_steps if steps is None else steps,
    }
    return {"jobs": {_CONFIG_JOB: producer}}


_PGVECTOR_OUTPUT = "${{ needs.testdb-config.outputs.PGVECTOR_IMAGE }}"


@pytest.mark.parametrize(
    ("workflow", "needs", "image", "is_pinned"),
    [
        pytest.param(
            _config_workflow(), None, "pgvector/pgvector@sha256:" + "a" * 64, True, id="digest"
        ),
        pytest.param(
            _config_workflow(),
            None,
            "pgvector/pgvector:0.8.6-pg17@sha256:" + "a" * 64,
            True,
            id="tag-and-digest",
        ),
        pytest.param(
            _config_workflow(), None, "pgvector/pgvector@sha256:" + "a" * 63, False, id="short"
        ),
        pytest.param(_config_workflow(), _CONFIG_JOB, _PGVECTOR_OUTPUT, True, id="pgvector-output"),
        pytest.param(
            _config_workflow(),
            _CONFIG_JOB,
            "${{ needs.testdb-config.outputs.REDIS_IMAGE }}",
            True,
            id="redis-output",
        ),
        pytest.param(
            _config_workflow(), ["other", _CONFIG_JOB], _PGVECTOR_OUTPUT, True, id="needs-list"
        ),
        pytest.param(_config_workflow(), _CONFIG_JOB, "pgvector/pgvector:pg17", False, id="tag"),
        pytest.param(
            _config_workflow(),
            _CONFIG_JOB,
            "${{ needs.testdb-config.outputs.PG_MAJOR }}",
            False,
            id="not-an-image",
        ),
        pytest.param(
            _config_workflow(),
            _CONFIG_JOB,
            "${{ needs.testdb-config.outputs.UNRELATED_IMAGE }}",
            False,
            id="unvalidated-key",
        ),
        pytest.param(
            _config_workflow(),
            _CONFIG_JOB,
            "${{ needs.other.outputs.PGVECTOR_IMAGE }}",
            False,
            id="other-job",
        ),
        pytest.param(
            _config_workflow(), _CONFIG_JOB, "${{ inputs.PGVECTOR_IMAGE }}", False, id="input"
        ),
        pytest.param(
            _config_workflow(),
            _CONFIG_JOB,
            f"{_PGVECTOR_OUTPUT}-suffix",
            False,
            id="suffixed-output",
        ),
        pytest.param(_config_workflow(), None, _PGVECTOR_OUTPUT, False, id="consumer-no-needs"),
        pytest.param(
            _config_workflow(), ["other"], _PGVECTOR_OUTPUT, False, id="consumer-needs-other"
        ),
        pytest.param(
            _config_workflow(
                outputs={"PGVECTOR_IMAGE": "${{ steps.other.outputs.PGVECTOR_IMAGE }}"}
            ),
            _CONFIG_JOB,
            _PGVECTOR_OUTPUT,
            False,
            id="producer-different-step",
        ),
        pytest.param(
            _config_workflow(outputs={"PGVECTOR_IMAGE": "${{ steps.pins.outputs.REDIS_IMAGE }}"}),
            _CONFIG_JOB,
            _PGVECTOR_OUTPUT,
            False,
            id="producer-different-key",
        ),
        pytest.param(
            _config_workflow(outputs={"PGVECTOR_IMAGE": "pgvector/pgvector:pg17"}),
            _CONFIG_JOB,
            _PGVECTOR_OUTPUT,
            False,
            id="producer-literal",
        ),
        pytest.param(
            _config_workflow(outputs={}),
            _CONFIG_JOB,
            _PGVECTOR_OUTPUT,
            False,
            id="producer-no-output",
        ),
        pytest.param(
            _config_workflow(steps=[{"id": "pins", "run": "echo PGVECTOR_IMAGE=pgvector:pg17"}]),
            _CONFIG_JOB,
            _PGVECTOR_OUTPUT,
            False,
            id="producer-step-not-validator",
        ),
        pytest.param(
            _config_workflow(steps=[{"id": "pins", "run": _EXPORT_RUN}] * 2),
            _CONFIG_JOB,
            _PGVECTOR_OUTPUT,
            False,
            id="producer-step-id-ambiguous",
        ),
        pytest.param(
            _config_workflow(steps=[{"id": "pins", "run": f"true # {_EXPORT_RUN}"}]),
            _CONFIG_JOB,
            _PGVECTOR_OUTPUT,
            False,
            id="producer-validator-only-in-comment",
        ),
        pytest.param(
            _config_workflow(steps=[{"id": "pins", "run": f"{_EXPORT_RUN} --file other.env"}]),
            _CONFIG_JOB,
            _PGVECTOR_OUTPUT,
            False,
            id="producer-validator-other-file",
        ),
        pytest.param(
            _config_workflow(
                steps=[
                    {"id": "pins", "run": _EXPORT_RUN},
                    {"id": "later", "run": 'echo PGVECTOR_IMAGE=x >> "$GITHUB_OUTPUT"'},
                ]
            ),
            _CONFIG_JOB,
            _PGVECTOR_OUTPUT,
            False,
            id="producer-output-overwritten-later",
        ),
        pytest.param(
            _config_workflow(steps=[{"id": "pins", "run": f"\n{_EXPORT_RUN}\n"}]),
            _CONFIG_JOB,
            _PGVECTOR_OUTPUT,
            True,
            id="producer-validator-surrounding-whitespace",
        ),
        pytest.param({"jobs": {}}, _CONFIG_JOB, _PGVECTOR_OUTPUT, False, id="no-producer-job"),
    ],
)
def test_a_service_image_is_pinned_only_by_digest_or_a_validated_config_output(
    workflow: dict[str, Any], needs: str | list[str] | None, image: str, is_pinned: bool
) -> None:
    job = {} if needs is None else {"needs": needs}

    assert _is_pinned_image(workflow, job, image) is is_pinned
