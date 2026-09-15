"""Contract test: the authoritative CI job keeps its pinned PostgreSQL client.

``tests/integration/test_audit_log_dedup_read_paths.py`` and
``test_audit_log_app_dedup_read.py`` shell out to ``psql`` and to
``deployment/scripts/verify-db-invariants.sh``. Those tests FAIL rather than skip
when ``psql`` is absent -- deliberately, so a missing client cannot report green
for a repair path nothing exercised. That makes the client a job PREREQUISITE.

A GitHub runner image ships whatever client major it happens to ship, and that
can change without any commit to this repo. Once the runner has some ``psql`` on
PATH, an accidental removal of the install step would not fail loudly -- it would
run the repair tests against a mismatched client major, whose catalog output the
tests parse. So the pin is asserted here, in a unit test that needs no database:
the step must exist in the job that runs the integration suite, must pin the same
major as the Postgres service image, and must put the versioned PGDG bin dir on
PATH for later steps.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

_WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "security-tests.yml"

# The job whose steps run the psql-dependent integration suite.
_JOB = "security-tests"
_INSTALL_STEP = "Install PostgreSQL 17 client"
_VERIFY_STEP = "Verify psql major matches the pin"

# Single source of truth for this test: the server major the service container
# provides. Every other assertion is derived from what the workflow declares, so
# a coordinated bump of service image + client pin passes while a drift of either
# one alone fails.
_EXPECTED_MAJOR = "17"


def _workflow() -> dict[str, Any]:
    parsed = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict), f"{_WORKFLOW} did not parse as a mapping"
    return parsed


def _job() -> dict[str, Any]:
    jobs = _workflow()["jobs"]
    assert _JOB in jobs, f"workflow no longer defines the {_JOB!r} job: {sorted(jobs)}"
    job = jobs[_JOB]
    assert isinstance(job, dict)
    return job


def _steps() -> list[dict[str, Any]]:
    return [step for step in _job()["steps"] if isinstance(step, dict)]


def _step(name: str) -> dict[str, Any]:
    for step in _steps():
        if step.get("name") == name:
            return step
    pytest.fail(
        f"the {_JOB!r} job has no {name!r} step. The audit_log dedup repair and "
        "invariant tests fail (not skip) without psql, and they parse catalog "
        "output from a client whose major must match the Postgres service. If "
        "this step was renamed, update this test; if it was removed, restore it. "
        f"Present steps: {[s.get('name') for s in _steps()]}"
    )


def test_service_image_declares_the_expected_postgres_major() -> None:
    """The pin this test enforces is the major the service container actually runs."""
    # Arrange / Act
    image = _job()["services"]["postgres"]["image"]

    # Assert
    assert image.endswith(f"pg{_EXPECTED_MAJOR}"), (
        f"the Postgres service image {image!r} is not major {_EXPECTED_MAJOR}; the "
        "client pin asserted by this test must be bumped with it"
    )


def test_postgres_client_install_step_pins_the_service_major() -> None:
    """The install step exists and pins the same major as the service image."""
    # Arrange / Act
    step = _step(_INSTALL_STEP)

    # Assert
    assert step.get("env", {}).get("PG_MAJOR") == _EXPECTED_MAJOR, (
        f"{_INSTALL_STEP!r} must pin PG_MAJOR={_EXPECTED_MAJOR} to match the "
        f"Postgres service image, got {step.get('env')!r}"
    )
    assert "postgresql-client-${PG_MAJOR}" in step["run"], (
        f"{_INSTALL_STEP!r} must install the versioned client package from the pin, "
        "not an unversioned postgresql-client that follows the runner image"
    )


def test_postgres_client_install_step_puts_the_pinned_binary_first_on_path() -> None:
    """Installing is not enough -- later steps must resolve psql to the pinned major.

    PGDG installs into ``/usr/lib/postgresql/<major>/bin``, which is not on PATH.
    Without the GITHUB_PATH append, ``psql`` would still resolve to whatever the
    runner image ships.
    """
    # Arrange / Act
    run = _step(_INSTALL_STEP)["run"]

    # Assert
    assert re.search(r'/usr/lib/postgresql/\$\{PG_MAJOR\}/bin"?\s*>>\s*"?\$GITHUB_PATH', run), (
        f"{_INSTALL_STEP!r} must append the versioned PGDG bin dir to GITHUB_PATH "
        f"so later steps resolve psql to the pin:\n{run}"
    )


def test_a_psql_major_verification_step_guards_the_pin_at_runtime() -> None:
    """A wrong-major client must fail the job, not silently run the repair tests.

    The install step can succeed while PATH still resolves an older client (a
    stale GITHUB_PATH ordering, a package that installed elsewhere). This step is
    what turns that into a red job.
    """
    # Arrange / Act
    step = _step(_VERIFY_STEP)

    # Assert
    assert step.get("env", {}).get("PG_MAJOR") == _EXPECTED_MAJOR, (
        f"{_VERIFY_STEP!r} must check against PG_MAJOR={_EXPECTED_MAJOR}, got {step.get('env')!r}"
    )
    run = step["run"]
    assert "psql --version" in run, f"{_VERIFY_STEP!r} must read the actual client version"
    assert "exit 1" in run, (
        f"{_VERIFY_STEP!r} must fail the job on a mismatch; a version print alone "
        f"is not a gate:\n{run}"
    )


def test_the_client_is_installed_before_the_integration_suite_runs() -> None:
    """Ordering is load-bearing: the suite that needs psql must run after the install."""
    # Arrange
    names = [step.get("name") for step in _steps()]
    suite_step = next(
        (name for name in names if name and name.startswith("Integration tests")),
        None,
    )
    assert suite_step is not None, (
        f"the {_JOB!r} job no longer has an 'Integration tests' step: {names}"
    )

    # Act / Assert
    for required in (_INSTALL_STEP, _VERIFY_STEP):
        assert names.index(required) < names.index(suite_step), (
            f"{required!r} must precede {suite_step!r}, otherwise the psql-dependent "
            f"tests run against whatever client the runner image ships: {names}"
        )
