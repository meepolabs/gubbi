"""Shared harness for exercising ``deployment/scripts/verify-db-invariants.sh``.

The verifier is a post-deploy assertion script, so a test that only checks its
exit code proves little: on a gubbi-only database it can report failures wholly
unrelated to the invariant under test (cloud-side objects it does not create).
Every test here instead diffs its FAILURE TAG SET between a clean run and a
mutated one -- the tag must be ABSENT while the invariant holds and PRESENT once
it is broken, which is a check that can fail in both directions.

An aborted run (a psql error under ``set -e``, before the report block) emits an
empty tag set for reasons unrelated to the assertion, which would let every
mutation test pass vacuously. So an abort fails loudly rather than skipping.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
VERIFY_SCRIPT = REPO_ROOT / "deployment" / "scripts" / "verify-db-invariants.sh"


def psql_bin() -> str:
    """The ``psql`` binary, or a hard failure.

    ``psql`` is not optional: the verifier and the grants.sql repair path both
    shell out to it, so skipping when it is absent would report green for a
    capability nothing exercised.
    """
    found = shutil.which("psql")
    if found is None:
        pytest.fail(
            "psql is not on PATH -- it is a required prerequisite for the "
            "grants.sql repair and verify-db-invariants.sh paths, not an "
            "optional extra. Install a postgresql-client matching the server "
            "major version, or expose the test container's psql on PATH."
        )
    return found


def run_argv(argv: list[str], **extra_env: str) -> subprocess.CompletedProcess[str]:
    """Run ``argv`` from the repo root with ``extra_env`` layered over os.environ."""
    return subprocess.run(  # noqa: S603 -- argv entries are literals or local paths
        argv,
        cwd=REPO_ROOT,
        env={**os.environ, **extra_env},
        capture_output=True,
        text=True,
        check=False,
    )


def verifier_failure_tags(dsn: str) -> frozenset[str]:
    """Run the verifier against ``dsn`` and return the ``FAIL: <tag>`` lines it emitted."""
    return verifier_run(dsn).tags


@dataclass(frozen=True)
class VerifierRun:
    """One verifier execution: its raw streams plus the failure tags it reported.

    Tests that only diff tag sets use :attr:`tags`. Tests asserting what the
    verifier may PRINT need the raw streams, and tests asserting it refused need
    :attr:`exit_code` -- all from a single run, so the views can never describe
    different executions.
    """

    stdout: str
    stderr: str
    tags: frozenset[str]
    exit_code: int

    @property
    def output(self) -> str:
        """Both streams, as a deploy log would interleave them."""
        return f"{self.stdout}\n{self.stderr}"


def verifier_run(dsn: str, *, require_report: bool = True) -> VerifierRun:
    """Execute the verifier once and return its streams, tags and exit code.

    Reaching the report block is asserted by default, not assumed: an abort under
    ``set -e`` (a psql error before the report) emits an EMPTY tag set for reasons
    unrelated to the invariant under test, which would let every mutation test
    pass vacuously. ``require_report=False`` is for the tests that deliberately
    exercise an aborting shape and assert on it themselves.
    """
    psql_bin()  # the verifier shells out to psql; fail early and with a clear reason
    result = run_argv([str(VERIFY_SCRIPT)], JOURNAL_DB_MIGRATION_URL=dsn)
    if require_report and not reached_report_block(result.stdout, result.stderr):
        pytest.fail(
            "verify-db-invariants.sh aborted before reaching its report block, so "
            "its failure tag set is empty for reasons unrelated to what is being "
            "asserted. Every prerequisite it references (psql, the gubbi tables) "
            "is provisioned by the test fixtures, and a missing ROLE is supposed "
            f"to be a labeled failure rather than an abort, so this is a defect:\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return VerifierRun(
        stdout=result.stdout,
        stderr=result.stderr,
        exit_code=result.returncode,
        tags=frozenset(
            line.removeprefix("FAIL: ").strip()
            for line in result.stderr.splitlines()
            if line.startswith("FAIL: ")
        ),
    )


def reached_report_block(stdout: str, stderr: str) -> bool:
    """Whether a verifier run got as far as aggregating its report.

    Exactly one of the two report headers is printed on every completed run, so
    their absence is the abort signal.
    """
    return "--- invariant check failed ---" in stderr or "verify-db-invariants: OK" in stdout


def tags_mentioning(tags: frozenset[str], needle: str) -> frozenset[str]:
    """The subset of ``tags`` containing ``needle``."""
    return frozenset(tag for tag in tags if needle in tag)


def assert_clean_of(dsn: str, needle: str) -> frozenset[str]:
    """Baseline the verifier: it must NOT flag ``needle`` while the invariant holds."""
    clean = verifier_failure_tags(dsn)
    assert not tags_mentioning(clean, needle), (
        f"verifier flagged {needle!r} on an intact DB: {sorted(clean)}"
    )
    return clean
