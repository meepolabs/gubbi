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
_INSTALL_STEP = "Install PostgreSQL client"
_VERIFY_STEP = "Verify psql major matches the pin"

# The server major the digest-pinned service containers provide. Both images are
# pinned by digest, which carries no readable version, so the workflow DECLARES the
# major in one job-level variable and every consumer derives from it. This constant
# is the test's independent copy: a coordinated bump of digest + declared major +
# this constant passes, while drift of any one alone fails.
_EXPECTED_MAJOR = "17"

# The job-level variable that declares it.
_DECLARED_MAJOR_VAR = "PGVECTOR_PG_MAJOR"


# The ONE accepted expression. Matching is by EQUALITY, not substring: a substring
# test accepts `${{ env.PGVECTOR_PG_MAJOR }}-beta`, `${{ env.PGVECTOR_PG_MAJOR_OLD }}`
# and `x${{ env.PGVECTOR_PG_MAJOR }}`, each of which resolves to something other than
# the declared major while still "containing" the reference.
_ACCEPTED_MAJOR_EXPRESSION = "${{ env." + _DECLARED_MAJOR_VAR + " }}"


def _declared_major() -> str:
    """The major the job declares once, for every consumer to derive from."""
    return str((_job().get("env") or {}).get(_DECLARED_MAJOR_VAR, ""))


def _pg_major_expression(step: dict[str, Any]) -> str:
    """A step's raw ``PG_MAJOR`` value, exactly as the workflow spells it."""
    return str((step.get("env") or {}).get("PG_MAJOR", ""))


def _resolved_pg_major(step: dict[str, Any]) -> str:
    """The major a step's ``PG_MAJOR`` resolves to at runtime.

    Only the single accepted expression is resolved. Anything else is returned
    unchanged, so a malformed or differently-targeted expression fails the equality
    assertions rather than being quietly normalised into the right answer.
    """
    raw = _pg_major_expression(step)
    if raw == _ACCEPTED_MAJOR_EXPRESSION:
        return _declared_major()
    return raw


def test_the_workflow_declares_the_postgres_major_once() -> None:
    """A digest-pinned image carries no version, so the major must be declared.

    Without an explicit declaration the client pin would have to infer the major
    from a tag -- and there is no longer a tag to infer it from.
    """
    declared = (_job().get("env") or {}).get(_DECLARED_MAJOR_VAR)
    assert str(declared) == _EXPECTED_MAJOR, (
        f"the job must declare {_DECLARED_MAJOR_VAR}={_EXPECTED_MAJOR} as the single "
        f"source for the client pin, got {declared!r}"
    )


@pytest.mark.parametrize("step_name", [_INSTALL_STEP, _VERIFY_STEP])
def test_client_pin_steps_use_exactly_the_accepted_major_expression(step_name: str) -> None:
    """``PG_MAJOR`` must be the accepted expression verbatim -- no literals, no variants.

    Equality, not containment. A substring check would accept a suffixed expression,
    a prefixed one, or a reference to a similarly-named variable, each of which
    resolves to something other than the declared major while still containing the
    reference text. Since both service images are digest-pinned, nothing else would
    catch the divergence until a catalog-output parse failed at runtime.
    """
    actual = _pg_major_expression(_step(step_name))
    assert actual == _ACCEPTED_MAJOR_EXPRESSION, (
        f"{step_name!r} must set PG_MAJOR to exactly "
        f"{_ACCEPTED_MAJOR_EXPRESSION!r}, got {actual!r}. A literal can drift from "
        "the declaration, and a variant expression resolves to a different value "
        "while still looking like a reference."
    )


@pytest.mark.parametrize(
    "rejected",
    [
        "17",
        "${{ env.PGVECTOR_PG_MAJOR }}-beta",
        "x${{ env.PGVECTOR_PG_MAJOR }}",
        "${{ env.PGVECTOR_PG_MAJOR_OLD }}",
        "${{ env.SOME_OTHER_MAJOR }}",
        "${{ envPGVECTOR_PG_MAJOR }}",
        "$PGVECTOR_PG_MAJOR",
        "",
    ],
    ids=[
        "bare_literal",
        "suffixed",
        "prefixed",
        "similar_variable_name",
        "other_variable",
        "malformed_expression",
        "shell_style_reference",
        "empty",
    ],
)
def test_the_accepted_major_expression_rejects_every_variant(rejected: str) -> None:
    """The contract's own discriminating power, checked against each near-miss.

    Without this, the equality assertion above could be satisfied by a comparison
    that happened to be too permissive, and the near-misses it is meant to exclude
    would go unexercised. Each value here resolves to something other than the
    declared major, or to nothing at all.
    """
    assert rejected != _ACCEPTED_MAJOR_EXPRESSION, (
        f"{rejected!r} must not be accepted as the major expression"
    )
    resolved = _resolved_pg_major({"env": {"PG_MAJOR": rejected}})
    assert resolved != _declared_major() or rejected == _declared_major(), (
        f"{rejected!r} resolved to the declared major without being the accepted "
        f"expression, so the resolver is normalising a variant into the right answer"
    )


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


def test_the_working_service_is_pinned_by_digest_not_by_tag() -> None:
    """The working service is digest-pinned, like the disposable one.

    A tag can move to a different build with no commit here, changing the server the
    RLS, grant and posture contracts run against and desynchronising it from the
    installed client major.
    """
    # Arrange / Act
    image = str(_job()["services"]["postgres"]["image"])

    # Assert
    assert "@" in image, (
        f"the working Postgres service must be pinned by digest, not by tag: {image}"
    )
    assert ":" not in image.split("@")[0], (
        f"the reference carries a mutable TAG, so the pin is not a pin: {image}"
    )


def test_postgres_client_install_step_pins_the_service_major() -> None:
    """The install step exists and pins the same major as the service image."""
    # Arrange / Act
    step = _step(_INSTALL_STEP)

    # Assert
    assert _resolved_pg_major(step) == _EXPECTED_MAJOR, (
        f"{_INSTALL_STEP!r} must resolve PG_MAJOR to {_EXPECTED_MAJOR} -- the major "
        f"the pinned service digests provide -- got {step.get('env')!r}"
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
    assert _resolved_pg_major(step) == _EXPECTED_MAJOR, (
        f"{_VERIFY_STEP!r} must check against the declared major {_EXPECTED_MAJOR}, "
        f"got {step.get('env')!r}"
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


# ---------------------------------------------------------------------------
# The disposable-cluster lane.
#
# The missing-required-role contracts need a role to be genuinely ABSENT, and
# roles are cluster-global: staging that on the working cluster would mean
# dropping a shared role, which the harness cannot restore (it captures four
# attributes, not the password or rolconfig). So absence gets its OWN throwaway
# service, and the DSN that unlocks those cases is exported for exactly one step.
#
# Both halves are asserted here, in a test that needs no database: the service
# must exist on its own port, and the variable must NOT be visible job-wide --
# because the code it gates drops roles, and a job-level variable would make it
# visible to steps whose DSN is the working cluster.
# ---------------------------------------------------------------------------

_DISPOSABLE_SERVICE = "postgres_disposable"
_DISPOSABLE_ENV = "TEST_DISPOSABLE_CLUSTER_URL"
_SUITE_STEP = "Integration tests (Postgres-only; excludes hosted_live live-stack)"


def _services() -> dict[str, Any]:
    services = _job().get("services")
    assert isinstance(services, dict), f"the {_JOB!r} job declares no services"
    return services


def _service_port_mappings(service: str) -> list[str]:
    ports = _services()[service].get("ports")
    assert isinstance(ports, list), f"service {service!r} declares no port mappings"
    return [str(port) for port in ports]


def test_a_separate_disposable_postgres_service_is_provisioned() -> None:
    """The missing-role lane gets its own cluster, on the same pinned major."""
    services = _services()
    assert _DISPOSABLE_SERVICE in services, (
        f"the {_JOB!r} job no longer declares the {_DISPOSABLE_SERVICE!r} service. "
        "Without it the missing-required-role contracts skip, which reports green "
        f"for behaviour nothing measured. Present services: {sorted(services)}"
    )
    image = str(services[_DISPOSABLE_SERVICE]["image"])
    assert image.startswith(f"{_PGVECTOR_REPOSITORY}@"), (
        "the disposable service must be pinned by immutable digest, since it gates "
        f"contracts that DROP roles: {image}"
    )


def test_the_disposable_service_uses_a_distinct_port() -> None:
    """Two clusters on one runner must not contend for a port.

    A shared host port would silently make "the disposable cluster" the working
    one, and the missing-role lane would then drop shared roles on the cluster
    every other test depends on.
    """
    working = _service_port_mappings("postgres")
    disposable = _service_port_mappings(_DISPOSABLE_SERVICE)
    working_hosts = {mapping.split(":")[0] for mapping in working}
    disposable_hosts = {mapping.split(":")[0] for mapping in disposable}
    assert not (working_hosts & disposable_hosts), (
        "the working and disposable Postgres services share a host port, so the "
        "role-dropping lane would target the cluster every other test depends on: "
        f"working={working} disposable={disposable}"
    )


def test_the_disposable_cluster_url_is_scoped_to_the_integration_step_only() -> None:
    """The role-dropping DSN is a step variable, never a job-wide one.

    Job scope would expose it to every step, including those whose DSN is the
    working cluster -- and the code it unlocks drops shared roles.
    """
    job_env = _job().get("env") or {}
    assert _DISPOSABLE_ENV not in job_env, (
        f"{_DISPOSABLE_ENV} is set at JOB level. It unlocks code that DROPS shared "
        "roles, so it must be scoped to the single step that owns those contracts; "
        f"job-wide exposure risks pointing it at the working cluster: {sorted(job_env)}"
    )
    owning = _step(_SUITE_STEP).get("env") or {}
    assert _DISPOSABLE_ENV in owning, (
        f"{_DISPOSABLE_ENV} is not set on the {_SUITE_STEP!r} step, so the "
        "missing-required-role contracts would skip there and report green for "
        f"behaviour nothing measured: {sorted(owning)}"
    )


def test_the_disposable_cluster_url_points_at_the_disposable_service_port() -> None:
    """The scoped DSN must name the throwaway service's port, not the working one."""
    dsn = str((_step(_SUITE_STEP).get("env") or {})[_DISPOSABLE_ENV])
    disposable_hosts = {
        mapping.split(":")[0] for mapping in _service_port_mappings(_DISPOSABLE_SERVICE)
    }
    assert any(f":{host}/" in dsn for host in disposable_hosts), (
        f"{_DISPOSABLE_ENV} does not point at the {_DISPOSABLE_SERVICE!r} host port "
        f"{sorted(disposable_hosts)}. If it points at the working cluster, the "
        f"role-dropping lane would destroy shared roles: {dsn}"
    )
    working_hosts = {mapping.split(":")[0] for mapping in _service_port_mappings("postgres")}
    assert not any(f":{host}/" in dsn for host in working_hosts), (
        f"{_DISPOSABLE_ENV} names the WORKING cluster's port {sorted(working_hosts)}, "
        f"which the role-dropping lane must never target: {dsn}"
    )


def test_no_other_step_exposes_the_disposable_cluster_url() -> None:
    """Exactly one step may carry the role-dropping DSN."""
    carriers = [
        str(step.get("name")) for step in _steps() if _DISPOSABLE_ENV in (step.get("env") or {})
    ]
    assert carriers == [_SUITE_STEP], (
        f"{_DISPOSABLE_ENV} must be exposed to exactly one step -- it unlocks code "
        f"that drops shared roles: {carriers}"
    )


# The immutable digest the disposable service is pinned to. Verified out of band:
# it resolves and runs PostgreSQL 17.11 with pgvector 0.8.6. Bumping the pin means
# re-verifying that and updating this constant in the same change.
_PGVECTOR_REPOSITORY = "pgvector/pgvector"
_DISPOSABLE_IMAGE_DIGEST = "sha256:cf134a767f474095eeba57e0117be8e568e011a63f33fbf252f14c9b760f8e6f"


def test_the_disposable_service_is_pinned_to_the_verified_digest() -> None:
    """The exact digest, not a tag and not some other digest.

    A mutable tag can point at a different build tomorrow with no commit here. This
    service gates the missing-required-role contracts, which DROP roles, so a silent
    change to its Postgres build could alter role semantics underneath them.
    """
    image = str(_services()[_DISPOSABLE_SERVICE]["image"])
    assert image == f"{_PGVECTOR_REPOSITORY}@{_DISPOSABLE_IMAGE_DIGEST}", (
        "the disposable Postgres service must be pinned to the exact verified "
        f"digest:\nwant={_PGVECTOR_REPOSITORY}@{_DISPOSABLE_IMAGE_DIGEST}\ngot={image}"
    )


def test_the_disposable_service_pin_is_not_tag_only() -> None:
    """A tag reference, with or without a digest suffix, is rejected.

    Spelled out separately from the equality check above so the FAILURE MODE is
    named: ``pgvector/pgvector:pg17`` is what a well-meaning revert would produce,
    and it must be caught as a pin regression rather than a typo.
    """
    image = str(_services()[_DISPOSABLE_SERVICE]["image"])
    assert ":" not in image.split("@")[0], (
        "the disposable service reference carries a TAG. A tag is mutable, so the "
        f"pin would not be a pin: {image}"
    )
    assert image.count("@") == 1, f"the reference must name exactly one digest: {image}"


def test_the_disposable_service_digest_has_the_expected_algorithm_and_length() -> None:
    """A digest must be a full sha256, not a truncated or differently-hashed one.

    A short digest is not immutable in the same way -- registries resolve prefixes --
    so the algorithm and the full 64 hex characters are both required.
    """
    digest = str(_services()[_DISPOSABLE_SERVICE]["image"]).split("@", 1)[1]
    algorithm, _, hexdigest = digest.partition(":")
    assert algorithm == "sha256", f"the digest must be sha256, not {algorithm!r}: {digest}"
    assert re.fullmatch(r"[0-9a-f]{64}", hexdigest), (
        "the digest must be a full 64-character lowercase sha256; a truncated "
        f"prefix is resolvable and therefore not immutable: {digest}"
    )


def test_both_services_share_the_same_verified_digest() -> None:
    """One digest, two services -- so role semantics cannot differ between them.

    The missing-role contracts observe absence on the disposable cluster and the
    posture contracts observe presence on the working one. If the two ran different
    PostgreSQL builds, a behaviour measured on one would not transfer to the other,
    and the whole split would be unsound.
    """
    working = str(_services()["postgres"]["image"])
    disposable = str(_services()[_DISPOSABLE_SERVICE]["image"])
    expected = f"{_PGVECTOR_REPOSITORY}@{_DISPOSABLE_IMAGE_DIGEST}"
    assert working == expected, (
        f"the working service is not on the verified digest:\nwant={expected}\ngot={working}"
    )
    assert disposable == expected, (
        f"the disposable service is not on the verified digest:\nwant={expected}\ngot={disposable}"
    )
    assert working == disposable, (
        "the two services must run the IDENTICAL image, or behaviour measured on "
        f"one does not transfer to the other:\nworking={working}\ndisposable={disposable}"
    )


def test_no_service_falls_back_to_a_mutable_tag() -> None:
    """Every Postgres service in the job is digest-pinned.

    Named separately from the equality check so the FAILURE MODE is explicit: a
    revert to `pgvector/pgvector:pg17` is what a well-meaning change produces, and it
    must read as a pin regression rather than a typo.
    """
    tagged = {
        name: str(service["image"])
        for name, service in _services().items()
        if "@" not in str(service.get("image", ""))
    }
    assert not tagged, f"these services are tag-pinned and therefore mutable: {tagged}"


def test_every_service_digest_has_the_expected_algorithm_and_length() -> None:
    """A truncated or differently-hashed digest is not immutable in the same way.

    Registries resolve digest PREFIXES, so a short digest would still pull -- and
    could pull something else later.
    """
    for name, service in _services().items():
        digest = str(service["image"]).split("@", 1)[1]
        algorithm, _, hexdigest = digest.partition(":")
        assert algorithm == "sha256", (
            f"service {name!r} digest must be sha256, not {algorithm!r}: {digest}"
        )
        assert re.fullmatch(r"[0-9a-f]{64}", hexdigest), (
            f"service {name!r} digest must be a full 64-character lowercase sha256; "
            f"a resolvable prefix is not immutable: {digest}"
        )
