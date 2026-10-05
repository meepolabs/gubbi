"""Unit tests for the test database controller in tools/testdb/testdb.py.

docker is never run: a fake command runner holds a table of containers, answers
``docker container inspect`` from it, and records every argv, so the tests pin
names, labels, argv shape and every refusal path. Each fake container is the
recorded real inspect sample in tests/fixtures/testdb_docker_inspect.json with
its identity fields replaced.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from tests.fixtures.testdb_tool import TESTDB_TOOL, load_testdb_tool

if TYPE_CHECKING:
    from collections.abc import Sequence

pytestmark = pytest.mark.unit

testdb = load_testdb_tool()

_UID = 4242
_TOPLEVEL = Path("/work/checkout-a")
_OTHER_TOPLEVEL = Path("/work/checkout-b")
_GUBBI = ["--repo", "gubbi", "--role", "pg", "--role", "pg-disposable", "--db", "journal_test"]
_PG_IMAGE = "pgvector/pgvector@sha256:" + "a" * 64
_REDIS_IMAGE = "redis:7-alpine@sha256:" + "b" * 64
_ENV_BYTES = f"PGVECTOR_IMAGE={_PG_IMAGE}\nPG_MAJOR=17\nREDIS_IMAGE={_REDIS_IMAGE}\n".encode()
_ENV_DIGEST = hashlib.sha256(_ENV_BYTES).hexdigest()
_SAMPLE = json.loads(
    (Path(__file__).resolve().parents[1] / "fixtures" / "testdb_docker_inspect.json").read_text(
        encoding="ascii"
    )
)


def _sample_entry() -> dict[str, Any]:
    entry: dict[str, Any] = copy.deepcopy(_SAMPLE["container_inspect"][0])
    return entry


@dataclass
class FakeDocker:
    """A scripted docker CLI: containers by name, plus every argv it was handed.

    Commands after the ownership check address containers by full ID, so
    ``start``, ``rm`` and ``exec`` only resolve IDs; a name there is an error.
    """

    containers: dict[str, dict[str, Any]] = field(default_factory=dict)
    calls: list[list[str]] = field(default_factory=list)
    ready: bool = True
    next_port: int = 40000
    daemon_down: bool = False
    probe_results: list[Any] = field(default_factory=list)
    replace_on_start: bool = False

    def __call__(self, argv: Sequence[str], timeout: float | None) -> Any:
        argv = list(argv)
        self.calls.append(argv)
        if self.daemon_down:
            return testdb.CommandResult(1, "", "Cannot connect to the Docker daemon")
        handler = getattr(self, f"_{argv[1]}", None)
        assert handler is not None, f"unexpected docker call {argv!r}"
        return handler(argv[2:])

    def add(self, name: str, labels: dict[str, str], *, running: bool = True) -> str:
        self.next_port += 1
        container_id = hashlib.sha256(f"{name}/{self.next_port}".encode()).hexdigest()
        spec = "6379/tcp" if name.endswith("-redis") else "5432/tcp"
        entry = _sample_entry()
        entry["Id"] = container_id
        entry["Name"] = f"/{name}"
        entry["Config"]["Labels"] = dict(labels)
        entry["State"]["Running"] = running
        binding = {"HostIp": "127.0.0.1", "HostPort": str(self.next_port)}
        entry["NetworkSettings"]["Ports"] = {spec: [binding]}
        self.containers[name] = entry
        return container_id

    def by_id(self, container_id: str) -> dict[str, Any]:
        matches = [entry for entry in self.containers.values() if entry["Id"] == container_id]
        assert len(matches) == 1, f"{container_id!r} is not a container ID"
        return matches[0]

    def _container(self, args: list[str]) -> Any:
        assert args[0] == "inspect"
        entry = self.containers.get(args[1])
        if entry is None:
            return testdb.CommandResult(1, "[]\n", f"Error: No such container: {args[1]}\n")
        return testdb.CommandResult(0, json.dumps([entry]), "")

    def _run(self, args: list[str]) -> Any:
        name = args[args.index("--name") + 1]
        labels = dict(args[i + 1].split("=", 1) for i, arg in enumerate(args) if arg == "--label")
        return testdb.CommandResult(0, f"{self.add(name, labels)}\n", "")

    def _start(self, args: list[str]) -> Any:
        entry = self.by_id(args[0])
        entry["State"]["Running"] = True
        if self.replace_on_start:
            entry["Id"] = "f" * 64
        return testdb.CommandResult(0, "", "")

    def _rm(self, args: list[str]) -> Any:
        entry = self.by_id(args[-1])
        del self.containers[entry["Name"].lstrip("/")]
        return testdb.CommandResult(0, "", "")

    def _exec(self, args: list[str]) -> Any:
        self.by_id(args[0])
        if args[1] == "psql":
            return testdb.CommandResult(0, "", "")
        if self.probe_results:
            return self.probe_results.pop(0)
        if args[1] == "redis-cli":
            return testdb.CommandResult(0, "PONG\n" if self.ready else "", "")
        return testdb.CommandResult(0 if self.ready else 2, "", "")

    def id_of(self, name: str) -> str:
        container_id: str = self.containers[name]["Id"]
        return container_id

    def removed(self) -> list[str]:
        return [call[-1] for call in self.calls if call[1] == "rm"]


@pytest.fixture
def env_file(tmp_path: Path) -> Path:
    path = tmp_path / "testdb.env"
    path.write_bytes(_ENV_BYTES)
    return path


@pytest.fixture
def docker() -> FakeDocker:
    return FakeDocker()


def _ctx(docker: FakeDocker, env_file: Path, toplevel: Path = _TOPLEVEL, uid: int = _UID) -> Any:
    return testdb.Context(
        run=docker,
        sleep=lambda _seconds: None,
        uid=uid,
        checkout=testdb.checkout_from_toplevel(toplevel),
        env_file=env_file,
    )


def _main(docker: FakeDocker, env_file: Path, argv: list[str], **kwargs: Any) -> int:
    return testdb.main(argv, ctx_factory=lambda: _ctx(docker, env_file, **kwargs))


def _short(toplevel: Path = _TOPLEVEL) -> str:
    return hashlib.sha256(str(toplevel).encode()).hexdigest()[:8]


def _name(role: str, toplevel: Path = _TOPLEVEL) -> str:
    return f"testdb-gubbi-{_short(toplevel)}-{role}"


def _labels(role: str, toplevel: Path = _TOPLEVEL, **overrides: str) -> dict[str, str]:
    labels = {
        "ai.gubbi.testdb.repo": "gubbi",
        "ai.gubbi.testdb.role": role,
        "ai.gubbi.testdb.owner-uid": str(_UID),
        "ai.gubbi.testdb.checkout": hashlib.sha256(str(toplevel).encode()).hexdigest(),
        "ai.gubbi.testdb.env-sha256": _ENV_DIGEST,
    }
    return {**labels, **overrides}


def _add_stack(docker: FakeDocker, toplevel: Path = _TOPLEVEL) -> None:
    for role in ("pg", "pg-disposable"):
        docker.add(_name(role, toplevel), _labels(role, toplevel))


def _port(docker: FakeDocker, name: str) -> str:
    port: str = docker.containers[name]["NetworkSettings"]["Ports"]["5432/tcp"][0]["HostPort"]
    return port


def _write_stack_env(docker: FakeDocker, toplevel: Path) -> Path:
    """Write the .testdb.env ``up`` would write for the fake stack's current ports."""
    profile = testdb.make_profile("gubbi", ["pg", "pg-disposable"], ["journal_test"])
    names = {role: _name(role, toplevel) for role in profile.roles}
    ports = {role: int(_port(docker, name)) for role, name in names.items()}
    path = toplevel / ".testdb.env"
    path.write_text(testdb.render_stack_env(testdb.stack_env(profile, names, ports)))
    return path


def _bash_source(cwd: Path, env_name: str, key: str) -> str:
    """Return ``$key`` after ``set -a; . ./<env_name>`` in a real bash."""
    bash = shutil.which("bash")
    assert bash is not None, "bash is required to source the stack env"
    return subprocess.run(  # noqa: S603 -- fixed script, test-controlled file
        [bash, "-c", f'set -a; . ./{env_name}; printf "%s" "${key}"'],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _run_tool(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 -- this interpreter running the controller
        [sys.executable, *args],
        capture_output=True,
        text=True,
        check=False,
    )


# ---------------------------------------------------------------------------
# Module shape
# ---------------------------------------------------------------------------


def test_module_imports_only_the_standard_library() -> None:
    tree = ast.parse(TESTDB_TOOL.read_text(encoding="utf-8"))
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }

    assert "json" in imported
    assert imported - {"__future__"} <= sys.stdlib_module_names


def test_module_runs_on_a_bare_interpreter() -> None:
    result = _run_tool("-I", "-S", str(TESTDB_TOOL), "check-env")

    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# Names, labels and docker run argv
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("toplevel", "role"),
    [
        pytest.param(_TOPLEVEL, "pg", id="pg"),
        pytest.param(_TOPLEVEL, "pg-disposable", id="disposable"),
        pytest.param(_OTHER_TOPLEVEL, "redis", id="other-checkout-redis"),
    ],
)
def test_container_name_is_repo_checkout_hash_and_role(toplevel: Path, role: str) -> None:
    profile = testdb.make_profile("gubbi", [role], [])

    name = testdb.container_name(profile, testdb.checkout_from_toplevel(toplevel), role)

    assert name == _name(role, toplevel)
    assert "gubbi-db-" not in name
    assert "-disp-pg" not in name


def test_container_name_refuses_names_other_stacks_sweep() -> None:
    profile = testdb.make_profile("gubbi-db", ["pg"], [])

    with pytest.raises(testdb.ConfigError, match="gubbi-db-"):
        testdb.container_name(profile, testdb.checkout_from_toplevel(_TOPLEVEL), "pg")


def test_up_starts_pg_with_labels_loopback_port_and_pinned_image(
    docker: FakeDocker, env_file: Path, tmp_path: Path
) -> None:
    toplevel = tmp_path / "repo"
    toplevel.mkdir()

    rc = _main(docker, env_file, ["up", "--repo", "gubbi", "--role", "pg"], toplevel=toplevel)

    runs = [call for call in docker.calls if call[1] == "run"]
    assert rc == 0
    assert runs == [
        [
            "docker", "run", "--detach", "--name", _name("pg", toplevel),
            "--label", "ai.gubbi.testdb.repo=gubbi",
            "--label", "ai.gubbi.testdb.role=pg",
            "--label", f"ai.gubbi.testdb.owner-uid={_UID}",
            "--label", f"ai.gubbi.testdb.checkout={hashlib.sha256(str(toplevel).encode()).hexdigest()}",
            "--label", f"ai.gubbi.testdb.env-sha256={_ENV_DIGEST}",
            "--publish", "127.0.0.1::5432",
            "--env", "POSTGRES_USER=journal", "--env", "POSTGRES_DB=postgres",
            "--env", "POSTGRES_PASSWORD=testpass", _PG_IMAGE,
        ]
    ]  # fmt: skip


def test_up_starts_redis_on_the_redis_pin(
    docker: FakeDocker, env_file: Path, tmp_path: Path
) -> None:
    rc = _main(docker, env_file, ["up", "--repo", "gubbi", "--role", "redis"], toplevel=tmp_path)

    (run,) = [call for call in docker.calls if call[1] == "run"]
    assert rc == 0
    assert run[-3:] == ["--publish", "127.0.0.1::6379", _REDIS_IMAGE]
    assert ["docker", "exec", docker.id_of(run[4]), "redis-cli", "ping"] in docker.calls


def test_up_probes_pg_over_tcp_and_creates_profile_databases(
    docker: FakeDocker, env_file: Path, tmp_path: Path
) -> None:
    rc = _main(docker, env_file, ["up", *_GUBBI], toplevel=tmp_path)

    pg = docker.id_of(_name("pg", tmp_path))
    execs = [call[2:] for call in docker.calls if call[1] == "exec"]
    assert rc == 0
    assert [pg, "pg_isready", "-q", "-h", "127.0.0.1", "-U", "journal", "-d", "postgres"] in execs
    assert execs[-1][-1] == 'CREATE DATABASE "journal_test"'
    assert all(call[0] == pg for call in execs if call[1] == "psql")


def test_up_writes_a_bash_sourceable_stack_env(
    docker: FakeDocker, env_file: Path, tmp_path: Path
) -> None:
    _main(docker, env_file, ["up", *_GUBBI], toplevel=tmp_path)
    port = docker.containers[_name("pg", tmp_path)]["NetworkSettings"]["Ports"]["5432/tcp"][0]

    sourced = _bash_source(tmp_path, ".testdb.env", "TESTDB_PG_URL_JOURNAL_TEST")

    assert sourced == f"postgresql://journal:testpass@127.0.0.1:{port['HostPort']}/journal_test"


def test_up_restarts_a_stopped_container_of_this_stack(
    docker: FakeDocker, env_file: Path, tmp_path: Path
) -> None:
    container_id = docker.add(_name("pg", tmp_path), _labels("pg", tmp_path), running=False)

    rc = _main(docker, env_file, ["up", "--repo", "gubbi", "--role", "pg"], toplevel=tmp_path)

    assert rc == 0
    assert ["docker", "start", container_id] in docker.calls
    assert not [call for call in docker.calls if call[1] == "run"]


def test_up_refuses_a_container_replaced_under_its_name_during_up(
    docker: FakeDocker, env_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    docker.add(_name("pg", tmp_path), _labels("pg", tmp_path), running=False)
    docker.replace_on_start = True

    rc = _main(docker, env_file, ["up", "--repo", "gubbi", "--role", "pg"], toplevel=tmp_path)

    assert rc == 1
    assert "replaced by another container" in capsys.readouterr().err
    assert not [call for call in docker.calls if call[1] == "exec"]
    assert not (tmp_path / ".testdb.env").exists()


def test_up_refuses_a_container_started_from_another_testdb_env(
    docker: FakeDocker, env_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stale = _labels("pg", tmp_path, **{"ai.gubbi.testdb.env-sha256": "0" * 64})
    docker.add(_name("pg", tmp_path), stale)

    rc = _main(docker, env_file, ["up", "--repo", "gubbi", "--role", "pg"], toplevel=tmp_path)

    assert rc == 1
    assert "stale env-sha256" in capsys.readouterr().err
    assert [call[1] for call in docker.calls] == ["container"]


def test_up_fails_after_bounded_readiness_attempts(
    docker: FakeDocker, env_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    docker.ready = False

    rc = _main(docker, env_file, ["up", "--repo", "gubbi", "--role", "pg"], toplevel=tmp_path)

    probes = [call for call in docker.calls if call[1] == "exec"]
    assert rc == 1
    assert len(probes) == testdb.READY_ATTEMPTS
    assert "not ready" in capsys.readouterr().err


def _probe(rc: int, stdout: str = "", stderr: str = "") -> Any:
    return testdb.CommandResult(rc, stdout, stderr)


@pytest.mark.parametrize(
    ("role", "not_ready"),
    [
        pytest.param("pg", _probe(1), id="pg-rejecting"),
        pytest.param("pg", _probe(2), id="pg-no-response"),
        pytest.param("pg", _probe(124, stderr="docker: timed out after 5.0s"), id="pg-timeout"),
        pytest.param("redis", _probe(0, "LOADING\n"), id="redis-loading"),
        pytest.param("redis", _probe(1, stderr="Could not connect to Redis"), id="redis-refused"),
    ],
)
def test_up_keeps_probing_while_the_service_starts(
    docker: FakeDocker, env_file: Path, tmp_path: Path, role: str, not_ready: Any
) -> None:
    docker.probe_results = [not_ready, not_ready]

    rc = _main(docker, env_file, ["up", "--repo", "gubbi", "--role", role], toplevel=tmp_path)

    probes = [call for call in docker.calls if call[1] == "exec" and call[3] != "psql"]
    assert rc == 0
    assert len(probes) == 3


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(_probe(125, stderr="docker: invalid flag"), id="docker-exit-125"),
        pytest.param(_probe(126, stderr="permission denied"), id="not-executable"),
        pytest.param(_probe(127, stderr="executable file not found"), id="not-found"),
        pytest.param(
            _probe(1, stderr="Error response from daemon: No such container: x"),
            id="no-such-container",
        ),
        pytest.param(
            _probe(1, stderr="Error response from daemon: container x is not running"),
            id="not-running",
        ),
        pytest.param(_probe(3, stderr="pg_isready: invalid option"), id="pg-usage-error"),
    ],
)
def test_up_stops_probing_on_a_docker_level_failure(
    docker: FakeDocker,
    env_file: Path,
    tmp_path: Path,
    failure: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    docker.probe_results = [failure]

    rc = _main(docker, env_file, ["up", "--repo", "gubbi", "--role", "pg"], toplevel=tmp_path)

    err = capsys.readouterr().err
    probes = [call for call in docker.calls if call[1] == "exec"]
    assert rc == 1
    assert len(probes) == 1
    assert f"docker exec {docker.id_of(_name('pg', tmp_path))} pg_isready" in err
    assert f"exit {failure.returncode}" in err
    assert failure.stderr in err


def test_up_rejects_an_invalid_testdb_env_naming_the_line(
    docker: FakeDocker, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    env_file = tmp_path / "testdb.env"
    env_file.write_bytes(b"PGVECTOR_IMAGE = pg\n")

    rc = _main(docker, env_file, ["up", "--repo", "gubbi", "--role", "pg"], toplevel=tmp_path)

    assert rc == 2
    assert "PGVECTOR_IMAGE = pg" in capsys.readouterr().err
    assert docker.calls == []


# ---------------------------------------------------------------------------
# down
# ---------------------------------------------------------------------------


def test_down_removes_this_checkouts_containers_and_stack_env(
    docker: FakeDocker, env_file: Path, tmp_path: Path
) -> None:
    _add_stack(docker, tmp_path)
    (tmp_path / ".testdb.env").write_text("X='1'\n")
    ids = [docker.id_of(_name(role, tmp_path)) for role in ("pg", "pg-disposable")]

    rc = _main(docker, env_file, ["down", *_GUBBI], toplevel=tmp_path)

    assert rc == 0
    assert docker.removed() == ids
    assert ["docker", "rm", "--force", "--volumes", ids[0]] in docker.calls
    assert not (tmp_path / ".testdb.env").exists()


@pytest.mark.parametrize(
    "labels",
    [
        pytest.param(_labels("pg", **{"ai.gubbi.testdb.owner-uid": "1"}), id="other-owner"),
        pytest.param(_labels("pg", **{"ai.gubbi.testdb.checkout": "f" * 64}), id="other-checkout"),
        pytest.param(_labels("pg", **{"ai.gubbi.testdb.repo": "other"}), id="other-repo"),
        pytest.param(_labels("pg", **{"ai.gubbi.testdb.role": "redis"}), id="other-role"),
        pytest.param(_labels("pg", **{"ai.gubbi.testdb.extra": "x"}), id="extra-label"),
        pytest.param(
            {k: v for k, v in _labels("pg").items() if not k.endswith("checkout")},
            id="missing-label",
        ),
        pytest.param({}, id="unlabelled"),
    ],
)
def test_down_refuses_container_with_foreign_labels(
    docker: FakeDocker, env_file: Path, labels: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    docker.add(_name("pg-disposable"), _labels("pg-disposable"))
    docker.add(_name("pg"), labels)

    rc = _main(docker, env_file, ["down", *_GUBBI])

    assert rc == 1
    assert "refusing to remove" in capsys.readouterr().err
    assert docker.removed() == []


def test_down_removes_a_container_with_a_stale_env_digest(
    docker: FakeDocker, env_file: Path, tmp_path: Path
) -> None:
    container_id = docker.add(
        _name("pg", tmp_path), _labels("pg", tmp_path, **{"ai.gubbi.testdb.env-sha256": "0" * 64})
    )

    rc = _main(docker, env_file, ["down", "--repo", "gubbi", "--role", "pg"], toplevel=tmp_path)

    assert rc == 0
    assert docker.removed() == [container_id]


def test_down_from_another_checkout_leaves_this_stack_running(
    docker: FakeDocker, env_file: Path, tmp_path: Path
) -> None:
    _add_stack(docker)

    rc = _main(docker, env_file, ["down", *_GUBBI], toplevel=tmp_path)

    assert rc == 0
    assert docker.removed() == []
    assert set(docker.containers) == {_name("pg"), _name("pg-disposable")}


def test_down_does_not_need_a_valid_testdb_env(docker: FakeDocker, tmp_path: Path) -> None:
    _add_stack(docker, tmp_path)
    broken = tmp_path / "testdb.env"
    broken.write_bytes(b"\x00")

    rc = _main(docker, broken, ["down", *_GUBBI], toplevel=tmp_path)

    assert rc == 0
    assert len(docker.removed()) == 2


# ---------------------------------------------------------------------------
# status / env
# ---------------------------------------------------------------------------


def test_status_require_ready_is_silent_on_a_ready_stack(
    docker: FakeDocker, env_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _add_stack(docker, tmp_path)
    _write_stack_env(docker, tmp_path)

    rc = _main(docker, env_file, ["status", *_GUBBI, "--require-ready"], toplevel=tmp_path)

    assert rc == 0
    assert capsys.readouterr() == ("", "")


def _pg(docker: FakeDocker, top: Path) -> dict[str, Any]:
    return docker.containers[_name("pg", top)]


def _stop(docker: FakeDocker, top: Path) -> None:
    _pg(docker, top)["State"]["Running"] = False


def _drop(docker: FakeDocker, top: Path) -> None:
    del docker.containers[_name("pg-disposable", top)]


def _foreign(docker: FakeDocker, top: Path) -> None:
    _pg(docker, top)["Config"]["Labels"]["ai.gubbi.testdb.owner-uid"] = "1"


def _stale(docker: FakeDocker, top: Path) -> None:
    _pg(docker, top)["Config"]["Labels"]["ai.gubbi.testdb.env-sha256"] = "0" * 64


def _unready(docker: FakeDocker, _top: Path) -> None:
    docker.ready = False


def _daemon_down(docker: FakeDocker, _top: Path) -> None:
    docker.daemon_down = True


def _public_port(docker: FakeDocker, top: Path) -> None:
    _pg(docker, top)["NetworkSettings"]["Ports"]["5432/tcp"][0]["HostIp"] = "0.0.0.0"  # noqa: S104


def _moved_port(docker: FakeDocker, top: Path) -> None:
    _pg(docker, top)["NetworkSettings"]["Ports"]["5432/tcp"][0]["HostPort"] = "49999"


def _no_stack_env(_docker: FakeDocker, top: Path) -> None:
    (top / ".testdb.env").unlink()


def _edit_stack_env(old: str, new: str) -> Any:
    def edit(_docker: FakeDocker, top: Path) -> None:
        path = top / ".testdb.env"
        path.write_text(path.read_text().replace(old, new, 1))

    return edit


@pytest.mark.parametrize(
    "breakage",
    [
        pytest.param(_drop, id="missing"),
        pytest.param(_stop, id="stopped"),
        pytest.param(_unready, id="not-ready"),
        pytest.param(_foreign, id="foreign-labels"),
        pytest.param(_stale, id="stale-env-digest"),
        pytest.param(_daemon_down, id="docker-unavailable"),
        pytest.param(_public_port, id="non-loopback-port"),
        pytest.param(_moved_port, id="port-moved-since-up"),
        pytest.param(_no_stack_env, id="stack-env-missing"),
        pytest.param(
            _edit_stack_env("TESTDB_PG_CONTAINER='testdb-", "TESTDB_PG_CONTAINER='other-"),
            id="stack-env-other-container",
        ),
        pytest.param(_edit_stack_env("TESTDB_PG_HOST=", "TESTDB_PG_HOSTX="), id="unknown-key"),
        pytest.param(_edit_stack_env("\n", "\nTESTDB_PG_PORT='1'\n"), id="duplicate-key"),
        pytest.param(_edit_stack_env("TESTDB_PG_HOST='", "TESTDB_PG_HOST="), id="unquoted"),
    ],
)
def test_status_require_ready_prints_exactly_the_fail_fast_line(
    docker: FakeDocker,
    env_file: Path,
    tmp_path: Path,
    breakage: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _add_stack(docker, tmp_path)
    _write_stack_env(docker, tmp_path)
    breakage(docker, tmp_path)

    rc = _main(docker, env_file, ["status", *_GUBBI, "--require-ready"], toplevel=tmp_path)

    assert rc == 1
    assert capsys.readouterr() == ("", "test stack not running: make test-stack-up\n")


def test_status_require_ready_never_starts_or_removes_anything(
    docker: FakeDocker, env_file: Path
) -> None:
    _main(docker, env_file, ["status", *_GUBBI, "--require-ready"])

    assert {call[1] for call in docker.calls} == {"container"}


def test_status_reports_each_role(
    docker: FakeDocker, env_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _add_stack(docker)
    _drop(docker, _TOPLEVEL)

    rc = _main(docker, env_file, ["status", *_GUBBI])

    assert rc == 1
    assert capsys.readouterr().out == (
        f"pg {_name('pg')} ready\npg-disposable {_name('pg-disposable')} missing\n"
    )


def test_env_prints_dsns_and_container_names(
    docker: FakeDocker, env_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _add_stack(docker)
    pg_port, disp_port = (_port(docker, _name(role)) for role in ("pg", "pg-disposable"))

    rc = _main(docker, env_file, ["env", *_GUBBI])

    lines = capsys.readouterr().out.splitlines()
    assert rc == 0
    assert lines[1:] == [
        f"TESTDB_PG_CONTAINER='{_name('pg')}'",
        "TESTDB_PG_HOST='127.0.0.1'",
        f"TESTDB_PG_PORT='{pg_port}'",
        f"TESTDB_PG_URL='postgresql://journal:testpass@127.0.0.1:{pg_port}/postgres'",
        f"TESTDB_PG_URL_JOURNAL_TEST='postgresql://journal:testpass@127.0.0.1:{pg_port}/journal_test'",
        f"TESTDB_PG_DISPOSABLE_CONTAINER='{_name('pg-disposable')}'",
        "TESTDB_PG_DISPOSABLE_HOST='127.0.0.1'",
        f"TESTDB_PG_DISPOSABLE_PORT='{disp_port}'",
        f"TESTDB_PG_DISPOSABLE_URL='postgresql://journal:testpass@127.0.0.1:{disp_port}/postgres'",
    ]


def test_env_refuses_a_missing_stack(docker: FakeDocker, env_file: Path) -> None:
    assert _main(docker, env_file, ["env", *_GUBBI]) == 1


# ---------------------------------------------------------------------------
# Port discovery and stack env rendering
# ---------------------------------------------------------------------------


def _binding(host_ip: str = "127.0.0.1", host_port: str = "40001") -> Any:
    return testdb.PortBinding(host_ip, host_port)


def _state(*bindings: Any) -> Any:
    return testdb.ContainerState("c" * 64, "c", {}, True, {"5432/tcp": bindings})


@pytest.mark.parametrize(
    "bindings",
    [
        pytest.param((), id="no-binding"),
        pytest.param((_binding("0.0.0.0"),), id="all-interfaces"),  # noqa: S104
        pytest.param((_binding("::1"),), id="ipv6-loopback"),
        pytest.param((_binding(host_port="5432x"),), id="non-numeric"),
        pytest.param((_binding(host_port="0"),), id="zero"),
        pytest.param((_binding(host_port="70000"),), id="out-of-range"),
        pytest.param((_binding(), _binding(host_port="40002")), id="two-bindings"),
    ],
)
def test_host_port_fails_closed(bindings: Any) -> None:
    with pytest.raises(testdb.StackError):
        testdb.host_port(_state(*bindings), "pg")


def test_host_port_fails_closed_on_an_unpublished_port() -> None:
    state = testdb.ContainerState("c" * 64, "c", {}, True, {})

    with pytest.raises(testdb.StackError):
        testdb.host_port(state, "pg")


def test_host_port_reads_the_loopback_binding() -> None:
    assert testdb.host_port(_state(_binding()), "pg") == 40001


def test_parse_inspect_reads_the_recorded_docker_sample() -> None:
    entry = _sample_entry()
    name = entry["Name"].lstrip("/")

    state = testdb.parse_inspect(name, json.dumps(_SAMPLE["container_inspect"]))

    assert state.id == entry["Id"]
    assert state.running is True
    assert state.labels["ai.gubbi.testdb.role"] == "pg"
    assert dict(state.ports) == {"5432/tcp": (testdb.PortBinding("127.0.0.1", "40001"),)}
    assert _SAMPLE["docker_port"] == "5432/tcp -> 127.0.0.1:40001\n"


def test_parse_inspect_returns_read_only_projections() -> None:
    state = testdb.parse_inspect("testdb-sample-00000000-pg", json.dumps([_sample_entry()]))

    with pytest.raises(TypeError):
        state.labels["ai.gubbi.testdb.role"] = "redis"
    with pytest.raises(TypeError):
        state.ports["6379/tcp"] = ()


def _without(path: str) -> Any:
    def mutate(entry: dict[str, Any]) -> None:
        *parents, leaf = path.split(".")
        target = entry
        for key in parents:
            target = target[key]
        del target[leaf]

    return mutate


def _setting(path: str, value: Any) -> Any:
    def mutate(entry: dict[str, Any]) -> None:
        *parents, leaf = path.split(".")
        target = entry
        for key in parents:
            target = target[key]
        target[leaf] = value

    return mutate


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(_setting("Name", "/other"), id="other-name"),
        pytest.param(_without("Id"), id="no-id"),
        pytest.param(_setting("Id", "c" * 12), id="short-id"),
        pytest.param(_setting("Id", "C" * 64), id="uppercase-id"),
        pytest.param(_setting("State.Running", "yes"), id="running-not-bool"),
        pytest.param(_setting("Config", []), id="config-not-object"),
        pytest.param(_setting("NetworkSettings.Ports", {"5432/tcp": {}}), id="bindings-object"),
        pytest.param(
            _setting(
                "NetworkSettings.Ports", {"5432/tcp": [{"HostIp": "127.0.0.1", "HostPort": 1}]}
            ),
            id="port-not-string",
        ),
    ],
)
def test_parse_inspect_fails_closed_on_one_bad_field(mutate: Any) -> None:
    entry = _sample_entry()
    mutate(entry)

    with pytest.raises(testdb.StackError):
        testdb.parse_inspect("testdb-sample-00000000-pg", json.dumps([entry]))


@pytest.mark.parametrize(
    "stdout",
    [
        pytest.param("not json", id="not-json"),
        pytest.param("[]", id="empty"),
        pytest.param(json.dumps([_SAMPLE["container_inspect"][0]] * 2), id="two-containers"),
    ],
)
def test_parse_inspect_fails_closed(stdout: str) -> None:
    with pytest.raises(testdb.StackError):
        testdb.parse_inspect("testdb-sample-00000000-pg", stdout)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("it's", id="single-quote"),
        pytest.param("a b", id="space"),
        pytest.param("a\nb", id="newline"),
        pytest.param("", id="empty"),
    ],
)
def test_render_stack_env_refuses_values_a_shell_would_split(value: str) -> None:
    with pytest.raises(testdb.StackError):
        testdb.render_stack_env([("KEY", value)])


def test_render_stack_env_keeps_shell_metacharacters_literal(tmp_path: Path) -> None:
    value = "$HOME`id`$(id)\\!*"
    (tmp_path / "e").write_text(testdb.render_stack_env([("KEY", value)]))

    assert _bash_source(tmp_path, "e", "KEY") == value


# ---------------------------------------------------------------------------
# Profile validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("repo", "roles", "dbs"),
    [
        pytest.param("Gubbi", ["pg"], [], id="uppercase-repo"),
        pytest.param("gubbi", [], [], id="no-role"),
        pytest.param("gubbi", ["mysql"], [], id="unknown-role"),
        pytest.param("gubbi", ["pg-"], [], id="empty-suffix"),
        pytest.param("gubbi", ["pg", "pg"], [], id="duplicate-role"),
        pytest.param("gubbi", ["pg"], ["journal-test"], id="db-with-dash"),
        pytest.param("gubbi", ["pg"], ["x'; drop"], id="db-with-quote"),
        pytest.param("gubbi", ["pg-disposable"], ["journal"], id="db-without-pg-role"),
    ],
)
def test_make_profile_rejects(repo: str, roles: list[str], dbs: list[str]) -> None:
    with pytest.raises(testdb.ConfigError):
        testdb.make_profile(repo, roles, dbs)


def test_usage_error_exits_2(docker: FakeDocker, env_file: Path) -> None:
    assert _main(docker, env_file, ["up", "--repo", "gubbi", "--role", "mysql"]) == 2
    assert docker.calls == []


# ---------------------------------------------------------------------------
# Image pins
# ---------------------------------------------------------------------------


def _pins(image: str) -> Any:
    return testdb.Pins(values={"PGVECTOR_IMAGE": image}, digest="0" * 64)


@pytest.mark.parametrize(
    "image",
    [
        pytest.param(_PG_IMAGE, id="repo-digest"),
        pytest.param(_REDIS_IMAGE, id="repo-tag-digest"),
        pytest.param("registry.example.com/team/pg:17.1_x@sha256:" + "0" * 64, id="registry"),
    ],
)
def test_image_pin_accepts_a_digest_reference(image: str) -> None:
    assert _pins(image).image("PGVECTOR_IMAGE") == image


@pytest.mark.parametrize(
    "image",
    [
        pytest.param("-pgvector@sha256:" + "a" * 64, id="leading-dash"),
        pytest.param("--privileged", id="docker-flag"),
        pytest.param("pgvector/pgvector:pg17", id="no-digest"),
        pytest.param("pgvector/pgvector@sha256:" + "A" * 64, id="uppercase-digest"),
        pytest.param("pgvector/pgvector@sha256:" + "a" * 63, id="short-digest"),
        pytest.param("PGVector@sha256:" + "a" * 64, id="uppercase-repo"),
    ],
)
def test_image_pin_rejects(image: str) -> None:
    with pytest.raises(testdb.ConfigError, match="must match"):
        _pins(image).image("PGVECTOR_IMAGE")


def test_the_real_pins_are_digest_references() -> None:
    pins = testdb.load_pins(testdb.TESTDB_ENV)

    assert pins.image("PGVECTOR_IMAGE")
    assert pins.image("REDIS_IMAGE")


def test_load_pins_returns_read_only_values(env_file: Path) -> None:
    pins = testdb.load_pins(env_file)

    with pytest.raises(TypeError):
        pins.values["PGVECTOR_IMAGE"] = "-x"


# ---------------------------------------------------------------------------
# check-env
# ---------------------------------------------------------------------------


def test_check_env_prints_validated_pairs(
    env_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = testdb.main(["check-env", "--file", str(env_file)])

    assert rc == 0
    assert capsys.readouterr().out == _ENV_BYTES.decode()


def test_check_env_appends_to_github_output(env_file: Path, tmp_path: Path) -> None:
    output = tmp_path / "github_output"
    output.write_text("EARLIER=1\n")

    rc = testdb.main(["check-env", "--file", str(env_file), "--github-output", str(output)])

    assert rc == 0
    assert output.read_bytes() == b"EARLIER=1\n" + _ENV_BYTES


@pytest.mark.parametrize(
    ("raw", "named"),
    [
        pytest.param(b"PG_MAJOR=1\x007\n", "PG_MAJOR=1\\x007", id="nul"),
        pytest.param(b"PG_MAJOR=17", "missing final newline", id="unterminated"),
        pytest.param(b"PG_MAJOR=17\nPG_MAJOR=18\n", "repeats keys", id="duplicate"),
    ],
)
def test_check_env_exits_2_naming_the_violation(
    tmp_path: Path, raw: bytes, named: str, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "testdb.env"
    bad.write_bytes(raw)
    output = tmp_path / "github_output"

    rc = testdb.main(["check-env", "--file", str(bad), "--github-output", str(output)])

    assert rc == 2
    assert named in capsys.readouterr().err
    assert not output.exists()


def test_check_env_on_the_real_file_prints_exactly_the_three_keys() -> None:
    result = _run_tool(str(TESTDB_TOOL), "check-env")

    assert result.returncode == 0, result.stderr
    keys = [line.split("=", 1)[0] for line in result.stdout.splitlines()]
    assert keys == ["PGVECTOR_IMAGE", "PG_MAJOR", "REDIS_IMAGE"]
