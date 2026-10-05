"""Contract test: tools/testdb/testdb.env obeys its strict ASCII KEY=value format.

Every consumer (CI config jobs, the local stack controller, sibling repos) reads
this file with the same line rule rather than a parser, so a file that breaks the
format must fail here before it fails in a consumer.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TESTDB_ENV = _REPO_ROOT / "tools" / "testdb" / "testdb.env"
_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "security-tests.yml"

_LINE_RULE = re.compile(r"[A-Z][A-Z0-9_]*=[!-~]+", re.ASCII)
_CONTROL_CHAR = re.compile(r"[\x00-\x09\x0b-\x1f\x7f]", re.ASCII)
_REQUIRED_KEYS = ("PGVECTOR_IMAGE", "PG_MAJOR", "REDIS_IMAGE")
_PGVECTOR_SERVICES = ("postgres", "postgres_disposable")


def _read_testdb_env(path: Path) -> bytes:
    # Raw bytes: read_text() would translate CRLF to LF before validation.
    return path.read_bytes()


def _is_ignorable(line: str) -> bool:
    return line == "" or line.startswith("#")


def _lines(text: str) -> list[str]:
    # Split on LF only: splitlines() would also eat CR, VT, FF and other
    # separators, hiding bytes the shell consumers do not tolerate.
    return text.removesuffix("\n").split("\n")


def _is_valid_line(line: str) -> bool:
    if _CONTROL_CHAR.search(line):
        return False
    return _is_ignorable(line) or _LINE_RULE.fullmatch(line) is not None


def _violations(raw: bytes) -> list[str]:
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        return [f"non-ASCII byte {raw[exc.start : exc.start + 1]!r} at offset {exc.start}"]
    bad_lines = [line for line in _lines(text) if not _is_valid_line(line)]
    # A line-by-line shell reader drops an unterminated last line without error.
    if raw and not raw.endswith(b"\n"):
        return [*bad_lines, "missing final newline"]
    return bad_lines


def _parse(raw: bytes) -> dict[str, str]:
    bad = _violations(raw)
    if bad:
        raise ValueError(f"testdb.env breaks the KEY=value format: {bad!r}")
    pairs = [line.split("=", 1) for line in _lines(raw.decode("ascii")) if not _is_ignorable(line)]
    keys = [key for key, _ in pairs]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ValueError(f"testdb.env repeats keys: {duplicates!r}")
    return dict(pairs)


def test_real_file_obeys_line_rule() -> None:
    raw = _read_testdb_env(_TESTDB_ENV)

    assert _violations(raw) == []


def test_real_file_declares_required_keys() -> None:
    values = _parse(_read_testdb_env(_TESTDB_ENV))

    assert set(_REQUIRED_KEYS) <= values.keys()
    assert values["PG_MAJOR"] == "17"


def test_pgvector_image_matches_every_ci_service() -> None:
    values = _parse(_read_testdb_env(_TESTDB_ENV))
    workflow = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    services = workflow["jobs"]["security-tests"]["services"]

    images = {name: services[name]["image"] for name in _PGVECTOR_SERVICES}

    assert images == dict.fromkeys(_PGVECTOR_SERVICES, values["PGVECTOR_IMAGE"])


def test_reader_rejects_crlf_file(tmp_path: Path) -> None:
    env_file = tmp_path / "testdb.env"
    env_file.write_bytes(b"# pins\r\nPG_MAJOR=17\r\n")

    assert _violations(_read_testdb_env(env_file)) == ["# pins\r", "PG_MAJOR=17\r"]


@pytest.mark.parametrize(
    "line",
    [
        pytest.param("PG_MAJOR=17", id="plain"),
        pytest.param("REDIS_IMAGE=redis:7-alpine@sha256:abc", id="colon-and-at"),
        pytest.param("A1_B=x", id="digit-and-underscore-in-key"),
        pytest.param("# PG_MAJOR = 17", id="comment"),
        pytest.param("", id="blank"),
    ],
)
def test_line_rule_accepts(line: str) -> None:
    assert _violations(f"{line}\n".encode("ascii")) == []


def test_format_accepts_empty_file() -> None:
    assert _violations(b"") == []


@pytest.mark.parametrize(
    "line",
    [
        pytest.param("PGVECTOR_IMAGE = x", id="spaces-around-equals"),
        pytest.param("PG_MAJOR =17", id="space-before-equals"),
        pytest.param("PG_MAJOR= 17", id="space-after-equals"),
        pytest.param("pg_major=17", id="lowercase-key"),
        pytest.param("1PG=17", id="key-starts-with-digit"),
        pytest.param("PG_MAJOR=", id="empty-value"),
        pytest.param("PG_MAJOR: 17", id="yaml-shaped"),
        pytest.param("PG_MAJOR=17 # trailing", id="trailing-comment"),
        pytest.param("export PG_MAJOR=17", id="shell-export"),
        pytest.param("  # indented comment", id="indented-comment"),
        pytest.param("   ", id="whitespace-only"),
        pytest.param("PG_MAJOR=17\r", id="crlf"),
        pytest.param("PG_MAJOR=1\v7", id="vertical-tab"),
        pytest.param("PG_MAJOR=1\f7", id="form-feed"),
        pytest.param("PG_MAJOR=1\x007", id="nul"),
        pytest.param("PG_MAJOR=1\x7f7", id="del"),
        pytest.param("# note\x00", id="nul-in-comment"),
        pytest.param("# note\t", id="tab-in-comment"),
    ],
)
def test_line_rule_rejects(line: str) -> None:
    assert _violations(f"{line}\n".encode("ascii")) == [line]


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(b"PG_MAJOR=17", id="single-unterminated-line"),
        pytest.param(b"PG_MAJOR=17\nREDIS_IMAGE=x", id="unterminated-final-line"),
    ],
)
def test_format_rejects_missing_final_newline(raw: bytes) -> None:
    assert _violations(raw) == ["missing final newline"]


@pytest.mark.parametrize(
    ("raw", "offset"),
    [
        pytest.param("PG_MAJOR=1 7".encode(), 10, id="line-separator-in-value"),
        pytest.param("PG_MAJOR=1\u00857".encode(), 10, id="next-line-in-value"),
        pytest.param("# café".encode(), 5, id="non-ascii-in-comment"),
    ],
)
def test_format_rejects_non_ascii(raw: bytes, offset: int) -> None:
    assert _violations(raw) == [f"non-ASCII byte {raw[offset : offset + 1]!r} at offset {offset}"]


def test_parse_rejects_duplicate_keys() -> None:
    with pytest.raises(ValueError, match="repeats keys"):
        _parse(b"PG_MAJOR=17\nPG_MAJOR=18\n")
