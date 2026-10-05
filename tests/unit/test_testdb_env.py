"""Contract test: tools/testdb/testdb.env obeys the strict KEY=value line rule.

Every consumer (CI config jobs, the local stack controller, sibling repos) reads
this file with the same line rule rather than a parser, so a line that breaks the
rule must fail here before it fails in a consumer.
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

_LINE_RULE = re.compile(r"^[A-Z][A-Z0-9_]*=[^\s]+$")
_REQUIRED_KEYS = ("PGVECTOR_IMAGE", "PG_MAJOR", "REDIS_IMAGE")
_PGVECTOR_SERVICES = ("postgres", "postgres_disposable")


def _is_ignorable(line: str) -> bool:
    return line == "" or line.startswith("#")


def _violations(text: str) -> list[str]:
    return [
        line for line in text.splitlines() if not _is_ignorable(line) and not _LINE_RULE.match(line)
    ]


def _parse(text: str) -> dict[str, str]:
    bad = _violations(text)
    if bad:
        raise ValueError(f"testdb.env lines break the KEY=value rule: {bad!r}")
    pairs = [line.split("=", 1) for line in text.splitlines() if not _is_ignorable(line)]
    keys = [key for key, _ in pairs]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ValueError(f"testdb.env repeats keys: {duplicates!r}")
    return dict(pairs)


def test_real_file_obeys_line_rule() -> None:
    text = _TESTDB_ENV.read_text(encoding="utf-8")

    assert _violations(text) == []


def test_real_file_declares_required_keys() -> None:
    values = _parse(_TESTDB_ENV.read_text(encoding="utf-8"))

    assert set(_REQUIRED_KEYS) <= values.keys()
    assert values["PG_MAJOR"] == "17"


def test_pgvector_image_matches_every_ci_service() -> None:
    values = _parse(_TESTDB_ENV.read_text(encoding="utf-8"))
    workflow = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    services = workflow["jobs"]["security-tests"]["services"]

    images = {name: services[name]["image"] for name in _PGVECTOR_SERVICES}

    assert images == dict.fromkeys(_PGVECTOR_SERVICES, values["PGVECTOR_IMAGE"])


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
    assert _violations(line) == []


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
    ],
)
def test_line_rule_rejects(line: str) -> None:
    assert _violations(line) == [line]


def test_parse_rejects_duplicate_keys() -> None:
    with pytest.raises(ValueError, match="repeats keys"):
        _parse("PG_MAJOR=17\nPG_MAJOR=18\n")
