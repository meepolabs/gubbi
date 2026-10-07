"""Every psql call in deployment/scripts/verify-db-invariants.sh skips ~/.psqlrc.

A user's ``.psqlrc`` can change psql output (``\\pset``, ``\\timing``, echo
settings), and the verifier compares that output against literals, so each
invocation, and the remediation command it prints, carries ``-X``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_SCRIPT = Path(__file__).resolve().parents[2] / "deployment" / "scripts" / "verify-db-invariants.sh"
_PSQL_WORD = re.compile(r"(?<![\w./-])psql(?![\w.-])")
_PSQL_NO_RC = re.compile(r"(?<![\w./-])psql\s+-X\s")


def _psql_lines() -> list[str]:
    lines = _SCRIPT.read_text(encoding="utf-8").splitlines()
    return [line for line in lines if not line.lstrip().startswith("#") and _PSQL_WORD.search(line)]


def test_the_psql_scan_finds_the_query_helper_and_the_printed_remedy() -> None:
    lines = _psql_lines()

    assert any('-tAc "$1" "${DB_URL}"' in line for line in lines)
    assert any("-f deployment/scripts/grants.sql" in line for line in lines)


def test_every_psql_invocation_in_the_verifier_passes_no_psqlrc() -> None:
    missing = [line for line in _psql_lines() if not _PSQL_NO_RC.search(line)]

    assert missing == []
