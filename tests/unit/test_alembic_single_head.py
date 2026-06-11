"""Regression guard against future alembic forks.

The chain was squashed on 2026-05-24 into a single mechanically-derived
baseline (``0001_squashed_baseline``); the prior 0001-0031 chain lives
under ``gubbi/alembic/_archive/`` and is not loaded by alembic.

This test asserts the chain stays single-headed and pinned to the
current head. If a future migration is added off the wrong parent (or
a duplicate revision number is introduced), ``ScriptDirectory.get_heads()``
will return more than one head -- or a different head name -- and this
test will fail before the fork can land. When a new migration is
intentionally added, update ``_EXPECTED_HEAD`` to its revision id.

No database is needed: alembic walks the version directory purely from
file metadata.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory

pytestmark = pytest.mark.unit


_EXPECTED_HEAD = "0002_add_onboarding_completed_at"


def _alembic_config() -> Config:
    """Load the project's alembic.ini relative to the test file."""
    repo_root = Path(__file__).resolve().parents[2]
    ini_path = repo_root / "alembic.ini"
    return Config(str(ini_path))


def test_alembic_has_single_head() -> None:
    # Arrange
    cfg = _alembic_config()
    script_dir = ScriptDirectory.from_config(cfg)

    # Act
    heads = script_dir.get_heads()

    # Assert
    assert len(heads) == 1, f"expected exactly one alembic head, got: {heads!r}"


def test_alembic_head_is_expected() -> None:
    # Arrange
    cfg = _alembic_config()
    script_dir = ScriptDirectory.from_config(cfg)

    # Act
    heads = script_dir.get_heads()

    # Assert
    assert heads == [_EXPECTED_HEAD], f"expected head {_EXPECTED_HEAD!r}, got: {heads!r}"
