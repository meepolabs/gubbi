"""Regression guard against future alembic forks.

The migration directory has forked once already (two ``0020_*`` revisions
both pointing at ``0019_rls_users`` as ``down_revision``); the orphan was
recovered as ``0028_audit_log_cross_attribution_guard``. This unit test
asserts that the chain stays single-headed and pinned to that specific
revision. If a future migration is added off the wrong parent (or a
duplicate revision number is introduced), ``ScriptDirectory.get_heads()``
will return more than one head -- or a different head name -- and this
test will fail before the fork can land.

No database is needed: alembic walks the version directory purely from
file metadata.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory

pytestmark = pytest.mark.unit


_EXPECTED_HEAD = "0029_audit_log_target_kind_check"


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


def test_alembic_head_is_audit_log_target_kind_check() -> None:
    # Arrange
    cfg = _alembic_config()
    script_dir = ScriptDirectory.from_config(cfg)

    # Act
    heads = script_dir.get_heads()

    # Assert
    assert heads == [_EXPECTED_HEAD], f"expected head {_EXPECTED_HEAD!r}, got: {heads!r}"
