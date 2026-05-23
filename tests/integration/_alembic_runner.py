"""Shared subprocess alembic runner for integration tests.

Used by per-migration round-trip tests that need to invoke ``alembic upgrade``
or ``alembic downgrade`` with a specific bootstrap DSN, isolated from
the test session's global alembic state.

Mirrors the same env-var / project-root / S603 conventions as the
canonical helper in ``test_audit_log_cross_attribution_guards.py`` so the
two patterns can converge on a single import in a future cleanup.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def run_alembic(bootstrap_dsn: str, *args: str) -> subprocess.CompletedProcess[str]:
    """Invoke alembic as a subprocess against ``bootstrap_dsn``.

    Sets ``JOURNAL_DB_MIGRATION_URL`` and ``JOURNAL_OPERATOR_EMAIL`` in the
    subprocess env so global alembic state cannot leak into the test
    session. ``cwd`` resolves to the gubbi repo root (the parents[2] of
    this file) so alembic finds its config.
    """
    project_root = Path(__file__).resolve().parents[2]
    env = {
        **os.environ,
        "JOURNAL_DB_MIGRATION_URL": bootstrap_dsn,
        "JOURNAL_OPERATOR_EMAIL": "operator@test.local",
    }
    return subprocess.run(  # noqa: S603 -- args are literals, no shell
        [sys.executable, "-m", "alembic", *args],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
