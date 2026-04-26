"""Migration 0011 no-op stub still runs cleanly.

Migration 0008 dropped ``entries.reasoning`` (plaintext), so migration 0011
was rewritten as a no-op stub that only logs. This test confirms the
upgrade function executes without raising. Real cleanup of encrypted XML
spill happens via ``deployment/cleanup_encrypted_xml_spill.py``.
"""

import pytest


@pytest.mark.integration
def test_migration_0011_noop_runs_cleanly() -> None:
    """Upgrade is a logger-only no-op; should not raise."""
    from journalctl.alembic.versions.v20260424_0011_cleanup_xml_spill import (
        upgrade,
    )

    upgrade()  # Should not raise
