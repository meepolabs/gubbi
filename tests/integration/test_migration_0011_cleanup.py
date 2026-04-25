"""Tests for migration 0011 XML spill cleanup logic.

Uses a mock bind to verify row-level cleanup behavior without a live DB.
"""

import json
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit


def _make_row(entry_id: int, reasoning: str, tags: object = None) -> MagicMock:
    """Create a mock row with id, reasoning, and tags attributes."""
    row = MagicMock()
    row.id = entry_id
    row.reasoning = reasoning
    row.tags = tags
    return row


class TestMigration0011Cleanup:
    def test_no_matching_rows_is_noop(self) -> None:
        from journalctl.alembic.versions.v20260424_0011_cleanup_xml_spill import _clean_rows

        bind = MagicMock()
        _clean_rows(bind, [])
        # No execute calls should happen when there are zero rows to clean.
        bind.execute.assert_not_called()

    def test_truncates_reasoning_at_first_parameter(self) -> None:
        from journalctl.alembic.versions.v20260424_0011_cleanup_xml_spill import _clean_rows

        reasoning = 'Good reasoning.\n<parameter name="tags">["health"]</parameter>'
        row = _make_row(42, reasoning)
        bind = MagicMock()
        _clean_rows(bind, [row])
        # First execute call is the UPDATE; params are positional arg 1.
        call = bind.execute.call_args_list[0]
        params = call.args[1]
        assert params["r"] == "Good reasoning."

    def test_recovers_tags_from_xml(self) -> None:
        from journalctl.alembic.versions.v20260424_0011_cleanup_xml_spill import _clean_rows

        reasoning = 'Decision made.\n<parameter name="tags">["health", "work"]</parameter>'
        row = _make_row(99, reasoning, tags=None)
        bind = MagicMock()
        _clean_rows(bind, [row])
        call = bind.execute.call_args_list[0]
        params = call.args[1]
        assert params["t"] == ["health", "work"]

    def test_audit_log_insert_contains_before_after(self) -> None:
        from journalctl.alembic.versions.v20260424_0011_cleanup_xml_spill import _clean_rows

        reasoning = 'Note.\n<parameter name="tags">["a"]</parameter>'
        row = _make_row(7, reasoning)
        bind = MagicMock()
        _clean_rows(bind, [row])
        audit_call = bind.execute.call_args_list[1]
        params = audit_call.args[1]
        meta = json.loads(params["meta"])
        assert "before" in meta
        assert "after" in meta

    def test_empty_tags_not_changed(self) -> None:
        from journalctl.alembic.versions.v20260424_0011_cleanup_xml_spill import _clean_rows

        # tags param exists but its value is not a JSON list
        reasoning = 'Thought.\n<parameter name="tags">not-a-list</parameter>'
        row = _make_row(1, reasoning, tags=["existing"])
        bind = MagicMock()
        _clean_rows(bind, [row])
        call = bind.execute.call_args_list[0]
        params = call.args[1]
        # tags should remain unchanged since recovery failed (not-a-list JSON)
        assert params["t"] == ["existing"]

    def test_reasoning_without_parameter_is_treated_as_no_match(self) -> None:
        from journalctl.alembic.versions.v20260424_0011_cleanup_xml_spill import _clean_rows

        reasoning = "Clean reasoning with no XML fragments."
        row = _make_row(5, reasoning)
        bind = MagicMock()
        _clean_rows(bind, [row])
        # No <parameter found -> early return, no execute calls.
        bind.execute.assert_not_called()
