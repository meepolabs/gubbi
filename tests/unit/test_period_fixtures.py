"""Period-resolution fixture contract test.

The journal web client re-implements ``resolve_period`` in TypeScript and pins
itself to a fixture generated from this resolver. Without this test that pin is
one-sided: the web suite proves the web has not drifted from its snapshot, not
that the snapshot still matches Python. This module closes that half -- a change
to the resolver's output fails here and forces a deliberate regeneration of both
copies.

The resolver is pure, so this module requests no fixtures and needs no database,
cache, or environment.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.dump_period_fixtures import build_document

pytestmark = pytest.mark.unit

FIXTURE_PATH = Path(__file__).parent.parent / "fixtures" / "period_resolution.json"

_REGEN_INSTRUCTIONS = (
    "The live period resolver has drifted from the committed fixture.\n"
    "If this change to period resolution is intentional, regenerate both copies\n"
    "from this repo's generator -- they must stay byte-identical:\n"
    "  1. In gubbi: python scripts/dump_period_fixtures.py > "
    "tests/fixtures/period_resolution.json\n"
    "  2. In gubbi-web: python ../gubbi/scripts/dump_period_fixtures.py > "
    "apps/journal/src/lib/timeline/period-fixtures.json\n"
    "Then commit both refreshed fixtures together with the resolver change."
)


def test_committed_fixture_matches_live_resolver() -> None:
    """Committed fixture must equal freshly generated output, byte-for-byte.

    Serialised with the same deterministic options as
    ``scripts/dump_period_fixtures.py`` (sorted keys, tab indent, ASCII-only) so
    a match here means the script would reproduce the committed file exactly.
    """
    live = json.dumps(build_document(), sort_keys=True, indent="\t", ensure_ascii=True) + "\n"
    committed = FIXTURE_PATH.read_text(encoding="ascii")

    assert live == committed, _REGEN_INSTRUCTIONS


def test_fixture_covers_both_accepted_and_rejected_inputs() -> None:
    """The matrix asserts in both directions.

    A fixture of only-rejected or only-accepted rows would let the resolver's
    accept/reject boundary move in one direction unnoticed.
    """
    rows = json.loads(FIXTURE_PATH.read_text(encoding="ascii"))["rows"]

    assert any(row["rejected"] for row in rows)
    assert any(not row["rejected"] for row in rows)


def test_fixture_excludes_inputs_the_resolver_raises_on() -> None:
    """The two known-defective inputs stay out of the fixture.

    Pinning them would enshrine the defect as contract; their absence means a
    later fix surfaces as a regeneration crash instead.
    """
    rows = json.loads(FIXTURE_PATH.read_text(encoding="ascii"))["rows"]
    periods = {row["period"] for row in rows}

    assert "9999-12" not in periods
    assert "9999-W52" not in periods
