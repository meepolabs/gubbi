"""OpenAPI export contract tests.

These tests guard the typed-client pipeline that feeds
``gubbi-web/packages/api-types``:

    - The application object must import and produce its OpenAPI schema with no
      database, cache, or environment available. The schema is generated at
      build time (and in CI) where none of those exist.
    - The committed snapshot must stay in lockstep with the live schema, so any
      API change is forced through a deliberate snapshot regeneration.

The module imports ``gubbi.main`` at top level and requests no fixtures, so it
exercises the import path without touching the heavyweight DB/Redis fixtures in
``tests/conftest.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# Top-level import: importing the app must not require a database, cache, or
# environment. If this line ever needs them, the failure surfaces at collection
# time -- which is the contract under test.
from gubbi.main import app

pytestmark = pytest.mark.unit

SNAPSHOT_PATH = Path(__file__).parent / "fixtures" / "openapi_snapshot.json"

_REGEN_INSTRUCTIONS = (
    "The live OpenAPI schema has drifted from the committed snapshot.\n"
    "If this change to the API is intentional, regenerate both artifacts:\n"
    "  1. In gubbi: python scripts/dump_openapi.py > "
    "tests/fixtures/openapi_snapshot.json\n"
    "  2. In gubbi-web: pnpm --filter @gubbi/api-types generate\n"
    "Then commit the refreshed snapshot together with the API change."
)


def test_app_openapi_schema_shape() -> None:
    """The exported schema has the expected OpenAPI 3.x shape and title.

    The top-level ``from gubbi.main import app`` is the primary guard: it fires
    at collection time, so importing the app and building its schema cannot
    require a database, cache, or environment. The autouse env fixture DOES run
    during this test, so this additionally confirms schema generation completes
    without opening any DB or cache connection -- it does not prove the env is
    absent.
    """
    schema = app.openapi()

    assert isinstance(schema, dict)
    assert schema["openapi"].startswith("3.")
    assert schema["info"]["title"] == "gubbi"


def test_app_openapi_has_paths() -> None:
    """The exported schema exposes at least one path."""
    assert app.openapi()["paths"], "expected at least one path in the schema"


def test_openapi_snapshot_has_no_servers_block() -> None:
    """The exported schema must not pin an internal server URL.

    A ``servers`` block would leak deploy hosts into a public artifact and the
    generated client; FastAPI omits it by default and we assert that holds.
    """
    assert "servers" not in app.openapi()


def test_openapi_snapshot_matches_live_schema() -> None:
    """Committed snapshot must equal the freshly generated schema, byte-for-byte.

    The snapshot is serialised with the same deterministic options as
    ``scripts/dump_openapi.py`` (sorted keys, two-space indent, ASCII-only) so a
    match here means the script would reproduce the committed file exactly.
    """
    live = json.dumps(app.openapi(), sort_keys=True, indent=2, ensure_ascii=True) + "\n"
    committed = SNAPSHOT_PATH.read_text(encoding="ascii")

    assert live == committed, _REGEN_INSTRUCTIONS
