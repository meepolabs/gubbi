"""Dump the FastAPI OpenAPI schema to stdout as deterministic JSON.

This script is the source of truth for the typed API client generated in
``gubbi-web/packages/api-types``. It imports the application object and
serialises ``app.openapi()`` with stable key ordering and ASCII-only output
so the result is byte-for-byte reproducible across machines and CI runs.

The FastAPI ``app`` is constructed at import time; its lifespan and settings
are lazy, so this runs with no database, cache, environment, or running
backend. Exit code is 0 on success.

Usage::

    python scripts/dump_openapi.py > tests/fixtures/openapi_snapshot.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    """Serialise the app's OpenAPI schema to stdout and return an exit code."""
    # The package is run from a source checkout (poetry package-mode is off), so
    # the repository root -- not this script's directory -- must be on sys.path
    # for ``import gubbi`` to resolve when invoked as ``scripts/dump_openapi.py``.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from gubbi.main import app

    json.dump(app.openapi(), sys.stdout, sort_keys=True, indent=2, ensure_ascii=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
