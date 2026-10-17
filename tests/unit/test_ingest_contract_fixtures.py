"""Ingest wire-contract fixture test.

The extension's Zod schema and the testbench's live run pin byte copies of
``tests/fixtures/ingest_contract.json``. This module keeps that file honest on
the server side: it must equal the generator's output, every row's declared
verdict must be the Pydantic request model's verdict, and the rows together
must exercise every field of the request schema.

Validation runs ``model_validate`` on ``json.loads`` output, the path FastAPI
takes for a request body. No database, cache, or environment is needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from gubbi.api.v1.ingest import IngestConversationRequest
from scripts.dump_ingest_fixtures import build_document

pytestmark = pytest.mark.unit

FIXTURE_PATH = Path(__file__).parent.parent / "fixtures" / "ingest_contract.json"

_REGEN_INSTRUCTIONS = (
    "The ingest contract generator has drifted from the committed fixture.\n"
    "If the contract change is intentional (additive only), regenerate and re-pin:\n"
    "  1. In gubbi: PYTHONPATH=. poetry run python scripts/dump_ingest_fixtures.py > "
    "tests/fixtures/ingest_contract.json\n"
    "  2. Copy the bytes to gubbi-extension tests/fixtures/ingest-contract.json and\n"
    "     gubbi-testbench tests/fixtures/ingest_contract.json, and update the sha256\n"
    "     pinned by each copy's test."
)


def _rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = json.loads(FIXTURE_PATH.read_text(encoding="ascii"))["rows"]
    return rows


def _schema_field_paths(schema: dict[str, Any], defs: dict[str, Any], prefix: str) -> set[str]:
    paths: set[str] = set()
    for field, spec in schema["properties"].items():
        path = prefix + field
        paths.add(path)
        items = spec.get("items")
        if items is not None and "$ref" in items:
            target = defs[items["$ref"].rsplit("/", 1)[-1]]
            paths |= _schema_field_paths(target, defs, f"{path}[].")
    return paths


def test_committed_fixture_matches_generator() -> None:
    live = json.dumps(build_document(), sort_keys=True, indent="\t", ensure_ascii=True) + "\n"
    committed = FIXTURE_PATH.read_text(encoding="ascii")

    assert live == committed, _REGEN_INSTRUCTIONS


@pytest.mark.parametrize("row", [pytest.param(row, id=row["name"]) for row in _rows()])
def test_row_verdict_matches_request_model(row: dict[str, Any]) -> None:
    try:
        IngestConversationRequest.model_validate(row["payload"])
        rejection = None
    except ValidationError as exc:
        rejection = str(exc)

    assert (rejection is None) == row["valid"], (
        f"row {row['name']!r} declared valid={row['valid']}, model rejection: {rejection}"
    )


def test_covers_equal_every_request_schema_field_path() -> None:
    schema = IngestConversationRequest.model_json_schema()
    expected = _schema_field_paths(schema, schema.get("$defs", {}), "")
    rows = _rows()

    unknown = {
        (row["name"], path) for row in rows for path in row["covers"] if path not in expected
    }
    covered = {path for row in rows for path in row["covers"]}

    assert unknown == set()
    assert covered == expected


def test_rows_are_uniquely_named_and_cover_both_verdicts() -> None:
    rows = _rows()
    names = [row["name"] for row in rows]

    assert len(names) == len(set(names))
    assert any(row["valid"] for row in rows)
    assert any(not row["valid"] for row in rows)


def test_lone_surrogates_survive_the_json_round_trip() -> None:
    by_name = {row["name"]: row for row in _rows()}

    content = by_name["content.lone_surrogate"]["payload"]["conversations"][0]["messages"][0][
        "content"
    ]
    message_id = by_name["message_id.lone_high_surrogate"]["payload"]["conversations"][0][
        "messages"
    ][0]["message_id"]

    assert "\ud800" in content
    assert message_id == "id-\ud800"
