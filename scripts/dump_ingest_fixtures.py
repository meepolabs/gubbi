"""Dump ingest wire-contract fixtures to stdout as deterministic JSON.

The browser extension re-implements the ``POST /api/v1/ingest/conversations``
request schema in Zod, and the testbench posts requests to a running stack.
This script emits one row per probed input, ``{name, covers, payload,
valid}``, so all three can pin the same bytes: this repo commits a copy under
``tests/fixtures/`` and checks every verdict against the Pydantic request
model; the extension and the testbench each commit a byte copy and pin its
sha256.

``valid`` is declared here by hand, never computed from the model, so the
check against Pydantic is a real comparison. ``covers`` names the request
field paths each row exercises; the test requires their union to equal every
field path of the request schema.

The 25 MiB request body cap is deliberately absent: it is a transport limit
enforced on raw bytes before the body is parsed, not a schema verdict, and a
row large enough to probe it would make every copy of the fixture 25 MiB. The
per-request conversation count is a schema bound and is probed at its limit
and one past it.

Rows are sorted by name and the output is ASCII-only with tab indentation.
With ``ensure_ascii`` a lone surrogate is written as a ``\\udXXX`` escape,
which both ``json.loads`` and JavaScript's ``JSON.parse`` read back as the
same single lone code unit. Exit code is 0 on success.

Usage::

    PYTHONPATH=. python scripts/dump_ingest_fixtures.py > tests/fixtures/ingest_contract.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    _RowBuilder = Callable[..., dict[str, Any]]

FIXTURE_COMMENT: list[str] = [
    "GENERATED -- do not hand-edit. Each row is a request body for",
    "POST /api/v1/ingest/conversations and the verdict the request schema must reach:",
    "valid=true is accepted (200), valid=false is rejected (422).",
    "",
    "Three copies of these bytes exist and all are pinned: gubbi checks every verdict",
    "against its Pydantic request model and diffs this file against its generator;",
    "gubbi-extension (tests/fixtures/ingest-contract.json) checks its Zod schema;",
    "gubbi-testbench (tests/fixtures/ingest_contract.json) posts each row to a stack.",
    "To change the contract: regenerate from gubbi/scripts/dump_ingest_fixtures.py,",
    "re-copy the bytes into both siblings, and update each copy's sha256 pin.",
    "Changes are additive only: never flip an existing row's verdict.",
    "",
    "The 25 MiB request body cap (413) is a transport limit and is not probed here.",
]

_ABSENT = object()

_SOURCE = "source"
_CONVERSATIONS = "conversations"
_CONV = "conversations[]."
_MSG = "conversations[].messages[]."

_SOURCE_ID_MAX = 512
_BMP_CHAR = "é"
_ASTRAL_CHAR = "\U0001f600"
_LONE_HIGH = "\ud800"
_LONE_LOW = "\udc00"

_BASE_TIME = "2026-03-01T17:00:00Z"

# Same accept and reject forms the timestamp grammar's own unit tests probe.
_TIMESTAMP_ACCEPTED: tuple[tuple[str, str], ...] = (
    ("zulu", "2026-03-01T17:00:00Z"),
    ("plus_0530", "2026-03-01T17:00:00+05:30"),
    ("minus_0000", "2026-03-01T17:00:00-00:00"),
    ("minus_0800", "2026-03-01T17:00:00-08:00"),
    ("naive", "2026-03-01T17:00:00"),
    ("minutes_only", "2026-03-01T17:05"),
    ("nine_digit_fraction", "2026-03-01T17:00:00.123456789Z"),
    ("year_floor", "0001-01-01T00:00Z"),
    ("leap_day", "2024-02-29T23:59:59Z"),
)

_TIMESTAMP_REJECTED: tuple[tuple[str, Any], ...] = (
    ("date_only", "2026-03-01"),
    ("space_separator", "2026-03-01 17:00:00Z"),
    ("lowercase_t", "2026-03-01t17:00:00Z"),
    ("lowercase_z", "2026-03-01T17:00:00z"),
    ("offset_without_colon", "2026-03-01T17:00:00+0530"),
    ("epoch_int", 1772384400),
    ("epoch_string", "1772384400"),
    ("year_0000", "0000-01-01T00:00:00Z"),
    ("feb_30", "2026-02-30T00:00:00Z"),
    ("hour_24", "2026-03-01T24:00:00Z"),
    ("second_60", "2026-03-01T23:59:60Z"),
    ("overflows_after_utc", "9999-12-31T23:59:00-01:00"),
    ("underflows_after_utc", "0001-01-01T00:00:00+01:00"),
)

# Shared by platform_id and message_id: both are 1-512 code points of
# well-formed text.
_SOURCE_ID_CASES: tuple[tuple[str, Any, bool], ...] = (
    ("one_char", "x", True),
    ("empty", "", False),
    ("len_512_bmp", _BMP_CHAR * _SOURCE_ID_MAX, True),
    ("len_513_bmp", _BMP_CHAR * (_SOURCE_ID_MAX + 1), False),
    ("len_512_astral", _ASTRAL_CHAR * _SOURCE_ID_MAX, True),
    ("len_513_astral", _ASTRAL_CHAR * (_SOURCE_ID_MAX + 1), False),
    ("lone_high_surrogate", f"id-{_LONE_HIGH}", False),
    ("lone_low_surrogate", f"id-{_LONE_LOW}", False),
    ("reversed_surrogate_pair", f"{_LONE_LOW}{_LONE_HIGH}", False),
    ("number", 42, False),
)


def _message(name: str, index: int, role: str) -> dict[str, Any]:
    return {
        "content": f"message {index}",
        "message_id": f"{name}:m{index}",
        "role": role,
        "timestamp": _BASE_TIME,
    }


def _conversation(name: str, index: int = 0) -> dict[str, Any]:
    key = f"ingest-contract:{name}:{index}"
    return {
        "created_at": _BASE_TIME,
        "messages": [_message(key, 0, "user"), _message(key, 1, "assistant")],
        "platform": "chatgpt",
        "platform_id": key,
        "title": "Ingest contract fixture",
        "updated_at": "2026-03-01T17:05:00Z",
    }


def _request(name: str) -> dict[str, Any]:
    return {"conversations": [_conversation(name)], "source": "extension_chatgpt"}


def _set(target: dict[str, Any], field: str, value: Any) -> None:
    if value is _ABSENT:
        del target[field]
    else:
        target[field] = value


def _row(name: str, covers: list[str], payload: dict[str, Any], *, valid: bool) -> dict[str, Any]:
    return {"covers": sorted(covers), "name": name, "payload": payload, "valid": valid}


def _source_row(name: str, value: Any, *, valid: bool) -> dict[str, Any]:
    payload = _request(name)
    _set(payload, "source", value)
    return _row(name, [_SOURCE], payload, valid=valid)


def _conv_row(name: str, field: str, value: Any, *, valid: bool) -> dict[str, Any]:
    payload = _request(name)
    _set(payload["conversations"][0], field, value)
    return _row(name, [_CONV + field], payload, valid=valid)


def _message_row(name: str, field: str, value: Any, *, valid: bool) -> dict[str, Any]:
    payload = _request(name)
    _set(payload["conversations"][0]["messages"][0], field, value)
    return _row(name, [_MSG + field], payload, valid=valid)


def _count_row(name: str, count: int, *, valid: bool) -> dict[str, Any]:
    payload = _request(name)
    payload["conversations"] = [_conversation(name, index) for index in range(count)]
    return _row(name, [_CONVERSATIONS], payload, valid=valid)


def _source_rows() -> list[dict[str, Any]]:
    sources = ("extension_chatgpt", "extension_claude", "paste_memories", "zip_upload")
    rows = [_source_row(f"source.{value}", value, valid=True) for value in sources]
    rows.append(_source_row("source.unknown_value", "extension_gemini", valid=False))
    rows.append(_source_row("source.absent", _ABSENT, valid=False))
    rows.append(_source_row("source.null", None, valid=False))
    return rows


def _conversations_rows() -> list[dict[str, Any]]:
    from gubbi.api.v1.ingest import MAX_CONVERSATIONS_PER_REQUEST

    rows = [
        _count_row("conversations.empty", 0, valid=True),
        _count_row("conversations.at_limit", MAX_CONVERSATIONS_PER_REQUEST, valid=True),
        _count_row("conversations.over_limit", MAX_CONVERSATIONS_PER_REQUEST + 1, valid=False),
    ]
    for label, value in (("absent", _ABSENT), ("null", None), ("object", {})):
        payload = _request(f"conversations.{label}")
        _set(payload, "conversations", value)
        rows.append(_row(f"conversations.{label}", [_CONVERSATIONS], payload, valid=False))
    return rows


def _timestamp_rows(prefix: str, build: _RowBuilder, field: str) -> list[dict[str, Any]]:
    rows = [
        build(f"{prefix}.{label}", field, value, valid=True) for label, value in _TIMESTAMP_ACCEPTED
    ]
    rows.extend(
        build(f"{prefix}.{label}", field, value, valid=False)
        for label, value in _TIMESTAMP_REJECTED
    )
    return rows


def _source_id_rows(prefix: str, build: _RowBuilder, field: str) -> list[dict[str, Any]]:
    return [
        build(f"{prefix}.{label}", field, value, valid=valid)
        for label, value, valid in _SOURCE_ID_CASES
    ]


def _conversation_field_rows() -> list[dict[str, Any]]:
    rows = [
        _conv_row("platform.chatgpt", "platform", "chatgpt", valid=True),
        _conv_row("platform.claude", "platform", "claude", valid=True),
        _conv_row("platform.unknown_value", "platform", "gemini", valid=False),
        _conv_row("platform.absent", "platform", _ABSENT, valid=False),
        _conv_row("platform.null", "platform", None, valid=False),
        _conv_row("platform_id.absent", "platform_id", _ABSENT, valid=False),
        _conv_row("platform_id.null", "platform_id", None, valid=False),
        _conv_row("title.absent", "title", _ABSENT, valid=True),
        _conv_row("title.empty", "title", "", valid=True),
        _conv_row("title.null", "title", None, valid=False),
        _conv_row("created_at.absent", "created_at", _ABSENT, valid=False),
        _conv_row("created_at.null", "created_at", None, valid=False),
        _conv_row("updated_at.absent", "updated_at", _ABSENT, valid=True),
        _conv_row("updated_at.null", "updated_at", None, valid=True),
        _conv_row("messages.empty", "messages", [], valid=False),
        _conv_row("messages.absent", "messages", _ABSENT, valid=False),
        _conv_row("messages.null", "messages", None, valid=False),
    ]
    rows.extend(_source_id_rows("platform_id", _conv_row, "platform_id"))
    rows.extend(_timestamp_rows("created_at", _conv_row, "created_at"))
    rows.extend(_timestamp_rows("updated_at", _conv_row, "updated_at"))
    return rows


def _message_field_rows() -> list[dict[str, Any]]:
    from gubbi.constants import MAX_MSG_CHARS

    rows = [
        _message_row(f"role.{role}", "role", role, valid=True)
        for role in ("user", "assistant", "system")
    ]
    rows += [
        _message_row("role.unknown_value", "role", "tool", valid=False),
        _message_row("role.absent", "role", _ABSENT, valid=False),
        _message_row("role.null", "role", None, valid=False),
        _message_row("content.empty", "content", "", valid=True),
        _message_row("content.over_length_limit", "content", "a" * (MAX_MSG_CHARS + 1), valid=True),
        _message_row("content.lone_surrogate", "content", f"text {_LONE_HIGH} text", valid=True),
        _message_row("content.astral", "content", f"text {_ASTRAL_CHAR} text", valid=True),
        _message_row("content.absent", "content", _ABSENT, valid=False),
        _message_row("content.null", "content", None, valid=False),
        _message_row("content.number", "content", 42, valid=False),
        _message_row("timestamp.absent", "timestamp", _ABSENT, valid=True),
        _message_row("timestamp.null", "timestamp", None, valid=True),
        _message_row("message_id.absent", "message_id", _ABSENT, valid=True),
        _message_row("message_id.null", "message_id", None, valid=True),
    ]
    rows.extend(_source_id_rows("message_id", _message_row, "message_id"))
    rows.extend(_timestamp_rows("timestamp", _message_row, "timestamp"))
    return rows


def build_document() -> dict[str, Any]:
    """Build every row and return the fixture document."""
    rows = [
        *_source_rows(),
        *_conversations_rows(),
        *_conversation_field_rows(),
        *_message_field_rows(),
    ]
    return {
        "_comment": FIXTURE_COMMENT,
        "rows": sorted(rows, key=lambda row: row["name"]),
    }


def main() -> int:
    """Serialise the fixture document to stdout and return an exit code."""
    # The package is run from a source checkout (poetry package-mode is off), so
    # the repository root -- not this script's directory -- must be on sys.path
    # for ``import gubbi`` to resolve when invoked as
    # ``scripts/dump_ingest_fixtures.py``.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    json.dump(build_document(), sys.stdout, sort_keys=True, indent="\t", ensure_ascii=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
