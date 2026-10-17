"""Tests for rule-based extraction skips and the conversation content hash."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta, timezone

import pytest

from gubbi.constants import EXTRACTION_MIN_USER_CHARS, EXTRACTION_VERSION
from gubbi.extraction import skips
from gubbi.extraction.skips import SkipReason, content_hash, evaluate_skip, evaluate_slice_skip
from gubbi.models.conversation import Message


@dataclass(frozen=True)
class _Msg:
    role: str
    content: str
    timestamp: datetime | str | None = None


CONVERSATION = (
    _Msg("user", "I got married last weekend", "2026-06-01T10:00:00+00:00"),
    _Msg("assistant", "Congratulations! How was the ceremony?", "2026-06-01T10:00:05+00:00"),
    _Msg("user", "Small, just family.", "2026-06-01T10:01:00+00:00"),
)


def _evaluate(
    messages: tuple[_Msg, ...],
    *,
    source: str | None = "claude",
    stored_hash: str | None = None,
    stored_version: int | None = None,
) -> SkipReason | None:
    return evaluate_skip(messages, source, stored_hash, stored_version)


def test_normal_conversation_is_not_skipped() -> None:
    assert _evaluate(CONVERSATION) is None


@pytest.mark.parametrize(
    "messages",
    [
        pytest.param((), id="empty"),
        pytest.param((_Msg("system", "You are helpful."),), id="system-only"),
        pytest.param((_Msg("tool", '{"ok": true}'), _Msg("user", "   ")), id="tool-and-blank"),
    ],
)
def test_no_text_after_role_filter_skips(messages: tuple[_Msg, ...]) -> None:
    assert _evaluate(messages) is SkipReason.NO_TEXT


@pytest.mark.parametrize(
    "messages",
    [
        pytest.param((_Msg("user", "hi"),), id="greeting"),
        pytest.param((_Msg("user", "  thanks  "), _Msg("assistant", "x" * 500)), id="ack"),
        pytest.param((_Msg("assistant", "A long assistant monologue " * 20),), id="assistant-only"),
        pytest.param(
            (_Msg("system", "a long system prompt " * 20), _Msg("user", "ok")),
            id="system-text-not-counted",
        ),
    ],
)
def test_trivial_user_text_skips(messages: tuple[_Msg, ...]) -> None:
    assert _evaluate(messages) is SkipReason.TRIVIAL_USER_TEXT


@pytest.mark.parametrize(
    "line",
    [
        pytest.param("I got married", id="short-life-event"),
        pytest.param("x" * EXTRACTION_MIN_USER_CHARS, id="exactly-threshold"),
    ],
)
def test_short_meaningful_user_text_is_not_trivial(line: str) -> None:
    assert _evaluate((_Msg("user", line),)) is None


def test_trivial_threshold_sums_all_user_turns() -> None:
    half = "x" * (EXTRACTION_MIN_USER_CHARS // 2 + 1)
    messages = (_Msg("user", half), _Msg("assistant", "?"), _Msg("user", half))

    assert _evaluate(messages) is None


@pytest.mark.parametrize(
    ("stored_hash_matches", "stored_version", "expected"),
    [
        pytest.param(True, EXTRACTION_VERSION, SkipReason.UNCHANGED, id="hash-and-version-match"),
        pytest.param(True, EXTRACTION_VERSION - 1, None, id="older-version-re-extracts"),
        pytest.param(True, None, None, id="never-extracted-version"),
        pytest.param(False, EXTRACTION_VERSION, None, id="content-changed"),
    ],
)
def test_unchanged_requires_hash_and_version_match(
    stored_hash_matches: bool, stored_version: int | None, expected: SkipReason | None
) -> None:
    stored_hash = content_hash(CONVERSATION) if stored_hash_matches else "0" * 64

    result = _evaluate(CONVERSATION, stored_hash=stored_hash, stored_version=stored_version)

    assert result is expected


def test_unchanged_ignores_missing_stored_hash() -> None:
    assert _evaluate(CONVERSATION, stored_hash=None, stored_version=EXTRACTION_VERSION) is None


def test_automated_source_skips(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(skips, "AUTOMATED_EXTRACTION_SOURCES", frozenset({"agent_run"}))

    assert _evaluate(CONVERSATION, source="agent_run") is SkipReason.AUTOMATED_SOURCE


@pytest.mark.parametrize(
    "source",
    ["claude", "chatgpt", "extension_chatgpt", "extension_claude", "paste_memories", None],
)
def test_no_current_source_is_automated(source: str | None) -> None:
    assert _evaluate(CONVERSATION, source=source) is None


def test_hash_ignores_dropped_turns() -> None:
    with_noise = (
        _Msg("system", "You are helpful.", "2026-06-01T09:59:00+00:00"),
        CONVERSATION[0],
        _Msg("tool", '{"result": 1}'),
        CONVERSATION[1],
        _Msg("user", "  \n"),
        CONVERSATION[2],
    )

    assert content_hash(with_noise) == content_hash(CONVERSATION)


@pytest.mark.parametrize(
    ("index", "changes"),
    [
        pytest.param(0, {"content": "I got engaged last weekend"}, id="user-content"),
        pytest.param(1, {"content": "Congratulations!"}, id="assistant-content"),
        pytest.param(2, {"timestamp": "2026-06-01T10:02:00+00:00"}, id="timestamp"),
        pytest.param(2, {"timestamp": None}, id="timestamp-removed"),
        pytest.param(1, {"role": "user"}, id="role"),
    ],
)
def test_hash_changes_when_kept_message_changes(index: int, changes: dict[str, object]) -> None:
    edited = list(CONVERSATION)
    edited[index] = replace(edited[index], **changes)

    assert content_hash(tuple(edited)) != content_hash(CONVERSATION)


def test_hash_changes_when_kept_turn_added() -> None:
    extended = (*CONVERSATION, _Msg("assistant", "Lovely."))

    assert content_hash(extended) != content_hash(CONVERSATION)


def test_hash_changes_when_turns_reorder() -> None:
    reordered = (CONVERSATION[2], CONVERSATION[1], CONVERSATION[0])

    assert content_hash(reordered) != content_hash(CONVERSATION)


@pytest.mark.parametrize(
    "timestamp",
    [
        pytest.param(datetime(2026, 6, 1, 10, 0, tzinfo=UTC), id="aware-datetime"),
        pytest.param(
            datetime(2026, 6, 1, 12, 0, tzinfo=timezone(timedelta(hours=2))), id="offset-datetime"
        ),
        pytest.param("2026-06-01T10:00:00Z", id="zulu-string"),
        pytest.param("2026-06-01T10:00:00+00:00", id="offset-string"),
    ],
)
def test_hash_is_stable_across_timestamp_spellings(timestamp: datetime | str) -> None:
    reference = (_Msg("user", "I got married", "2026-06-01T10:00:00+00:00"),)

    assert content_hash((_Msg("user", "I got married", timestamp),)) == content_hash(reference)


def test_hash_accepts_stored_message_model() -> None:
    stored = tuple(
        Message(role=m.role, content=m.content, timestamp=m.timestamp) for m in CONVERSATION
    )

    assert content_hash(stored) == content_hash(CONVERSATION)


@pytest.mark.parametrize(
    ("new_messages", "source", "watermark", "stored_matches", "expected"),
    [
        pytest.param(
            CONVERSATION, "claude", None, True, SkipReason.UNCHANGED, id="null-hash-match"
        ),
        pytest.param(CONVERSATION, "claude", None, False, None, id="null-hash-differs"),
        pytest.param((), "claude", None, False, SkipReason.NO_TEXT, id="null-empty"),
        pytest.param((), "claude", 4, False, SkipReason.UNCHANGED, id="watermark-nothing-new"),
        pytest.param(
            (_Msg("system", "prompt"), _Msg("tool", "{}")),
            "claude",
            4,
            False,
            SkipReason.UNCHANGED,
            id="watermark-all-filtered",
        ),
        pytest.param(CONVERSATION, "claude", 4, True, None, id="watermark-ignores-hash"),
        pytest.param(
            (_Msg("user", "ok"),), "claude", 4, False, SkipReason.TRIVIAL_USER_TEXT, id="trivial"
        ),
    ],
)
def test_slice_skip_judges_only_the_new_messages(
    new_messages: tuple[_Msg, ...],
    source: str,
    watermark: int | None,
    stored_matches: bool,
    expected: SkipReason | None,
) -> None:
    stored_hash = content_hash(CONVERSATION) if stored_matches else "0" * 64

    reason = evaluate_slice_skip(new_messages, source, watermark, stored_hash, EXTRACTION_VERSION)

    assert reason is expected


def test_slice_skip_keeps_the_automated_source_rule_above_a_watermark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(skips, "AUTOMATED_EXTRACTION_SOURCES", frozenset({"agent_run"}))

    assert evaluate_slice_skip((), "agent_run", 4, None, None) is SkipReason.AUTOMATED_SOURCE
