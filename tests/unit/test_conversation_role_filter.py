"""Unit tests for the shared conversation role filter."""

from __future__ import annotations

import pytest

from gubbi.models.conversation import Message, keep_conversation_roles


def _msg(role: str, content: str) -> Message:
    return Message(role=role, content=content)


def test_keeps_user_and_assistant_turns_in_input_order() -> None:
    messages = [
        _msg("assistant", "a1"),
        _msg("user", "u1"),
        _msg("user", "u2"),
        _msg("assistant", "a2"),
    ]

    kept = keep_conversation_roles(messages)

    assert [(m.role, m.content) for m in kept] == [
        ("assistant", "a1"),
        ("user", "u1"),
        ("user", "u2"),
        ("assistant", "a2"),
    ]


@pytest.mark.parametrize(
    "dropped",
    [
        pytest.param(_msg("system", "You are a helpful assistant."), id="system-role"),
        pytest.param(_msg("tool", '{"result": 42}'), id="tool-role"),
        pytest.param(_msg("function", "call()"), id="function-role"),
        pytest.param(_msg("", "no role"), id="empty-role"),
        pytest.param(_msg("user", ""), id="empty-user-content"),
        pytest.param(_msg("assistant", "  \n\t "), id="whitespace-assistant-content"),
    ],
)
def test_drops_non_conversation_turns(dropped: Message) -> None:
    messages = [_msg("user", "before"), dropped, _msg("assistant", "after")]

    kept = keep_conversation_roles(messages)

    assert [(m.role, m.content) for m in kept] == [("user", "before"), ("assistant", "after")]


def test_returns_empty_list_when_nothing_survives() -> None:
    messages = [_msg("system", "prompt"), _msg("user", " ")]

    assert keep_conversation_roles(messages) == []
