"""Unit tests for tool-layer constants (TASK-03.23)."""

import pytest

from journalctl.tools.constants import (
    DEFAULT_CONVERSATION_MESSAGES_LIMIT,
    DEFAULT_CONVERSATIONS_LIMIT,
    DEFAULT_TIMELINE_LIMIT,
    DEFAULT_TOPICS_LIMIT,
    MAX_CONVERSATION_MESSAGES,
    MAX_CONVERSATIONS_RESULTS,
    MAX_READ_ENTRIES,
    MAX_SEARCH_RESULTS,
    MAX_TIMELINE_ENTRIES,
    MAX_TOPICS_RESULTS,
)

pytestmark = pytest.mark.unit


def test_read_maxes_are_tightened() -> None:
    assert MAX_SEARCH_RESULTS == 20
    assert MAX_TOPICS_RESULTS == 20
    assert MAX_CONVERSATIONS_RESULTS == 20
    assert MAX_READ_ENTRIES == 20


def test_new_constants_exist() -> None:
    assert MAX_TIMELINE_ENTRIES == 50
    assert DEFAULT_TIMELINE_LIMIT == 20
    assert MAX_CONVERSATION_MESSAGES == 100
    assert DEFAULT_CONVERSATION_MESSAGES_LIMIT == 20


def test_defaults_not_exceed_maxes() -> None:
    assert DEFAULT_TOPICS_LIMIT <= MAX_TOPICS_RESULTS
    assert DEFAULT_CONVERSATIONS_LIMIT <= MAX_CONVERSATIONS_RESULTS
    assert DEFAULT_TIMELINE_LIMIT <= MAX_TIMELINE_ENTRIES
    assert DEFAULT_CONVERSATION_MESSAGES_LIMIT <= MAX_CONVERSATION_MESSAGES
