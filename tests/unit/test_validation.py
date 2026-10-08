"""Tests for input validation, sanitization, and path traversal prevention."""

import pytest
from pydantic import ValidationError

from gubbi.validation import (
    harden_llm_topic_path,
    sanitize_freetext,
    sanitize_label,
    slugify,
    validate_title,
    validate_topic,
)


class TestTopicValidation:
    """Topic path validation."""

    @pytest.mark.parametrize(
        "raw",
        [
            pytest.param("notes", id="valid_single_level"),
            pytest.param("work/acme", id="valid_two_levels"),
            pytest.param("projects/my-app", id="valid_with_hyphens"),
        ],
    )
    def test_valid_topic_returned_unchanged(self, raw: str) -> None:
        assert validate_topic(raw) == raw

    def test_lowercase_uppercase(self) -> None:
        assert validate_topic("Work/Acme") == "work/acme"
        assert validate_topic("HEALTH") == "health"

    @pytest.mark.parametrize(
        "raw",
        [
            pytest.param("../../etc/passwd", id="reject_path_traversal"),
            pytest.param("/etc/passwd", id="reject_absolute_path"),
            pytest.param("a/b/c", id="reject_three_levels"),
            pytest.param("work/acme@corp", id="reject_special_chars"),
            pytest.param("work/my job", id="reject_spaces"),
            pytest.param("", id="reject_empty"),
            pytest.param("../secrets", id="reject_dot_dot"),
            pytest.param("work/", id="reject_trailing_slash"),
        ],
    )
    def test_invalid_topic_raises(self, raw: str) -> None:
        with pytest.raises(ValueError):
            validate_topic(raw)


class TestTitleValidation:
    """Conversation title validation."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            pytest.param("Q3 Planning Session", "Q3 Planning Session", id="valid_title"),
            pytest.param("Day-1 Setup Notes", "Day-1 Setup Notes", id="valid_with_hyphens"),
            pytest.param("X", "X", id="valid_single_char"),
            pytest.param(" Leading space", "Leading space", id="strips_leading_space"),
        ],
    )
    def test_sanitized_title(self, raw: str, expected: str) -> None:
        assert validate_title(raw) == expected

    def test_strips_punctuation(self) -> None:
        assert validate_title("Q3 Planning (2025)") == "Q3 Planning 2025"
        assert validate_title("Project: Alpha") == "Project Alpha"
        assert validate_title("Team's Decision") == "Teams Decision"

    @pytest.mark.parametrize(
        "raw",
        [
            pytest.param("", id="reject_empty"),
            pytest.param("!!!", id="reject_all_punctuation"),
        ],
    )
    def test_unsanitizable_title_raises(self, raw: str) -> None:
        with pytest.raises(ValueError):
            validate_title(raw)

    def test_truncates_too_long(self) -> None:
        result = validate_title("A" * 200)
        assert len(result) <= 100


class TestSanitizeLabel:
    """Label sanitization for frontmatter-safe values."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            pytest.param("claude-3.5", "claude-3.5", id="normal_label_unchanged"),
            pytest.param("test\x00tag", "testtag", id="strips_control_chars"),
            pytest.param("hello@world!", "helloworld", id="strips_unsafe_chars"),
            pytest.param(
                "my_tag.v2-beta", "my_tag.v2-beta", id="preserves_dots_hyphens_underscores"
            ),
            pytest.param("a" * 100, "a" * 50, id="enforces_max_length"),
            pytest.param("!@#$%", "", id="empty_after_strip_returns_empty"),
            pytest.param("", "", id="empty_string_returns_empty"),
            pytest.param("   ", "", id="whitespace_only_returns_empty"),
        ],
    )
    def test_sanitized_label(self, raw: str, expected: str) -> None:
        assert sanitize_label(raw) == expected

    def test_custom_max_length(self) -> None:
        assert sanitize_label("a" * 200, max_len=100) == "a" * 100


class TestSanitizeFreetext:
    """Free-text sanitization for markdown content."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            pytest.param("hello\nworld", "hello\nworld", id="preserves_newlines"),
            pytest.param("col1\tcol2", "col1\tcol2", id="preserves_tabs"),
            pytest.param("line\r\n", "line\r\n", id="preserves_carriage_return"),
            pytest.param("test\x00content", "testcontent", id="strips_null_bytes"),
            pytest.param("test\x1bcontent", "testcontent", id="strips_escape_chars"),
            pytest.param(
                "hello \U0001f30d world", "hello \U0001f30d world", id="unicode_preserved"
            ),
        ],
    )
    def test_sanitized_freetext(self, raw: str, expected: str) -> None:
        assert sanitize_freetext(raw) == expected

    def test_enforces_max_length(self) -> None:
        result = sanitize_freetext("a" * 2_000_000)
        assert len(result) == 1_000_000

    def test_custom_max_length(self) -> None:
        result = sanitize_freetext("a" * 1000, max_len=500)
        assert len(result) == 500


class TestSlugify:
    """Title to filename slug conversion."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            pytest.param("Q3 Planning Session", "q3-planning-session", id="basic"),
            pytest.param(
                "Release v2.0 \u2014 Launch Day #1", "release-v2-0-launch-day-1", id="special_chars"
            ),
            pytest.param("a   b   c", "a-b-c", id="multiple_spaces"),
            pytest.param("  hello world  ", "hello-world", id="strips_edges"),
        ],
    )
    def test_slugify(self, raw: str, expected: str) -> None:
        assert slugify(raw) == expected


class TestHardenLLMTopicPath:
    """Validation + sanitization of LLM-sourced topic paths."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (None, None),
            ("", None),
            ("a" * 257, None),
            ("a" * 256, "a" * 256),
            ("work\x00/acme", "work/acme"),
            ("../etc/passwd", None),
            ("Work/Acme", "work/acme"),
            ("  work/acme  ", "work/acme"),
            ("\t\n\r", None),
            # Interior whitespace stripped (LLM-glitch tolerance).
            ("work / acme", "work/acme"),
            ("work\t/acme", "work/acme"),
            ("work \t/ \nacme", "work/acme"),
            # Unicode survives the ASCII-only strip and falls to validate_topic
            # which rejects non-ASCII via [a-z0-9-/] regex.
            ("work\u200b/acme", None),
        ],
    )
    def test_contract(self, raw: str | None, expected: str | None) -> None:
        assert harden_llm_topic_path(raw) == expected

    def test_custom_max_len(self) -> None:
        short_path = "a" * 10
        assert harden_llm_topic_path(short_path, max_len=5) is None
        assert harden_llm_topic_path(short_path, max_len=10) == short_path


class TestTagsMaxLen:
    """Pydantic input validation rejects tags exceeding 128 characters."""

    def test_conversation_accepts_128_char_tag(self) -> None:
        from gubbi.models.conversation import ConversationMeta

        long_ok = "a" * 128
        meta = ConversationMeta(title="t", topic="x", created="2025-01-01", updated="2025-01-01")
        meta.tags = [long_ok]
        assert meta.tags[-1] == long_ok

    def test_conversation_rejects_129_char_tag(self) -> None:
        from gubbi.models.conversation import ConversationMeta

        with pytest.raises(ValidationError):
            ConversationMeta(
                title="t",
                topic="x",
                created="2025-01-01",
                updated="2025-01-01",
                tags=["a" * 129],
            )

    def test_entry_accepts_128_char_tag(self) -> None:
        from gubbi.models.journal import Entry

        e = Entry(id=1, date="2025-01-01", content="hi")
        long_ok = "b" * 128
        e.tags = [long_ok]
        assert e.tags[-1] == long_ok

    def test_entry_rejects_129_char_tag(self) -> None:
        from gubbi.models.journal import Entry

        with pytest.raises(ValidationError):
            Entry(id=1, date="2025-01-01", content="hi", tags=["b" * 129])
