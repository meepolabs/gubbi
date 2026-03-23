"""Tests for input validation, sanitization, and path traversal prevention."""

import pytest

from journalctl.models.entry import (
    sanitize_freetext,
    sanitize_label,
    slugify,
    validate_title,
    validate_topic,
)


class TestTopicValidation:
    """Topic path validation."""

    def test_valid_single_level(self) -> None:
        assert validate_topic("notes") == "notes"

    def test_valid_two_levels(self) -> None:
        assert validate_topic("work/acme") == "work/acme"

    def test_valid_with_hyphens(self) -> None:
        assert validate_topic("projects/my-app") == "projects/my-app"

    def test_reject_path_traversal(self) -> None:
        with pytest.raises(ValueError):
            validate_topic("../../etc/passwd")

    def test_reject_absolute_path(self) -> None:
        with pytest.raises(ValueError):
            validate_topic("/etc/passwd")

    def test_reject_uppercase(self) -> None:
        with pytest.raises(ValueError):
            validate_topic("Work/Acme")

    def test_reject_three_levels(self) -> None:
        with pytest.raises(ValueError):
            validate_topic("a/b/c")

    def test_reject_special_chars(self) -> None:
        with pytest.raises(ValueError):
            validate_topic("work/acme@corp")

    def test_reject_spaces(self) -> None:
        with pytest.raises(ValueError):
            validate_topic("work/my job")

    def test_reject_empty(self) -> None:
        with pytest.raises(ValueError):
            validate_topic("")

    def test_reject_dot_dot(self) -> None:
        with pytest.raises(ValueError):
            validate_topic("../secrets")

    def test_reject_trailing_slash(self) -> None:
        with pytest.raises(ValueError):
            validate_topic("work/")


class TestTitleValidation:
    """Conversation title validation."""

    def test_valid_title(self) -> None:
        assert validate_title("Q3 Planning Session") == "Q3 Planning Session"

    def test_valid_with_hyphens(self) -> None:
        assert validate_title("Day-1 Setup Notes") == "Day-1 Setup Notes"

    def test_valid_single_char(self) -> None:
        assert validate_title("X") == "X"

    def test_reject_empty(self) -> None:
        with pytest.raises(ValueError):
            validate_title("")

    def test_reject_too_long(self) -> None:
        with pytest.raises(ValueError):
            validate_title("A" * 200)

    def test_reject_special_start(self) -> None:
        with pytest.raises(ValueError):
            validate_title(" Leading space")


class TestSanitizeLabel:
    """Label sanitization for frontmatter-safe values."""

    def test_normal_label_unchanged(self) -> None:
        assert sanitize_label("claude-3.5") == "claude-3.5"

    def test_strips_control_chars(self) -> None:
        assert sanitize_label("test\x00tag") == "testtag"

    def test_strips_unsafe_chars(self) -> None:
        assert sanitize_label("hello@world!") == "helloworld"

    def test_preserves_dots_hyphens_underscores(self) -> None:
        assert sanitize_label("my_tag.v2-beta") == "my_tag.v2-beta"

    def test_enforces_max_length(self) -> None:
        assert sanitize_label("a" * 100) == "a" * 50

    def test_custom_max_length(self) -> None:
        assert sanitize_label("a" * 200, max_len=100) == "a" * 100

    def test_empty_after_strip_returns_unknown(self) -> None:
        assert sanitize_label("!@#$%") == "unknown"

    def test_empty_string_returns_unknown(self) -> None:
        assert sanitize_label("") == "unknown"

    def test_whitespace_only_returns_unknown(self) -> None:
        assert sanitize_label("   ") == "unknown"


class TestSanitizeFreetext:
    """Free-text sanitization for markdown content."""

    def test_preserves_newlines(self) -> None:
        assert sanitize_freetext("hello\nworld") == "hello\nworld"

    def test_preserves_tabs(self) -> None:
        assert sanitize_freetext("col1\tcol2") == "col1\tcol2"

    def test_preserves_carriage_return(self) -> None:
        assert sanitize_freetext("line\r\n") == "line\r\n"

    def test_strips_null_bytes(self) -> None:
        assert sanitize_freetext("test\x00content") == "testcontent"

    def test_strips_escape_chars(self) -> None:
        assert sanitize_freetext("test\x1bcontent") == "testcontent"

    def test_enforces_max_length(self) -> None:
        result = sanitize_freetext("a" * 2_000_000)
        assert len(result) == 1_000_000

    def test_custom_max_length(self) -> None:
        result = sanitize_freetext("a" * 1000, max_len=500)
        assert len(result) == 500

    def test_unicode_preserved(self) -> None:
        assert sanitize_freetext("hello 🌍 world") == "hello 🌍 world"


class TestSlugify:
    """Title to filename slug conversion."""

    def test_basic(self) -> None:
        assert slugify("Q3 Planning Session") == "q3-planning-session"

    def test_special_chars(self) -> None:
        assert slugify("Release v2.0 — Launch Day #1") == "release-v2-0-launch-day-1"

    def test_multiple_spaces(self) -> None:
        assert slugify("a   b   c") == "a-b-c"

    def test_strips_edges(self) -> None:
        assert slugify("  hello world  ") == "hello-world"
