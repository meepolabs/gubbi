"""Tests for input validation — path traversal prevention."""

import pytest

from journalctl.models.entry import slugify, validate_title, validate_topic


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
