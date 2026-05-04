"""Unit tests for reject_tool_call_syntax helper."""

import pytest

from gubbi.core.validation import reject_tool_call_syntax

pytestmark = pytest.mark.unit


class TestRejectToolCallSyntax:
    def test_clean_text_passes(self) -> None:
        reject_tool_call_syntax("Normal journal entry content")

    def test_rejects_opening_tag(self) -> None:
        with pytest.raises(ValueError, match="unparsed tool-call syntax"):
            reject_tool_call_syntax('<parameter name="tags">["health"]</parameter>')

    def test_rejects_closing_tag_alone(self) -> None:
        with pytest.raises(ValueError, match="unparsed tool-call syntax"):
            reject_tool_call_syntax("some text </parameter> more text")

    def test_rejects_bare_parameter_open(self) -> None:
        with pytest.raises(ValueError, match="unparsed tool-call syntax"):
            reject_tool_call_syntax("text <parameter more text")

    def test_rejects_multiline_content(self) -> None:
        bad = 'Good reasoning.\n<parameter name="tags">["health", "work"]\n</parameter>'
        with pytest.raises(ValueError):
            reject_tool_call_syntax(bad)

    def test_empty_string_passes(self) -> None:
        reject_tool_call_syntax("")

    def test_html_unrelated_passes(self) -> None:
        reject_tool_call_syntax("<b>bold</b> and <em>emphasis</em>")

    def test_angled_brackets_without_parameter_pass(self) -> None:
        reject_tool_call_syntax("a < b and b > c")
