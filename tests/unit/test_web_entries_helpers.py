"""DB-free unit tests for the web entries router helpers.

Covers the pure mapping + guard helpers that need no database:
- ``_has_update_field`` -- the PATCH "at least one field" guard.
- ``row_to_item`` / ``row_to_detail`` -- row -> API-shape mapping, including the
  decryption-failure sentinel path and the list-vs-detail reasoning split.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

from gubbi.api.v1.web.entries import row_to_detail, row_to_item
from gubbi.api.v1.web.entries_admin import EntryUpdateRequest, _has_update_field
from gubbi.crypto.cipher import ContentCipher

_CIPHER = ContentCipher({1: bytes([1]) * 32})


def _row(*, content: str, reasoning: str | None) -> dict[str, object]:
    """Build a fake repository row with real ciphertext for the test cipher."""
    content_ct, content_nonce = _CIPHER.encrypt(content)
    if reasoning is not None:
        reasoning_ct, reasoning_nonce = _CIPHER.encrypt(reasoning)
    else:
        reasoning_ct, reasoning_nonce = None, None
    return {
        "id": 7,
        "topic_path": "work/acme",
        "date": date(2026, 6, 10),
        "content_encrypted": content_ct,
        "content_nonce": content_nonce,
        "reasoning_encrypted": reasoning_ct,
        "reasoning_nonce": reasoning_nonce,
        "tags": ["decision"],
        "conversation_id": None,
        "created_at": datetime(2026, 6, 10, 12, 0, tzinfo=UTC),
        "updated_at": datetime(2026, 6, 10, 12, 0, tzinfo=UTC),
    }


class TestHasUpdateField:
    """The PATCH all-optional guard rejects an empty body."""

    def test_empty_request_is_false(self) -> None:
        """No fields set -> guard returns False."""
        assert _has_update_field(EntryUpdateRequest()) is False

    def test_any_single_field_is_true(self) -> None:
        """Each field alone satisfies the guard."""
        assert _has_update_field(EntryUpdateRequest(content="x")) is True
        assert _has_update_field(EntryUpdateRequest(tags=[])) is True
        assert _has_update_field(EntryUpdateRequest(topic_path="a/b")) is True


class TestRowToItem:
    """List-item mapping omits reasoning; surfaces the decryption flag."""

    def test_decrypts_content_no_reasoning_key(self) -> None:
        """A good row maps content decrypted with decryption_failed False."""
        item = row_to_item(_CIPHER, _row(content="hello", reasoning="ignored"))
        assert item.content == "hello"
        assert item.decryption_failed is False
        assert not hasattr(item, "reasoning")
        assert item.created_at.startswith("2026-06-10")

    def test_corrupt_row_sets_sentinel_and_flag(self) -> None:
        """A row whose ciphertext cannot decrypt yields the sentinel + flag."""
        row = _row(content="x", reasoning=None)
        row["content_encrypted"] = b"\x00garbage"
        item = row_to_item(_CIPHER, row)
        assert item.decryption_failed is True
        assert item.content == "[decryption failed]"


class TestRowToDetail:
    """Detail mapping adds reasoning and ORs the failure flags."""

    def test_includes_reasoning(self) -> None:
        """Detail surfaces decrypted reasoning alongside content."""
        detail = row_to_detail(_CIPHER, _row(content="c", reasoning="r"))
        assert detail.content == "c"
        assert detail.reasoning == "r"
        assert detail.decryption_failed is False

    def test_reasoning_null_when_absent(self) -> None:
        """A NULL reasoning column maps to reasoning=None, not the sentinel."""
        detail = row_to_detail(_CIPHER, _row(content="c", reasoning=None))
        assert detail.reasoning is None
        assert detail.decryption_failed is False
