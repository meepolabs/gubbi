"""Tests for m234-review LOW bundle: mode-3 operator rejection, tags max_length, scope precompute."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

# Item 8 scope precompute tests are in test_scope.py (TestScopePrecomputeCorrectness).


# ===========================================================================
# Item 1: Mode-3 deploy rejects JOURNAL_OPERATOR_EMAIL alongside Hydra
# ===========================================================================


def _make_mode3(*, extra: dict[str, str] | None = None) -> object:
    """Build settings for a mode-3 (Hydra) environment."""
    base: dict[str, str] = {
        "JOURNAL_DB_APP_URL": "postgresql://user:pass@localhost/db",
        "JOURNAL_API_KEY": "a-valid-api-key-that-is-at-least-32-chars-long",
        "JOURNAL_AUTH__HYDRA_ADMIN_URL": "https://hydra.example.com",
        "JOURNAL_AUTH__HYDRA_PUBLIC_ISSUER_URL": "https://auth.example.com",
        "JOURNAL_AUTH__HYDRA_PUBLIC_URL": "https://example.com",
    }
    if extra:
        base.update(extra)
    # Clear PASSWORD_HASH (mutual-exclusivity check). Also clear flat-form
    # JOURNAL_OPERATOR_EMAIL because the conftest _set_env fixture (autouse)
    # always sets it to "operator@test.local". For the rejection test, callers
    # set JOURNAL_AUTH__OPERATOR_EMAIL in extra.
    env = {
        **base,
        "JOURNAL_AUTH__PASSWORD_HASH": "",
        "JOURNAL_OPERATOR_EMAIL": "",
    }
    with patch.dict(os.environ, env):
        from gubbi.config import Settings  # noqa: PLC0415

        return Settings()


class TestMode3OperatorEmailRejection:
    """Reject when both JOURNAL_HYDRA_ADMIN_URL and JOURNAL_OPERATOR_EMAIL are set."""

    def test_mode3_with_operator_email_raises(self) -> None:
        """When Hydra is on AND operator_email is set -- raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            _make_mode3(extra={"JOURNAL_AUTH__OPERATOR_EMAIL": "op@example.com"})
        msg = str(exc_info.value)
        assert "JOURNAL_OPERATOR_EMAIL must not be set when JOURNAL_HYDRA_ADMIN_URL" in msg
        assert "mode 3 (multi-tenant hosted) has no operator concept" in msg

    def test_mode3_without_operator_email_succeeds(self) -> None:
        """Mode 3 without operator_email should work fine."""
        _make_mode3()
        # Settings validated -- operator email defaults empty.

    def test_mode2_with_operator_email_succeeds(self) -> None:
        """Mode 2 (password-based, no Hydra) with operator_email works."""
        env: dict[str, str] = {
            "JOURNAL_DB_APP_URL": "postgresql://user:pass@localhost/db",
            "JOURNAL_API_KEY": "a-valid-api-key-that-is-at-least-32-chars-long",
            "JOURNAL_AUTH__PASSWORD_HASH": "$2b$12$salt_hash_here",
            "JOURNAL_AUTH__OPERATOR_EMAIL": "op@example.com",
        }
        with patch.dict(os.environ, {**env, "JOURNAL_AUTH__HYDRA_ADMIN_URL": ""}):
            from gubbi.config import Settings  # noqa: PLC0415

            Settings()


# ===========================================================================
# Item 2: tags list[str] capped at 128 chars per element
# ===========================================================================


class TestTagsMaxLen:
    """Pydantic input validation rejects tags exceeding 128 characters."""

    def test_conversation_accepts_128_char_tag(self) -> None:
        from gubbi.models.conversation import ConversationMeta  # noqa: PLC0415

        long_ok = "a" * 128
        meta = ConversationMeta(title="t", topic="x", created="2025-01-01", updated="2025-01-01")
        meta.tags = [long_ok]
        assert meta.tags[-1] == long_ok

    def test_conversation_rejects_129_char_tag(self) -> None:
        from pydantic import ValidationError  # noqa: PLC0415 from Pydantic v2

        from gubbi.models.conversation import ConversationMeta  # noqa: PLC0415

        with pytest.raises(ValidationError):
            ConversationMeta(
                title="t",
                topic="x",
                created="2025-01-01",
                updated="2025-01-01",
                tags=["a" * 129],
            )

    def test_entry_accepts_128_char_tag(self) -> None:
        from gubbi.models.journal import Entry  # noqa: PLC0415

        e = Entry(id=1, date="2025-01-01", content="hi")
        long_ok = "b" * 128
        e.tags = [long_ok]
        assert e.tags[-1] == long_ok

    def test_entry_rejects_129_char_tag(self) -> None:
        from pydantic import ValidationError  # noqa: PLC0415 -- Pydantic v2

        from gubbi.models.journal import Entry  # noqa: PLC0415

        with pytest.raises(ValidationError):
            Entry(id=1, date="2025-01-01", content="hi", tags=["b" * 129])


# ===========================================================================
# Item 8: Scope precompute correctness -- non-trivial hierarchies still work
# ===========================================================================
