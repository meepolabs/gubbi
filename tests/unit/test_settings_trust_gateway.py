"""Tests for the trust_gateway / gateway_require_signature safety validator.

The outer Settings._validate_trust_gateway_signature validator must refuse
auth.trust_gateway=True together with auth.gateway_require_signature=False
when app_env != "dev" -- closes the C-011 default-unsafe footgun.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from gubbi.config import Settings


def _base_env(**extra: str) -> dict[str, str]:
    base: dict[str, str] = {
        "JOURNAL_DB_APP_URL": "postgresql://user:pass@localhost/db",
        "JOURNAL_API_KEY": "a-valid-api-key-that-is-at-least-32-chars-long",
        "JOURNAL_OPERATOR_EMAIL": "op@example.com",
    }
    base.update(extra)
    return base


def _make_settings(**extra: str) -> Settings:
    env = _base_env(**extra)
    with patch.dict(os.environ, env, clear=False):
        return Settings()


def test_prod_trust_gateway_without_signature_raises() -> None:
    # Arrange + Act + Assert
    with pytest.raises(ValidationError, match="gateway_require_signature=True"):
        _make_settings(
            JOURNAL_APP_ENV="prod",
            JOURNAL_TRUST_GATEWAY="true",
            JOURNAL_GATEWAY_REQUIRE_SIGNATURE="false",
        )


def test_dev_trust_gateway_without_signature_allowed() -> None:
    # Arrange + Act
    s = _make_settings(
        JOURNAL_APP_ENV="dev",
        JOURNAL_TRUST_GATEWAY="true",
        JOURNAL_GATEWAY_REQUIRE_SIGNATURE="false",
    )

    # Assert
    assert s.app_env == "dev"
    assert s.auth.trust_gateway is True
    assert s.auth.gateway_require_signature is False


def test_prod_trust_gateway_with_signature_allowed() -> None:
    # Arrange + Act
    s = _make_settings(
        JOURNAL_APP_ENV="prod",
        JOURNAL_TRUST_GATEWAY="true",
        JOURNAL_GATEWAY_REQUIRE_SIGNATURE="true",
    )

    # Assert
    assert s.app_env == "prod"
    assert s.auth.trust_gateway is True
    assert s.auth.gateway_require_signature is True
