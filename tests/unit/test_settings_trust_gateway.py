"""Tests for the trust_gateway / gateway_require_signature safety validator.

The outer Settings._validate_trust_gateway_signature validator must refuse
auth.trust_gateway=True together with auth.gateway_require_signature=False
when self.is_deployed is True (staging + production) -- closes the C-011
default-unsafe footgun. dev + ci are non-deployed and intentionally
permissive (DEC-094 / H-A2).
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
        "JOURNAL_API_KEY": "placeholder-not-a-real-key-padding-to-32plus",
        "JOURNAL_OPERATOR_EMAIL": "op@example.com",
    }
    base.update(extra)
    return base


def _make_settings(**extra: str) -> Settings:
    env = _base_env(**extra)
    with patch.dict(os.environ, env, clear=True):
        return Settings()


def test_prod_trust_gateway_without_signature_raises() -> None:
    # Arrange + Act + Assert
    with pytest.raises(ValidationError, match="gateway_require_signature=True"):
        _make_settings(
            JOURNAL_APP_ENV="production",
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
        JOURNAL_APP_ENV="production",
        JOURNAL_TRUST_GATEWAY="true",
        JOURNAL_GATEWAY_REQUIRE_SIGNATURE="true",
    )

    # Assert
    assert s.app_env == "production"
    assert s.auth.trust_gateway is True
    assert s.auth.gateway_require_signature is True


def test_ci_allows_trust_gateway_without_signature() -> None:
    """Pin the intentional CI permissiveness introduced by B1's `is_deployed` flip.

    Before B1 the validator gated on ``app_env != "dev"`` which caught ``ci`` too;
    after B1 (per architect H-A2 / DEC-094) it gates on ``self.is_deployed``,
    which is False for both ``dev`` and ``ci``. CI test harnesses set
    ``trust_gateway=True`` + ``gateway_require_signature=False`` to bypass the
    signed envelope when wiring synthetic users into gubbi -- this combination
    must construct without raising under ``app_env=ci``.
    """
    # Arrange + Act
    s = _make_settings(
        JOURNAL_APP_ENV="ci",
        JOURNAL_TRUST_GATEWAY="true",
        JOURNAL_GATEWAY_REQUIRE_SIGNATURE="false",
    )

    # Assert
    assert s.app_env == "ci"
    assert s.is_deployed is False
    assert s.auth.trust_gateway is True
    assert s.auth.gateway_require_signature is False


def test_staging_trust_gateway_without_signature_raises() -> None:
    """Mirror the production test: staging is a deployed env and must fail closed."""
    # Arrange + Act + Assert
    with pytest.raises(ValidationError, match="gateway_require_signature=True"):
        _make_settings(
            JOURNAL_APP_ENV="staging",
            JOURNAL_TRUST_GATEWAY="true",
            JOURNAL_GATEWAY_REQUIRE_SIGNATURE="false",
        )
