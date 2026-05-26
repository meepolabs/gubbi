"""Tests for `gubbi.config.Settings` -- canonical Environment Literal + properties.

The config layer provides the byte-identical
``Environment = Literal["dev", "ci", "staging", "production"]`` alias and the
``is_deployed`` property. These tests pin both, mirroring the corresponding
gubbi-cloud module.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

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
    with patch.dict(os.environ, env, clear=True):
        return Settings()


@pytest.mark.parametrize(
    ("app_env", "expected_is_deployed"),
    [
        ("dev", False),
        ("ci", False),
        ("staging", True),
        ("production", True),
    ],
)
def test_is_deployed_property_enumerates_four_values(
    app_env: str, expected_is_deployed: bool
) -> None:
    """`is_deployed` must be True iff app_env in (staging, production)."""
    # Arrange + Act
    settings = _make_settings(JOURNAL_APP_ENV=app_env)

    # Assert
    assert settings.app_env == app_env
    assert settings.is_deployed is expected_is_deployed
