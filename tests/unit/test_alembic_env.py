"""Unit tests for alembic env.py DSN resolution.

Verifies the role-aware fallback chain:
  JOURNAL_DB_MIGRATION_URL -> JOURNAL_DB_ADMIN_URL -> JOURNAL_DB_APP_URL
with a deprecation warning on the app-url fallback path.
"""

import logging
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit


_ENV_KEYS = (
    "JOURNAL_DB_MIGRATION_URL",
    "JOURNAL_DB_ADMIN_URL",
    "JOURNAL_DB_APP_URL",
)


def _load_get_database_url() -> Any:
    """Import alembic env.py without invoking its module-level alembic context.

    env.py runs ``run_migrations_online()`` at import time, which requires a
    live alembic context. We sidestep that by reading the file and exec'ing
    only the helper definitions we need.
    """
    env_path = Path(__file__).resolve().parents[2] / "gubbi" / "alembic" / "env.py"
    source = env_path.read_text(encoding="utf-8")
    cutoff = source.index("def run_migrations_offline")
    head = source[:cutoff]
    head = head.replace("from logging.config import fileConfig", "")
    head = head.replace("from alembic import context", "")
    head = head.replace("from sqlalchemy import engine_from_config, pool", "")
    head = head.replace("config = context.config", "")
    head = head.replace(
        "if config.config_file_name is not None:\n    fileConfig(config.config_file_name)",
        "",
    )
    namespace: dict[str, Any] = {}
    exec(compile(head, str(env_path), "exec"), namespace)  # noqa: S102 -- test fixture
    return namespace["get_database_url"]


@pytest.fixture
def get_database_url(monkeypatch: pytest.MonkeyPatch) -> Any:
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    return _load_get_database_url()


class TestMigrationUrlPreferred:
    def test_migration_url_takes_precedence(
        self, get_database_url: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("JOURNAL_DB_MIGRATION_URL", "postgresql://m:p@h:5432/d")
        monkeypatch.setenv("JOURNAL_DB_ADMIN_URL", "postgresql://a:p@h:5432/d")
        monkeypatch.setenv("JOURNAL_DB_APP_URL", "postgresql://x:p@h:5432/d")
        assert get_database_url() == "postgresql+psycopg://m:p@h:5432/d"


class TestAdminUrlFallback:
    def test_admin_url_used_when_migration_url_absent(
        self, get_database_url: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("JOURNAL_DB_ADMIN_URL", "postgresql://a:p@h:5432/d")
        monkeypatch.setenv("JOURNAL_DB_APP_URL", "postgresql://x:p@h:5432/d")
        assert get_database_url() == "postgresql+psycopg://a:p@h:5432/d"

    def test_admin_url_no_deprecation_warning(
        self,
        get_database_url: Any,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setenv("JOURNAL_DB_ADMIN_URL", "postgresql://a:p@h:5432/d")
        with caplog.at_level(logging.WARNING):
            get_database_url()
        assert not any("least-privilege" in r.message for r in caplog.records)


class TestAppUrlDeprecatedFallback:
    def test_app_url_used_when_others_absent(
        self, get_database_url: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("JOURNAL_DB_APP_URL", "postgresql://x:p@h:5432/d")
        assert get_database_url() == "postgresql+psycopg://x:p@h:5432/d"

    def test_app_url_emits_deprecation_warning(
        self,
        get_database_url: Any,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setenv("JOURNAL_DB_APP_URL", "postgresql://x:p@h:5432/d")
        with caplog.at_level(logging.WARNING):
            get_database_url()
        assert any(
            "JOURNAL_DB_APP_URL" in r.message and "least-privilege" in r.message
            for r in caplog.records
        )


class TestNoUrlSet:
    def test_raises_when_all_three_absent(self, get_database_url: Any) -> None:
        with pytest.raises(RuntimeError, match="No migration DSN found"):
            get_database_url()


class TestDriverSchemeRewrite:
    def test_postgresql_rewrites_to_psycopg(
        self, get_database_url: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("JOURNAL_DB_MIGRATION_URL", "postgresql://m:p@h:5432/d")
        assert get_database_url().startswith("postgresql+psycopg://")

    def test_asyncpg_rewrites_to_psycopg(
        self, get_database_url: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("JOURNAL_DB_MIGRATION_URL", "postgresql+asyncpg://m:p@h:5432/d")
        assert get_database_url() == "postgresql+psycopg://m:p@h:5432/d"

    def test_already_psycopg_left_alone(
        self, get_database_url: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        url = "postgresql+psycopg://m:p@h:5432/d"
        monkeypatch.setenv("JOURNAL_DB_MIGRATION_URL", url)
        assert get_database_url() == url
