"""Shared test fixtures."""

import os
from pathlib import Path

import bcrypt
import pytest

from journalctl.config import get_settings
from journalctl.oauth.storage import OAuthStorage
from journalctl.storage.database import DatabaseStorage
from journalctl.storage.index import SearchIndex

TEST_PASSWORD = "test-password"
TEST_PASSWORD_HASH = bcrypt.hashpw(TEST_PASSWORD.encode(), bcrypt.gensalt()).decode()


@pytest.fixture
def tmp_journal(tmp_path: Path) -> Path:
    """Create a temporary journal directory structure."""
    (tmp_path / "knowledge").mkdir()
    return tmp_path


@pytest.fixture
def storage(tmp_journal: Path, tmp_path: Path) -> DatabaseStorage:
    """DatabaseStorage pointed at a temp directory."""
    db = DatabaseStorage(tmp_path / "test.db", tmp_journal)
    _ = db.conn  # Force schema init
    yield db
    db.close()


@pytest.fixture
def index(tmp_path: Path) -> SearchIndex:
    """SearchIndex with a temp database (shares same db as storage)."""
    idx = SearchIndex(tmp_path / "test.db")
    _ = idx.conn  # Force schema init
    yield idx
    idx.close()


@pytest.fixture(autouse=True)
def _set_env(tmp_journal: Path, tmp_path: Path) -> None:
    """Set environment variables for tests and clear settings cache."""
    os.environ["JOURNAL_API_KEY"] = "test-key"
    os.environ["JOURNAL_JOURNAL_ROOT"] = str(tmp_journal)
    os.environ["JOURNAL_DB_PATH"] = str(tmp_path / "test.db")
    os.environ["JOURNAL_TRANSPORT"] = "stdio"
    os.environ["JOURNAL_SERVER_URL"] = "http://localhost:8100"
    os.environ["JOURNAL_OWNER_PASSWORD_HASH"] = TEST_PASSWORD_HASH
    os.environ["JOURNAL_OAUTH_DB_PATH"] = str(tmp_path / "oauth.db")
    os.environ["JOURNAL_OAUTH_ACCESS_TOKEN_TTL"] = "3600"
    os.environ["JOURNAL_OAUTH_REFRESH_TOKEN_TTL"] = "2592000"
    os.environ["JOURNAL_OAUTH_AUTH_CODE_TTL"] = "300"
    get_settings.cache_clear()


@pytest.fixture
def oauth_storage(tmp_path: Path) -> OAuthStorage:
    """OAuthStorage with a temp database."""
    db = OAuthStorage(tmp_path / "oauth.db")
    _ = db.conn  # Force schema init
    yield db
    db.close()
