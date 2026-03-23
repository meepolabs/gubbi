"""Shared test fixtures."""

import os
from pathlib import Path

import bcrypt
import pytest

from journalctl.config import get_settings
from journalctl.oauth.storage import OAuthStorage
from journalctl.storage.index import SearchIndex
from journalctl.storage.markdown import MarkdownStorage

TEST_PASSWORD = "test-password"
TEST_PASSWORD_HASH = bcrypt.hashpw(TEST_PASSWORD.encode(), bcrypt.gensalt()).decode()


@pytest.fixture
def tmp_journal(tmp_path: Path) -> Path:
    """Create a temporary journal directory structure."""
    (tmp_path / "topics").mkdir()
    (tmp_path / "conversations").mkdir()
    (tmp_path / "knowledge").mkdir()
    (tmp_path / "timeline").mkdir()
    return tmp_path


@pytest.fixture
def storage(tmp_journal: Path) -> MarkdownStorage:
    """MarkdownStorage pointed at a temp directory."""
    return MarkdownStorage(tmp_journal)


@pytest.fixture
def index(tmp_path: Path, tmp_journal: Path) -> SearchIndex:
    """SearchIndex with a temp database."""
    db_path = tmp_path / "test.db"
    idx = SearchIndex(db_path, tmp_journal)
    # Force schema init
    _ = idx.conn
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
