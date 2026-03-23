"""Shared test fixtures."""

import os
from pathlib import Path

import pytest

from journalctl.config import get_settings
from journalctl.storage.index import SearchIndex
from journalctl.storage.markdown import MarkdownStorage


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
    get_settings.cache_clear()
