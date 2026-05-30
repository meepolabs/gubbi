"""Storage layer constants shared across database, search index, and tools."""

from typing import Final

# SQLite connection tuning
DB_BUSY_TIMEOUT_MS: Final = 5000  # ms to wait on a locked DB before raising OperationalError

# Snippet / display lengths
SNIPPET_PREVIEW_LEN: Final = 200  # chars shown in briefing / list previews
SUMMARY_TRUNCATE_LEN: Final = 300  # chars kept for search result snippets
