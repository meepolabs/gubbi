"""Tool layer constants: limits, defaults, and allowed values for MCP tools."""

from typing import Final

# Allowed message roles when saving conversations
KEEP_ROLES: Final = frozenset({"user", "assistant"})

# Per-message character cap — prevents runaway tool output from bloating storage
MAX_MSG_CHARS: Final = 20_000

# Briefing / context tool display limits
BRIEFING_MAX_TOPICS: Final = 20
BRIEFING_MAX_WEEK_ENTRIES: Final = 25
BRIEFING_KEY_FACTS_QUERY: Final = "user identity preferences habits goals current status"
BRIEFING_KEY_FACTS_COUNT: Final = 7

# Upper-bound cap for journal_read_topic n parameter — prevents loading entire journal into memory
MAX_READ_ENTRIES: Final = 20

# Default pagination / result limits for tool parameters
DEFAULT_ENTRIES_LIMIT: Final = 10
DEFAULT_SEARCH_LIMIT: Final = 10
DEFAULT_TOPICS_LIMIT: Final = 20
DEFAULT_CONVERSATIONS_LIMIT: Final = 20
DEFAULT_TIMELINE_LIMIT: Final = 20

# Upper-bound caps for unbounded list/search parameters
MAX_SEARCH_RESULTS: Final = 20
MAX_TOPICS_RESULTS: Final = 20
MAX_CONVERSATIONS_RESULTS: Final = 20
MAX_TIMELINE_ENTRIES: Final = 50
MAX_CONVERSATION_MESSAGES: Final = 100
DEFAULT_CONVERSATION_MESSAGES_LIMIT: Final = 20

# Search query input guard - truncate before FTS5 or embedding call
MAX_QUERY_LEN: Final = 2000

# Per-result content cap for search/context responses
MAX_SEARCH_CONTENT_CHARS: Final = 4096

# Characters shown for memory content-hash previews in search results
MEMORY_HASH_PREVIEW_LEN: Final = 12

# Batch size for reindex semantic embedding loop
REINDEX_BATCH_SIZE: Final = 100

# Maximum messages accepted per journal_save_conversation call
MAX_MESSAGES_PER_CONVERSATION: Final = 1000

# Truncation length for conversation summaries in list view
LIST_SUMMARY_PREVIEW_CHARS: Final = 400
