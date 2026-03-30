"""Search result model."""

from pydantic import BaseModel


class SearchResult(BaseModel):
    """A single search result from FTS5."""

    source_key: str  # 'entry:42', 'conversation:17'
    doc_type: str  # 'entry' or 'conversation'
    topic: str
    title: str
    snippet: str
    rank: float
    date: str
    entry_id: int | None = None
    conversation_id: int | None = None
