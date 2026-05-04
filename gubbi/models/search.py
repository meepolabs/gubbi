"""Search result model."""

from typing import Literal

from pydantic import BaseModel


class SearchResult(BaseModel):
    """A single search result from search backends.

    ``doc_type="memory"`` represents a semantic-only result with no matching
    journal entry or conversation record. In that case both ``entry_id`` and
    ``conversation_id`` may be ``None``.
    """

    source_key: str  # e.g. 'entry:42', 'conversation:17', or a memory hash key
    doc_type: Literal["entry", "conversation", "memory"]
    topic: str
    rank: float
    date: str
    entry_id: int | None = None
    conversation_id: int | None = None
    content: str | None = None
    title: str | None = None
    summary: str | None = None
    decryption_failed: bool = False  # True when content was unreadable (M-9.8)
