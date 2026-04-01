"""Journal data models: TopicMeta and Entry."""

from pydantic import BaseModel, field_validator

from journalctl.core.validation import validate_topic


class TopicMeta(BaseModel):
    """Metadata for a topic."""

    id: int | None = None  # stable DB row ID
    topic: str
    title: str
    description: str = ""
    created: str  # YYYY-MM-DD
    updated: str  # YYYY-MM-DD
    entry_count: int = 0

    @field_validator("topic")
    @classmethod
    def check_topic(cls, v: str) -> str:
        return validate_topic(v)


class Entry(BaseModel):
    """A single dated journal entry."""

    id: int  # stable DB primary key
    date: str  # YYYY-MM-DD
    content: str  # what happened (headline)
    reasoning: str | None = None  # why it happened (loaded on demand)
    conversation_id: int | None = None  # FK to conversations table
    tags: list[str] = []
