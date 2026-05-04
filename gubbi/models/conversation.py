"""Conversation data models: ConversationMeta and Message."""

from typing import Annotated

from pydantic import BaseModel, StringConstraints, field_validator

from gubbi.core.validation import validate_topic

Tag128 = Annotated[str, StringConstraints(max_length=128)]


class ConversationMeta(BaseModel):
    """Metadata for a conversation archive."""

    id: int | None = None  # stable DB row ID
    source: str = "claude"
    title: str
    topic: str
    tags: list[Tag128] = []
    created: str  # YYYY-MM-DD
    updated: str  # YYYY-MM-DD
    summary: str = ""
    participants: list[str] = []
    message_count: int = 0

    @field_validator("topic")
    @classmethod
    def check_topic(cls, v: str) -> str:
        return validate_topic(v)


class Message(BaseModel):
    """A single message in a conversation."""

    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str | None = None
