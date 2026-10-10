"""Conversation data models: ConversationMeta, Message, and the shared role filter."""

from collections.abc import Iterable
from typing import Annotated

from pydantic import BaseModel, StringConstraints, field_validator

from gubbi.tools.constants import KEEP_ROLES
from gubbi.validation import validate_topic

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
    # True when title or summary could not be decrypted and the field holds the
    # soft-fail sentinel rather than plaintext. Always False on the loud MCP
    # path (which raises on corruption); set only by the soft-fail web path.
    decryption_failed: bool = False

    @field_validator("topic")
    @classmethod
    def check_topic(cls, v: str) -> str:
        """Validate conversation topic path against the shared topic-rules grammar."""
        return validate_topic(v)


class Message(BaseModel):
    """A single message in a conversation."""

    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str | None = None
    # True when the message content could not be decrypted and ``content`` holds
    # the soft-fail sentinel. Always False on the loud MCP path.
    decryption_failed: bool = False


def keep_conversation_roles(messages: Iterable[Message]) -> list[Message]:
    """Return the user/assistant turns with non-blank content, in input order.

    System prompts, tool calls, and tool results are infrastructure, not part of
    the conversation record, so every ingest path stores and hashes only what
    this returns.
    """
    return [m for m in messages if m.role in KEEP_ROLES and m.content.strip()]
