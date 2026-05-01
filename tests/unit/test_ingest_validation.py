"""Unit tests for ingest request validation.

Tests the Pydantic models: IngestConversationRequest, ConversationPayload,
MessagePayload.  No DB required.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from journalctl.api.v1.ingest import (
    MAX_CONVERSATIONS_PER_REQUEST,
    ConversationPayload,
    IngestConversationRequest,
    MessagePayload,
)


def make_valid_conv(
    platform_id: str = "conv-001",
    title: str = "Test Conversation",
    msg_count: int = 2,
) -> dict:
    """Build a minimal valid conversation payload dict."""
    now = datetime.now(UTC)
    return {
        "platform": "chatgpt",
        "platform_id": platform_id,
        "title": title,
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
        "messages": [
            {"role": "user", "content": f"Message {i}", "timestamp": now.isoformat()}
            for i in range(msg_count)
        ],
    }


class TestIngestConversationRequest:
    """IngestConversationRequest model validation."""

    def test_valid_request(self) -> None:
        """A well-formed request should parse without error."""
        payload = {
            "source": "extension_chatgpt",
            "conversations": [make_valid_conv("conv-001")],
        }
        model = IngestConversationRequest.model_validate(payload)
        assert model.source == "extension_chatgpt"
        assert len(model.conversations) == 1
        assert model.conversations[0].platform_id == "conv-001"

    def test_too_many_conversations(self) -> None:
        """More than MAX_CONVERSATIONS_PER_REQUEST conversations must be rejected."""
        convs = [
            make_valid_conv(platform_id=f"conv-{i:03d}")
            for i in range(MAX_CONVERSATIONS_PER_REQUEST + 1)
        ]
        payload = {"source": "extension_claude", "conversations": convs}
        with pytest.raises(ValidationError, match="too_long"):
            IngestConversationRequest.model_validate(payload)

    def test_max_conversations_boundary(self) -> None:
        """MAX_CONVERSATIONS_PER_REQUEST conversations should be accepted."""
        convs = [
            make_valid_conv(platform_id=f"conv-{i:03d}")
            for i in range(MAX_CONVERSATIONS_PER_REQUEST)
        ]
        payload = {"source": "paste_memories", "conversations": convs}
        model = IngestConversationRequest.model_validate(payload)
        assert len(model.conversations) == MAX_CONVERSATIONS_PER_REQUEST

    def test_missing_source_field(self) -> None:
        """Missing source should raise a validation error."""
        payload = {"conversations": [make_valid_conv()]}
        with pytest.raises(ValidationError):
            IngestConversationRequest.model_validate(payload)

    def test_invalid_source_literal(self) -> None:
        """Source must be one of the allowed literals."""
        payload = {
            "source": "unknown_source",
            "conversations": [make_valid_conv()],
        }
        with pytest.raises(ValidationError):
            IngestConversationRequest.model_validate(payload)

    def test_empty_conversations_list(self) -> None:
        """An empty conversations list is valid (max_length allows 0)."""
        payload = {"source": "zip_upload", "conversations": []}
        model = IngestConversationRequest.model_validate(payload)
        assert len(model.conversations) == 0

    def test_missing_conversations_field(self) -> None:
        """Missing conversations should raise a validation error."""
        payload = {"source": "extension_chatgpt"}
        with pytest.raises(ValidationError):
            IngestConversationRequest.model_validate(payload)


class TestConversationPayload:
    """ConversationPayload model validation."""

    def test_valid_conversation(self) -> None:
        payload = {
            "platform": "claude",
            "platform_id": "claude-conv-abc",
            "title": "A chat with Claude",
            "created_at": datetime.now(UTC).isoformat(),
            "messages": [{"role": "user", "content": "Hello"}],
        }
        model = ConversationPayload.model_validate(payload)
        assert model.platform == "claude"
        assert model.title == "A chat with Claude"

    def test_empty_messages_rejected(self) -> None:
        """Conversation with no messages must be rejected."""
        payload = {
            "platform": "chatgpt",
            "platform_id": "empty-conv",
            "created_at": datetime.now(UTC).isoformat(),
            "messages": [],
        }
        with pytest.raises(ValidationError):
            ConversationPayload.model_validate(payload)

    def test_empty_title_defaults(self) -> None:
        """Empty title should default to empty string."""
        payload = {
            "platform": "chatgpt",
            "platform_id": "no-title",
            "created_at": datetime.now(UTC).isoformat(),
            "messages": [{"role": "user", "content": "Hello"}],
        }
        model = ConversationPayload.model_validate(payload)
        assert model.title == ""

    def test_invalid_platform_literal(self) -> None:
        """Platform must be 'chatgpt' or 'claude'."""
        payload = {
            "platform": "unknown",
            "platform_id": "x",
            "created_at": datetime.now(UTC).isoformat(),
            "messages": [{"role": "user", "content": "Hello"}],
        }
        with pytest.raises(ValidationError):
            ConversationPayload.model_validate(payload)

    def test_missing_platform_id(self) -> None:
        """platform_id is required."""
        payload = {
            "platform": "chatgpt",
            "created_at": datetime.now(UTC).isoformat(),
            "messages": [{"role": "user", "content": "Hello"}],
        }
        with pytest.raises(ValidationError):
            ConversationPayload.model_validate(payload)

    def test_missing_created_at(self) -> None:
        """created_at is required."""
        payload = {
            "platform": "chatgpt",
            "platform_id": "no-date",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        with pytest.raises(ValidationError):
            ConversationPayload.model_validate(payload)


class TestMessagePayload:
    """MessagePayload model validation."""

    def test_valid_message(self) -> None:
        payload = {"role": "user", "content": "Hello, world!"}
        model = MessagePayload.model_validate(payload)
        assert model.role == "user"
        assert model.content == "Hello, world!"
        assert model.timestamp is None

    def test_valid_message_with_timestamp(self) -> None:
        now = datetime.now(UTC)
        payload = {
            "role": "assistant",
            "content": "Hi there!",
            "timestamp": now.isoformat(),
        }
        model = MessagePayload.model_validate(payload)
        assert model.timestamp is not None

    def test_invalid_role(self) -> None:
        """Role must be user, assistant, or system."""
        payload = {"role": "admin", "content": "Hello"}
        with pytest.raises(ValidationError):
            MessagePayload.model_validate(payload)

    def test_system_role_accepted(self) -> None:
        """System role is a valid role."""
        payload = {"role": "system", "content": "You are a helpful assistant."}
        model = MessagePayload.model_validate(payload)
        assert model.role == "system"

    def test_missing_content(self) -> None:
        """Content is required."""
        payload = {"role": "user"}
        with pytest.raises(ValidationError):
            MessagePayload.model_validate(payload)
