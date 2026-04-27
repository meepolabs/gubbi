"""Unit tests for storage/embedding_service.py.

Uses direct attribute injection to mock the ONNX session and tokenizer,
keeping tests fast and free of model download requirements.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest


@pytest.fixture
def mock_embedding_service() -> Any:
    """EmbeddingService with mocked ONNX session and tokenizer via attribute injection.

    Bypasses __init__ (which would try to download the ONNX model) and
    directly sets the internal attributes the methods use.
    """
    from journalctl.storage.embedding_service import EmbeddingService

    # Mock ONNX session that returns [batch=1, seq=8, dim=384] output
    mock_session = MagicMock()
    output = np.random.default_rng(42).random((1, 8, 384)).astype(np.float32)
    mock_session.run.return_value = [output]
    mock_session.get_inputs.return_value = [
        MagicMock(name="input_ids"),
        MagicMock(name="attention_mask"),
    ]

    # Mock tokenizer encoding
    encoding = MagicMock()
    encoding.ids = [1, 2, 3, 4, 0, 0, 0, 0]
    encoding.attention_mask = [1, 1, 1, 1, 0, 0, 0, 0]
    encoding.type_ids = [0] * 8
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = encoding

    svc = EmbeddingService.__new__(EmbeddingService)
    svc._session = mock_session
    svc._tokenizer = mock_tokenizer
    svc._input_names = {"input_ids", "attention_mask"}
    return svc


class TestEncode:
    """EmbeddingService.encode() — sync ONNX inference."""

    def test_returns_384_floats(self, mock_embedding_service: Any) -> None:
        result = mock_embedding_service.encode("Hello, world!")
        assert len(result) == 384
        assert all(isinstance(v, float) for v in result)

    def test_result_is_normalized(self, mock_embedding_service: Any) -> None:
        result = mock_embedding_service.encode("Some text")
        norm = sum(v**2 for v in result) ** 0.5
        assert abs(norm - 1.0) < 1e-4

    def test_empty_string_does_not_crash(self, mock_embedding_service: Any) -> None:
        result = mock_embedding_service.encode("")
        assert len(result) == 384

    def test_long_text_does_not_crash(self, mock_embedding_service: Any) -> None:
        result = mock_embedding_service.encode("word " * 500)
        assert len(result) == 384


class TestStoreAndSearch:
    """EmbeddingService.store() and search() — async DB operations."""

    async def test_store_calls_execute(self, mock_embedding_service: Any) -> None:
        conn = AsyncMock()
        await mock_embedding_service.store(conn, entry_id=42, text="Test content")
        conn.execute.assert_called_once()
        sql = conn.execute.call_args[0][0]
        assert "entry_embeddings" in sql
        assert "ON CONFLICT" in sql

    async def test_search_calls_fetch(self, mock_embedding_service: Any) -> None:
        conn = AsyncMock()
        conn.fetch.return_value = []
        result = await mock_embedding_service.search(conn, "test query", limit=5)
        conn.fetch.assert_called_once()
        assert result == []

    async def test_search_with_topic_prefix_passes_filter(
        self, mock_embedding_service: Any
    ) -> None:
        conn = AsyncMock()
        conn.fetch.return_value = []
        await mock_embedding_service.search(conn, "query", limit=3, topic_prefix="work/")
        call_args = conn.fetch.call_args[0]
        assert "t.path LIKE" in call_args[0]

    async def test_search_result_format(self, mock_embedding_service: Any) -> None:
        conn = AsyncMock()
        conn.fetch.return_value = [
            {
                "entry_id": 1,
                "topic": "work/acme",
                "date": "2025-06-01",
                "similarity": 0.95,
            }
        ]
        results = await mock_embedding_service.search(conn, "test", limit=5)
        assert len(results) == 1
        assert results[0]["entry_id"] == 1
        assert results[0]["topic"] == "work/acme"
        assert results[0]["similarity"] == 0.95
