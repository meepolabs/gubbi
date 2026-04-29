"""EmbeddingService — ONNX-based semantic embeddings backed by pgvector."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from opentelemetry import trace

if TYPE_CHECKING:
    import asyncpg

from journalctl.telemetry.attrs import _NS_PER_MS, _TRACER_NAME, SpanNames, safe_set_attributes

logger = logging.getLogger(__name__)

# ── Model constants ───────────────────────────────────────────────────────────
_MODEL_REPO = "sentence-transformers/all-MiniLM-L6-v2"
_CACHE_DIR = Path.home() / ".cache" / "journalctl" / "onnx_models"
_MODEL_DIR_NAME = "sentence-transformers_all-MiniLM-L6-v2"
_MAX_SEQ_LEN = 256
_EMBEDDING_DIM = 384

# HuggingFace direct download URLs
_HF_BASE = f"https://huggingface.co/{_MODEL_REPO}/resolve/main"
_TOKENIZER_URL = f"{_HF_BASE}/tokenizer.json"
_MODEL_URL = f"{_HF_BASE}/onnx/model_O2.onnx"
_MODEL_URL_FALLBACK = f"{_HF_BASE}/onnx/model_O1.onnx"


def _download_file(url: str, dest: Path) -> bool:
    """Download url to dest.  Returns True on success."""
    import requests  # noqa: PLC0415  # type: ignore[import-untyped]

    try:
        response = requests.get(url, timeout=120, stream=True)
        response.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info("Downloaded %s → %s", url, dest)
        return True
    except Exception as exc:
        logger.warning("Failed to download %s: %s", url, exc)
        return False


def _locate_or_download(cache_dir: Path = _CACHE_DIR) -> tuple[Path, Path]:
    """Return (model_path, tokenizer_path), downloading if necessary."""
    base = cache_dir / _MODEL_DIR_NAME

    # Look for cached model files (any .onnx file + tokenizer.json)
    model_candidates = list(base.glob("*.onnx")) + list(base.glob("**/*.onnx"))
    tokenizer_candidates = [base / "tokenizer.json"] + list(base.glob("**/tokenizer.json"))
    tokenizer_candidates = [p for p in tokenizer_candidates if p.exists()]

    model_path = next((p for p in model_candidates if p.exists()), None)
    tokenizer_path = tokenizer_candidates[0] if tokenizer_candidates else None

    if model_path and tokenizer_path:
        logger.info("Using cached ONNX model: %s", model_path)
        return model_path, tokenizer_path

    # Download tokenizer
    if not tokenizer_path:
        tokenizer_path = base / "tokenizer.json"
        if not _download_file(_TOKENIZER_URL, tokenizer_path):
            raise RuntimeError(f"Failed to download tokenizer from {_TOKENIZER_URL}")

    # Download model (try quantized first, fallback to full)
    if not model_path:
        model_path = base / "model_quantized.onnx"
        if not _download_file(_MODEL_URL, model_path):
            model_path = base / "model.onnx"
            if not _download_file(_MODEL_URL_FALLBACK, model_path):
                raise RuntimeError("Failed to download ONNX model from HuggingFace")

    return model_path, tokenizer_path


class EmbeddingService:
    """Thin ONNX wrapper over all-MiniLM-L6-v2 with pgvector persistence.

    encode() is synchronous (CPU-bound ONNX inference).
    store() and search() are async (touch PostgreSQL via asyncpg).
    """

    def __init__(self, model_cache_dir: Path | None = None) -> None:
        import onnxruntime as ort  # noqa: PLC0415
        from tokenizers import Tokenizer  # noqa: PLC0415

        cache_dir = model_cache_dir if model_cache_dir is not None else _CACHE_DIR
        model_path, tokenizer_path = _locate_or_download(cache_dir)

        sess_options = ort.SessionOptions()
        sess_options.inter_op_num_threads = 1
        sess_options.intra_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        self._tokenizer: Tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=_MAX_SEQ_LEN)  # noqa: S106
        self._tokenizer.enable_truncation(max_length=_MAX_SEQ_LEN)

        # Determine which inputs the model expects
        self._input_names = {inp.name for inp in self._session.get_inputs()}
        logger.info("EmbeddingService ready (model=%s)", model_path.name)

    def encode(self, text: str) -> list[float]:
        """Encode text → 384-dim normalised embedding vector (sync).

        Uses mean pooling over non-padding tokens followed by L2 normalisation,
        which is the standard approach for all-MiniLM-L6-v2.

        Records an ``embedding.encode`` OTel span with text_hash (sha256),
        text_len, and latency_ms. NO raw text is stored in span attributes.
        """
        span_name = SpanNames.EMBEDDING_ENCODE
        start_ns = time.monotonic_ns()
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        text_len = len(text)

        attrs: dict[str, Any] = {
            "text_hash": text_hash,
            "text_len": text_len,
        }

        with trace.get_tracer(_TRACER_NAME).start_as_current_span(span_name) as span:
            safe_set_attributes(span_name, span, attrs)

            encoding = self._tokenizer.encode(text)

            input_ids = np.array([encoding.ids], dtype=np.int64)
            attention_mask = np.array([encoding.attention_mask], dtype=np.int64)

            feed: dict[str, np.ndarray] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if "token_type_ids" in self._input_names:
                feed["token_type_ids"] = np.array([encoding.type_ids], dtype=np.int64)

            outputs = self._session.run(None, feed)
            token_embeddings: np.ndarray = outputs[0]  # [1, seq_len, 384]

            # Mean pool over non-padding positions
            mask = attention_mask.astype(np.float32)[:, :, np.newaxis]  # [1, seq_len, 1]
            summed = (token_embeddings * mask).sum(axis=1)  # [1, 384]
            counts = mask.sum(axis=1).clip(min=1e-9)  # [1, 1]
            mean_embedding = summed / counts  # [1, 384]

            # L2 normalise
            norm = np.linalg.norm(mean_embedding, axis=1, keepdims=True).clip(min=1e-9)
            normalised: np.ndarray = mean_embedding / norm  # [1, 384]

            result = normalised[0].tolist()

            latency_ms = (time.monotonic_ns() - start_ns) / _NS_PER_MS
            safe_set_attributes(
                span_name,
                span,
                {"latency_ms": round(latency_ms, 2)},
            )

        return result  # type: ignore[no-any-return]

    async def store_by_vector(
        self,
        conn: asyncpg.Connection,
        entry_id: int,
        embedding: list[float],
    ) -> None:
        """Upsert a pre-computed embedding vector.

        Callers should encode text via asyncio.to_thread(self.encode, text) before
        acquiring a DB connection, then pass the result here. This keeps the
        connection free during CPU-bound ONNX inference.
        """
        await conn.execute(
            """
            INSERT INTO entry_embeddings (entry_id, embedding, user_id, indexed_at)
            VALUES (
                $1,
                $2,
                (SELECT NULLIF(current_setting('app.current_user_id', true), '')::uuid),
                now()
            )
            ON CONFLICT (entry_id) DO UPDATE
                SET embedding  = excluded.embedding,
                    indexed_at = excluded.indexed_at,
                    user_id    = excluded.user_id
            """,
            entry_id,
            embedding,
        )

    async def store(
        self,
        conn: asyncpg.Connection,
        entry_id: int,
        text: str,
    ) -> None:
        """Upsert embedding for entry_id. Encodes text then calls store_by_vector.

        Prefer encode() + store_by_vector() when conn comes from a pool to avoid
        pinning a connection during CPU-bound ONNX inference.
        """
        embedding = await asyncio.to_thread(self.encode, text)
        await conn.execute(
            """
            INSERT INTO entry_embeddings (entry_id, embedding, user_id, indexed_at)
            VALUES (
                $1,
                $2,
                (SELECT NULLIF(current_setting('app.current_user_id', true), '')::uuid),
                now()
            )
            ON CONFLICT (entry_id) DO UPDATE
                SET embedding  = excluded.embedding,
                    indexed_at = excluded.indexed_at,
                    user_id    = excluded.user_id
            """,
            entry_id,
            embedding,
        )

    async def search_by_vector(
        self,
        conn: asyncpg.Connection,
        embedding: list[float],
        limit: int = 10,
        topic_prefix: str | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
    ) -> list[dict]:
        """Return top-k entries by a pre-computed embedding vector.

        Callers should encode text via asyncio.to_thread(self.encode, text) before
        acquiring a DB connection, then pass the result here. This keeps the
        connection free during CPU-bound ONNX inference.

        topic_prefix: filter to topics whose path starts with this prefix.
        date_from / date_to: filter by entry date (datetime.date objects).
        """
        params: list = [embedding, limit]  # $1=embedding, $2=limit
        where_clauses = ["e.deleted_at IS NULL"]

        if topic_prefix:
            escaped = topic_prefix.replace("!", "!!").replace("%", "!%").replace("_", "!_")
            params.append(escaped + "%")
            where_clauses.append(f"t.path LIKE ${len(params)} ESCAPE '!'")
        if date_from:
            params.append(date_from)
            where_clauses.append(f"e.date >= ${len(params)}")
        if date_to:
            params.append(date_to)
            where_clauses.append(f"e.date <= ${len(params)}")

        where = " AND ".join(where_clauses)
        rows = await conn.fetch(
            f"""
            SELECT
                e.id          AS entry_id,
                t.path        AS topic,
                e.date::text  AS date,
                1 - (ee.embedding <=> $1::vector) AS similarity
            FROM entry_embeddings ee
            JOIN entries e ON e.id = ee.entry_id
            JOIN topics  t ON t.id = e.topic_id
            WHERE {where}
            ORDER BY ee.embedding <=> $1::vector
            LIMIT $2
            """,
            *params,
        )
        return [dict(r) for r in rows]

    async def search(
        self,
        conn: asyncpg.Connection,
        text: str,
        limit: int = 10,
        topic_prefix: str | None = None,
    ) -> list[dict]:
        """Encode text then search. Convenience wrapper — holds conn during inference.

        Prefer encode() + search_by_vector() when conn comes from a pool to avoid
        pinning a connection during CPU-bound ONNX inference.
        """
        embedding = await asyncio.to_thread(self.encode, text)
        return await self.search_by_vector(conn, embedding, limit=limit, topic_prefix=topic_prefix)
