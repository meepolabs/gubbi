"""Database seeding and context helpers for extraction job integration tests."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from datetime import date, datetime
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

from gubbi.extraction.service import ExtractionService
from gubbi.storage.embedding_service import EmbeddingService
from gubbi.storage.repositories import extraction_jobs, extraction_queue

if TYPE_CHECKING:
    from collections.abc import Sequence

    import asyncpg

    from gubbi.crypto.cipher import ContentCipher
    from gubbi.extraction.llm.provider import LLMProvider


@dataclass(frozen=True)
class SeedMessage:
    """A stored transcript message; ``timestamp`` None stores an undated message."""

    role: str
    content: str
    timestamp: datetime | None = None


@functools.cache
def embedding_service() -> EmbeddingService:
    """The real ONNX embedding service, loaded once per test session."""
    return EmbeddingService()


async def seed_conversation(
    admin_pool: asyncpg.Pool,
    cipher: ContentCipher,
    user_id: UUID,
    messages: Sequence[SeedMessage],
    *,
    platform: str | None = None,
    created_at: datetime | None = None,
) -> int:
    """Insert a conversation and its messages exactly as given, bypassing ingest filters."""
    suffix = uuid4().hex[:8]
    title_ct, title_nonce = cipher.encrypt("Seeded conversation")
    summary_ct, summary_nonce = cipher.encrypt("Seeded summary")
    async with admin_pool.acquire() as conn, conn.transaction():
        topic_id = await conn.fetchval(
            "INSERT INTO topics (path, title, description, user_id, created_at, updated_at)"
            " VALUES ($1, 'Inbox', '', $2, now(), now()) RETURNING id",
            f"inbox-{suffix}",
            user_id,
        )
        conv_id: int = await conn.fetchval(
            """
            INSERT INTO conversations
                (topic_id, user_id, title_encrypted, title_nonce, slug, source,
                 summary_encrypted, summary_nonce, tags, participants,
                 message_count, created_at, updated_at, json_path, search_vector,
                 platform, platform_id)
            VALUES ($1, $2, $3, $4, $5, 'extension_chatgpt', $6, $7, '{}', '{}',
                    $8, COALESCE($9, now()), now(), $10, to_tsvector('english', 'seed'),
                    $11, $12)
            RETURNING id
            """,
            topic_id,
            user_id,
            title_ct,
            title_nonce,
            f"seed-{suffix}",
            summary_ct,
            summary_nonce,
            len(messages),
            created_at,
            f"conversations_json/seed-{suffix}.json",
            platform,
            None if platform is None else f"platform-{suffix}",
        )
        for position, message in enumerate(messages):
            content_ct, content_nonce = cipher.encrypt(message.content)
            await conn.execute(
                """
                INSERT INTO messages
                    (conversation_id, user_id, role, timestamp, position,
                     content_encrypted, content_nonce, search_vector)
                VALUES ($1, $2, $3, $4, $5, $6, $7, to_tsvector('english', 'seed'))
                """,
                conv_id,
                user_id,
                message.role,
                message.timestamp,
                position,
                content_ct,
                content_nonce,
            )
    return int(conv_id)


async def seed_job(
    app_pool: asyncpg.Pool,
    user_id: UUID,
    conversation_id: int,
    period_start: date = date(2026, 5, 1),
    *,
    admitted: bool = True,
) -> UUID:
    """Insert a pending extraction job for the conversation, under the user's RLS scope.

    ``admitted`` charges it to ``period_start`` as the scheduler does before
    dispatch; only an admitted job can be claimed by a worker.
    """
    from gubbi_common.db.user_scoped import user_scoped_connection

    async with user_scoped_connection(app_pool, user_id=user_id) as conn:
        job_id = await extraction_jobs.create_pending(
            conn, user_id, conversation_id, "extension_chatgpt", period_start=period_start
        )
        if admitted:
            await extraction_queue.admit(conn, [job_id], period_start)
        return job_id


def job_ctx(app_pool: asyncpg.Pool, cipher: ContentCipher, provider: LLMProvider) -> Any:
    """Arq context for a real-database job run against ``provider``."""
    return {
        "pool": app_pool,
        "cipher": cipher,
        "extraction_service": ExtractionService(provider),
        "embedding_service": embedding_service(),
        "redis": AsyncMock(),
    }
