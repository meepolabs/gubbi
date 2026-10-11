from __future__ import annotations

from typing import TYPE_CHECKING, NotRequired, TypedDict

if TYPE_CHECKING:
    import asyncio
    import threading

    import asyncpg
    from redis.asyncio import ConnectionPool as RedisConnectionPool
    from redis.asyncio import Redis as RedisClient

    from gubbi.crypto.cipher import ContentCipher
    from gubbi.extraction.service import ExtractionService
    from gubbi.storage.embedding_service import EmbeddingService


class ExtractionContext(TypedDict):
    pool: asyncpg.Pool
    cipher: ContentCipher | None
    extraction_service: ExtractionService
    embedding_service: EmbeddingService
    redis: RedisClient
    redis_pool: RedisConnectionPool
    health_thread: NotRequired[threading.Thread]
    job_id: NotRequired[str]
    cancel_cleanups: NotRequired[set[asyncio.Task[None]]]
