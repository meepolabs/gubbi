from __future__ import annotations

from typing import TYPE_CHECKING, NotRequired, TypedDict

if TYPE_CHECKING:
    import threading

    import asyncpg
    from gubbi_common.budget import BudgetHelper
    from redis.asyncio import ConnectionPool as RedisConnectionPool
    from redis.asyncio import Redis as RedisClient

    from gubbi.crypto.cipher import ContentCipher
    from gubbi.extraction.service import ExtractionService


class ExtractionContext(TypedDict):
    pool: asyncpg.Pool
    cipher: ContentCipher | None
    extraction_service: ExtractionService
    redis: RedisClient
    redis_pool: RedisConnectionPool
    health_thread: NotRequired[threading.Thread]
    job_id: NotRequired[str]
    budget_helper: NotRequired[BudgetHelper]
