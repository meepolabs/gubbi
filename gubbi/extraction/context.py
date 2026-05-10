from __future__ import annotations

import threading
from typing import NotRequired, TypedDict

import asyncpg
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
