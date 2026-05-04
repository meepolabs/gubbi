from __future__ import annotations

import threading
from typing import NotRequired, TypedDict

import asyncpg
from redis.asyncio import Redis as RedisClient

from gubbi.core.crypto import ContentCipher
from gubbi.extraction.service import ExtractionService


class ExtractionContext(TypedDict):
    pool: asyncpg.Pool
    cipher: ContentCipher | None
    extraction_service: ExtractionService
    redis: RedisClient
    health_thread: NotRequired[threading.Thread]
    job_id: NotRequired[str]
