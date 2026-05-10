"""Arq job: extract structured topics and entries from a saved conversation.

See ``extract_conversation`` function docstring for the CALLER CONTRACT
(security-critical).
"""

from __future__ import annotations

import json
import logging
from typing import Any, cast
from uuid import UUID

import asyncpg
from gubbi_common.audit.actions import Action
from gubbi_common.db.user_scoped import user_scoped_connection

from gubbi.audit import record_audit
from gubbi.crypto.cipher import ContentCipher
from gubbi.extraction.context import ExtractionContext
from gubbi.extraction.llm.provider import LLMMessage
from gubbi.extraction.service import CategorizationResult, ExtractedEntry, ExtractionService
from gubbi.storage.exceptions import TopicNotFoundError
from gubbi.storage.repositories import conversations as conv_repo
from gubbi.storage.repositories import entries as entry_repo
from gubbi.storage.repositories import topics as topic_repo
from gubbi.validation import harden_llm_topic_path

__all__: list[str] = ["extract_conversation"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


async def _check_idempotent(
    conn: asyncpg.Connection,
    conversation_id: int,
    user_id: str,
    log: logging.Logger,
) -> bool:
    """Check whether this conversation was already processed.

    Returns True if *already* processed (caller short-circuits with skip
    result).  Returns False otherwise so processing can proceed.
    """
    already_processed = await conv_repo.get_processed_at(conn, conversation_id)
    if already_processed is not None:
        log.info(
            "Conversation already processed, skipping",
            extra={"user_id": user_id, "conversation_id": conversation_id},
        )
        return True
    return False


async def _load_conversation_for_extraction(
    conn: asyncpg.Connection,
    cipher: ContentCipher,
    conversation_id: int,
    user_id: str,
    log: logging.Logger,
) -> tuple[Any, list[LLMMessage], list[str]]:
    """Load a conversation + its messages and existing topics.

    Returns ``(meta, message_dicts, existing_topics)`` where
    *message_dicts* is the list-of-dicts serialization needed by the LLM
    service, and *existing_topics* contains the raw topic strings.

    On load failure logs an error with extras and re-raises.
    """
    try:
        meta, messages, _total = await conv_repo.read_conversation_by_id(
            conn, cipher, conversation_id
        )
    except Exception:
        log.error(
            "Failed to load conversation",
            extra={"user_id": user_id, "conversation_id": conversation_id},
            exc_info=True,
        )
        raise

    existing_topic_metas, _ = await topic_repo.list_all(conn)
    existing_topics = [t.topic for t in existing_topic_metas]

    message_dicts: list[LLMMessage] = [{"role": m.role, "content": m.content} for m in messages]

    return meta, message_dicts, existing_topics


async def _categorize_and_resolve_topic(
    extraction_service: ExtractionService,
    message_dicts: list[LLMMessage],
    existing_topics: list[str],
    user_id: str,
    conversation_id: int,
    log: logging.Logger,
) -> tuple[CategorizationResult, str | None]:
    """Categorise the conversation and produce a hardened topic path.

    Returns ``(categorization, topic_path)`` -- *topic_path* may be None if
    hardening rejects the LLM output (non-fatal skip signal).

    On categorization failure logs an error with extras and re-raises.
    """
    try:
        categorization = await extraction_service.categorize_conversation(
            message_dicts, existing_topics
        )
    except Exception:
        log.error(
            "Categorization failed",
            extra={"user_id": user_id, "conversation_id": conversation_id},
            exc_info=True,
        )
        raise

    raw_topic_path: str | None = categorization.topic_path
    topic_path: str | None = harden_llm_topic_path(raw_topic_path)

    if topic_path is None:
        log.warning(
            "extraction: no usable topic_path from LLM, returning early (%d chars)",
            len(raw_topic_path) if raw_topic_path else 0,
        )

    return categorization, topic_path


async def _persist_entries(
    conn: asyncpg.Connection,
    cipher: ContentCipher,
    extraction_service: ExtractionService,
    message_dicts: list[LLMMessage],
    topic_path: str | None,
    categorization: CategorizationResult,
    user_id: str,
    conversation_id: int,
    log: logging.Logger,
) -> int:
    """Upsert the topic (if needed), extract entries, and persist them.

    Handles topic create with race-tolerance, entry extraction via the LLM
    service, then appends each extracted entry into the journal.

    Returns the count of persisted entries.  On extraction failure logs an
    error with extras and re-raises so that ``mark_processed`` is *not*
    written (fail-closed).
    """
    # Topic upsert (get_id / create with race-tolerance)
    if topic_path:
        try:
            await topic_repo.get_id(conn, topic_path)
        except TopicNotFoundError:
            assert topic_path  # noqa: S101 -- narrowed by truthiness guard above
            try:
                await topic_repo.create(conn, topic_path, title=categorization.topic_title)
            except ValueError as exc:
                if "already exists" in str(exc):
                    log.debug("Topic race on create, proceeding: %s", exc)
                else:
                    raise

    # Extract entries
    extracted: list[ExtractedEntry] = []
    try:
        if topic_path:
            extracted = await extraction_service.extract_entries(message_dicts, topic_path)
    except Exception:
        log.error(
            "Entry extraction failed",
            extra={"user_id": user_id, "conversation_id": conversation_id},
            exc_info=True,
        )
        raise

    # Persist entries (append loop)
    entries_created = 0
    assert topic_path  # noqa: S101 -- truthy topic required for entry persistence
    for entry in extracted:
        await entry_repo.append(
            conn,
            cipher,
            topic=topic_path,
            content=entry.content,
            reasoning=entry.reasoning,
            tags=entry.tags,
            date=entry.entry_date,
        )
        entries_created += 1

    return entries_created


async def _publish_progress(
    redis: Any,
    user_id: str,
    conversation_id: int,
    job_id: str,
    topic_path: str | None,
    entries_created: int,
    log: logging.Logger,
) -> None:
    """Publish extraction completion event on the user's Redis channel.

    Redis publish failure is non-fatal: logs a warning and returns silently.
    """
    event = {
        "topic_path": topic_path,
        "entries_created": entries_created,
        "job_id": job_id,
        "conversation_id": conversation_id,
    }
    try:
        channel = f"extraction:user:{user_id}:job:{job_id}"
        await redis.publish(channel, json.dumps(event))
    except Exception:
        log.warning(
            "Failed to publish extraction event to Redis",
            extra={"user_id": user_id, "conversation_id": conversation_id},
            exc_info=True,
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def extract_conversation(
    ctx: ExtractionContext,
    conversation_id: int,
    user_id: str,
) -> dict:
    """Arq job: categorize a conversation and write structured journal entries.

    NOTE: conversation_id is ``int`` (DB integer primary key) even though the
    original spec said ``UUID`` -- the conversations table uses integer PKs.

    CALLER CONTRACT (security-critical):
    The caller MUST authenticate that ``user_id`` owns ``conversation_id``
    BEFORE enqueueing this job. Today this is the API endpoint that
    enqueues extraction (POST /api/v1/extraction) -- it derives user_id
    from the authenticated request and only enqueues against
    conversations that belong to that user (verified via RLS-scoped
    SELECT).

    The job runs under user_scoped_connection(user_id=...), so all reads
    and writes are RLS-protected. A wrong user_id at enqueue time will:
      - Read the wrong tenant's conversation (RLS blocks; results in 0
        rows; the job loads an empty conversation and returns).
      - Or, if a future caller bypasses the API path with admin pool:
        could read across tenants. Don't bypass the API path.

    See llm_context/audit_contract.md for actor_type semantics on the
    summary audit row this job produces.

    Args:
        ctx: Arq worker context (pool, cipher, extraction_service, redis
            injected by on_startup).
        conversation_id: Database integer ID of the conversation to process.
        user_id: UUID string of the owning user (used for RLS scoping and
            pub/sub channel).

    Returns:
        Summary dict with topic_path, entries_created, input_tokens (0 until
        service layer exposes token counts), output_tokens (same), and skipped
        (bool, True if idempotency check short-circuited).
    """
    pool = ctx["pool"]
    cipher = cast(ContentCipher, ctx["cipher"])
    extraction_service = ctx["extraction_service"]
    redis = ctx["redis"]

    log = logger.getChild("extract_conversation")
    user_uuid = user_id if isinstance(user_id, UUID) else UUID(user_id)

    # --- Idempotency / full pipeline ---
    async with user_scoped_connection(pool, user_id=user_uuid) as conn:
        # Skip if already processed.
        if await _check_idempotent(conn, conversation_id, user_id, log):
            return {
                "topic_path": None,
                "entries_created": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "skipped": True,
            }

        # Load conversation + messages + existing topics.
        _meta, message_dicts, existing_topics = await _load_conversation_for_extraction(
            conn, cipher, conversation_id, user_id, log
        )

        # Categorize & harden topic path.
        categorization, topic_path = await _categorize_and_resolve_topic(
            extraction_service, message_dicts, existing_topics, user_id, conversation_id, log
        )

        # If hardening rejected the topic, mark and skip (no entries).
        if topic_path is None:
            await conv_repo.mark_processed(conn, conversation_id)
            return {
                "topic_path": None,
                "entries_created": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "skipped": True,
            }

        # Persist: topic upsert + entry extraction + write.
        entries_created = await _persist_entries(
            conn,
            cipher,
            extraction_service,
            message_dicts,
            topic_path,
            categorization,
            user_id,
            conversation_id,
            log,
        )

        # Mark processed (committed with everything above).
        await conv_repo.mark_processed(conn, conversation_id)

        # Audit row -- last DB write; committed only when everything succeeds.
        await record_audit(
            conn,
            actor_type="user",
            actor_id=str(user_uuid),
            action=Action.CONVERSATION_EXTRACTED,
            target_kind="conversation",
            target_id=str(conversation_id),
            metadata={
                "via": "extraction-worker",
                "entries_created": entries_created,
                "topics_touched": 1,
            },
        )

    # --- Publish progress event (outside DB connection) ---
    job_id = ctx.get("job_id", "unknown")
    await _publish_progress(
        redis, user_id, conversation_id, str(job_id), topic_path, entries_created, log
    )

    log.info(
        "Extraction complete",
        extra={
            "user_id": user_id,
            "conversation_id": conversation_id,
            "topic_path": topic_path,
            "entries_created": entries_created,
        },
    )

    return {
        "topic_path": topic_path,
        "entries_created": entries_created,
        "input_tokens": 0,
        "output_tokens": 0,
        "skipped": False,
    }
