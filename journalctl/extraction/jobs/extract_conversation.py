"""Arq job: extract structured topics and entries from a saved conversation.

See ``extract_conversation`` function docstring for the CALLER CONTRACT
(security-critical).
"""

from __future__ import annotations

import json
import logging
from typing import cast
from uuid import UUID

from gubbi_common.db.user_scoped import user_scoped_connection

from journalctl.audit import record_audit
from journalctl.core.crypto import ContentCipher
from journalctl.extraction.context import ExtractionContext
from journalctl.extraction.llm.provider import LLMMessage
from journalctl.storage.exceptions import TopicNotFoundError
from journalctl.storage.repositories import conversations as conv_repo
from journalctl.storage.repositories import entries as entry_repo
from journalctl.storage.repositories import topics as topic_repo

logger = logging.getLogger(__name__)


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

    # Convert from str (Arq deserializes JSON strings) to UUID for the
    # user_scoped_connection API.
    user_uuid = user_id if isinstance(user_id, UUID) else UUID(user_id)

    # --- Idempotency check ---
    async with user_scoped_connection(pool, user_id=user_uuid) as conn:
        already_processed = await conn.fetchval(
            "SELECT processed_at FROM conversations WHERE id = $1",
            conversation_id,
        )
        if already_processed is not None:
            log.info(
                "Conversation already processed, skipping",
                extra={
                    "user_id": user_id,
                    "conversation_id": conversation_id,
                },
            )
            return {
                "topic_path": None,
                "entries_created": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "skipped": True,
            }

        # --- Load conversation ---
        try:
            meta, messages, _total = await conv_repo.read_conversation_by_id(
                conn, cipher, conversation_id
            )
        except Exception:
            log.error(
                "Failed to load conversation",
                extra={
                    "user_id": user_id,
                    "conversation_id": conversation_id,
                },
                exc_info=True,
            )
            raise

        # --- Load existing topics for categorization context ---
        existing_topic_metas, _ = await topic_repo.list_all(conn)
        existing_topics = [t.topic for t in existing_topic_metas]

        # --- Categorize ---
        message_dicts: list[LLMMessage] = [{"role": m.role, "content": m.content} for m in messages]
        try:
            categorization = await extraction_service.categorize_conversation(
                message_dicts, existing_topics
            )
        except Exception:
            log.error(
                "Categorization failed",
                extra={
                    "user_id": user_id,
                    "conversation_id": conversation_id,
                },
                exc_info=True,
            )
            raise

        topic_path = categorization.topic_path

        # --- Upsert topic ---
        try:
            await topic_repo.get_id(conn, topic_path)
        except TopicNotFoundError:
            try:
                await topic_repo.create(
                    conn,
                    topic_path,
                    title=categorization.topic_title,
                )
            except ValueError as exc:
                if "already exists" in str(exc):
                    # Race: another worker created the topic between get_id
                    # and create. Topic exists, so proceed.
                    log.debug("Topic race on create, proceeding: %s", exc)
                else:
                    raise

        # --- Extract entries ---
        try:
            extracted = await extraction_service.extract_entries(message_dicts, topic_path)
        except Exception:
            log.error(
                "Entry extraction failed",
                extra={
                    "user_id": user_id,
                    "conversation_id": conversation_id,
                },
                exc_info=True,
            )
            raise

        # --- Persist entries ---
        entries_created = 0
        # Token tracking is not exposed from the service layer yet; plumbed later.
        total_input_tokens = 0
        total_output_tokens = 0

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

        # --- Mark conversation as processed ---
        await conn.execute(
            "UPDATE conversations SET processed_at = now() WHERE id = $1",
            conversation_id,
        )

        # --- Summary audit row (last DB write; committed only when everything succeeds) ---
        await record_audit(
            conn,
            actor_type="user",
            actor_id=str(user_uuid),
            action="conversation.extracted",
            target_kind="conversation",
            target_id=str(conversation_id),
            metadata={
                "via": "extraction-worker",
                "entries_created": entries_created,
                "topics_touched": 1,
            },
        )

    # --- Publish progress event ---
    job_id = ctx.get("job_id", "unknown")
    event = {
        "topic_path": topic_path,
        "entries_created": entries_created,
        "job_id": str(job_id),
        "conversation_id": conversation_id,
    }
    try:
        channel = f"extraction:user:{user_id}:job:{job_id}"
        await redis.publish(channel, json.dumps(event))
    except Exception:
        log.warning(
            "Failed to publish extraction event to Redis",
            extra={
                "user_id": user_id,
                "conversation_id": conversation_id,
            },
            exc_info=True,
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
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "skipped": False,
    }
