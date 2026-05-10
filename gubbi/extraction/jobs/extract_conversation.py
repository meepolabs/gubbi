"""Arq job: extract structured topics and entries from a saved conversation.

See ``extract_conversation`` function docstring for the CALLER CONTRACT
(security-critical).

Connection-split design (m-h5-h6):
  Phase 1 -- conn1: read-only load + early idempotency check.
  Phase 2 -- LLM phase: NO database connection held.
  Phase 3 -- conn2: persistence under an explicit nested SAVEPOINT transaction.

This ensures the database connection is released during the LLM calls
(which can take several seconds), so the pool is not exhausted when
max_jobs concurrent workers are all mid-LLM.
"""

from __future__ import annotations

import json
from typing import Any, cast
from uuid import UUID, uuid4

import asyncpg
import structlog
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

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


async def _check_idempotent(
    conn: asyncpg.Connection,
    conversation_id: int,
    user_id: str,
    log: structlog.stdlib.AsyncBoundLogger,
) -> bool:
    """Check whether this conversation was already processed.

    Returns True if *already* processed (caller short-circuits with skip
    result).  Returns False otherwise so processing can proceed.
    """
    already_processed = await conv_repo.get_processed_at(conn, conversation_id)
    if already_processed is not None:
        await log.info(
            "Conversation already processed, skipping",
            user_id=user_id,
            conversation_id=conversation_id,
        )
        return True
    return False


async def _load_conversation_for_extraction(
    conn: asyncpg.Connection,
    cipher: ContentCipher,
    conversation_id: int,
    user_id: str,
    log: structlog.stdlib.AsyncBoundLogger,
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
        await log.error(
            "Failed to load conversation",
            user_id=user_id,
            conversation_id=conversation_id,
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
    log: structlog.stdlib.AsyncBoundLogger,
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
        await log.error(
            "Categorization failed",
            user_id=user_id,
            conversation_id=conversation_id,
            exc_info=True,
        )
        raise

    raw_topic_path: str | None = categorization.topic_path
    topic_path: str | None = harden_llm_topic_path(raw_topic_path)

    if topic_path is None:
        await log.warning(
            "extraction: no usable topic_path from LLM, returning early",
            raw_topic_path_chars=len(raw_topic_path) if raw_topic_path else 0,
        )

    return categorization, topic_path


async def _mark_skipped_no_topic(
    pool: asyncpg.Pool,
    user_uuid: UUID,
    conversation_id: int,
) -> None:
    """Open a short-lived connection to mark a conversation as processed (no-topic path).

    Called when the LLM categorization returns a topic_path that hardening rejects.
    Uses a dedicated connection so the no-topic skip has exactly the same persistence
    contract as the full success path.
    """
    async with user_scoped_connection(pool, user_id=user_uuid) as conn:
        await conv_repo.mark_processed(conn, conversation_id)


async def _persist_extraction(
    conn: asyncpg.Connection,
    cipher: ContentCipher,
    user_uuid: UUID,
    conversation_id: int,
    topic_path: str,
    categorization: CategorizationResult,
    extracted_entries: list[ExtractedEntry],
    extraction_attempt_id: str,
    log: structlog.stdlib.AsyncBoundLogger,
) -> int:
    """Upsert the topic (if needed) and persist pre-extracted entries.

    All writes happen inside the SAVEPOINT that the caller wraps around this
    function. No LLM calls here -- entries are pre-extracted before conn2 is
    acquired.

    Returns the count of persisted entries.

    NOTE-m-h5-h6: entry_repo.append uses a plain INSERT with no ON CONFLICT
    clause.  The entries table has no uniqueness constraint on
    (conversation_id, topic_id, content) that would make ON CONFLICT DO NOTHING
    meaningful without a schema migration.  The conn2 second-idempotency-check
    (caller) prevents double-insert under normal retry storms.  A partial conn2
    failure mid-batch (e.g. after some entries are written but before
    mark_processed) will roll back via the SAVEPOINT, so partial state is
    never committed.  Retries after that are safe: the SAVEPOINT re-runs from
    the start of Phase 3.
    """
    # Topic upsert (get_id / create with race-tolerance).
    try:
        await topic_repo.get_id(conn, topic_path)
    except TopicNotFoundError:
        try:
            await topic_repo.create(conn, topic_path, title=categorization.topic_title)
        except ValueError as exc:
            if "already exists" in str(exc):
                await log.debug("Topic race on create, proceeding", error=str(exc))
            else:
                raise

    # Persist entries (append loop -- no LLM calls here).
    entries_created = 0
    for entry in extracted_entries:
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

    # Mark processed (WHERE processed_at IS NULL -- race-safe no-op if already set).
    await conv_repo.mark_processed(conn, conversation_id)

    # Audit row -- inside SAVEPOINT; rolled back on any failure above.
    await record_audit(
        conn,
        actor_type="user",
        actor_id=str(user_uuid),
        action=Action.CONVERSATION_EXTRACTED,
        target_type="conversation",
        target_kind="conversation",
        target_id=str(conversation_id),
        metadata={
            "via": "extraction-worker",
            "entries_created": entries_created,
            "topics_touched": 1,
            "extraction_attempt_id": extraction_attempt_id,
        },
    )

    return entries_created


async def _publish_progress(
    redis: Any,
    user_id: str,
    conversation_id: int,
    job_id: str,
    topic_path: str | None,
    entries_created: int,
    log: structlog.stdlib.AsyncBoundLogger,
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
        await log.warning(
            "Failed to publish extraction event to Redis",
            user_id=user_id,
            conversation_id=conversation_id,
            exc_info=True,
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def extract_conversation(
    ctx: ExtractionContext,
    conversation_id: int,
    user_id: str,
) -> dict[str, Any]:
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

    Connection-split design (m-h5-h6):
      Phase 1 -- conn1: read-only load + early idempotency check.
      Phase 2 -- LLM phase: no database connection held.
      Phase 3 -- conn2: persistence under explicit nested SAVEPOINT.

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

    # Bind an attempt-level correlation ID for trace correlation across the two DB spans.
    extraction_attempt_id = str(uuid4())

    log = cast(
        "structlog.stdlib.AsyncBoundLogger",
        logger.bind(
            component="extract_conversation",
            extraction_attempt_id=extraction_attempt_id,
        ),
    )
    user_uuid = user_id if isinstance(user_id, UUID) else UUID(user_id)

    # ------------------------------------------------------------------
    # Phase 1 -- conn1: read-only load + early idempotency check.
    # conn1 is released before any LLM call.
    # ------------------------------------------------------------------
    async with user_scoped_connection(pool, user_id=user_uuid) as conn1:
        if await _check_idempotent(conn1, conversation_id, user_id, log):
            return {
                "topic_path": None,
                "entries_created": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "skipped": True,
            }

        _meta, message_dicts, existing_topics = await _load_conversation_for_extraction(
            conn1, cipher, conversation_id, user_id, log
        )
    # conn1 released here -- pool slot returned before LLM calls.

    # ------------------------------------------------------------------
    # Phase 2 -- LLM phase: no database connection held.
    # ------------------------------------------------------------------
    categorization, topic_path = await _categorize_and_resolve_topic(
        extraction_service, message_dicts, existing_topics, user_id, conversation_id, log
    )

    if topic_path is None:
        # No usable topic -- mark processed via a short dedicated connection.
        await _mark_skipped_no_topic(pool, user_uuid, conversation_id)
        return {
            "topic_path": None,
            "entries_created": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "skipped": True,
        }

    # Extract entries while no DB connection is held.
    extracted_entries: list[ExtractedEntry] = []
    try:
        extracted_entries = await extraction_service.extract_entries(message_dicts, topic_path)
    except Exception:
        await log.error(
            "Entry extraction failed",
            user_id=user_id,
            conversation_id=conversation_id,
            exc_info=True,
        )
        raise

    # ------------------------------------------------------------------
    # Phase 3 -- conn2: persistence under explicit nested SAVEPOINT.
    # Second idempotency check here guards against a concurrent worker
    # that raced through Phase 2 while we were doing LLM calls.
    # ------------------------------------------------------------------
    async with user_scoped_connection(pool, user_id=user_uuid) as conn2:  # noqa: SIM117
        async with conn2.transaction():  # nested SAVEPOINT inside user_scoped_connection
            # Last-write-wins guard: if another worker committed first, bail.
            if await _check_idempotent(conn2, conversation_id, user_id, log):
                return {
                    "topic_path": None,
                    "entries_created": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "skipped": True,
                }

            entries_created = await _persist_extraction(
                conn2,
                cipher,
                user_uuid,
                conversation_id,
                topic_path,
                categorization,
                extracted_entries,
                extraction_attempt_id,
                log,
            )
    # conn2 released here -- SAVEPOINT committed atomically.

    # ------------------------------------------------------------------
    # Publish progress event (outside DB connection -- non-fatal).
    # ------------------------------------------------------------------
    job_id = ctx.get("job_id", "unknown")
    await _publish_progress(
        redis, user_id, conversation_id, str(job_id), topic_path, entries_created, log
    )

    await log.info(
        "Extraction complete",
        user_id=user_id,
        conversation_id=conversation_id,
        topic_path=topic_path,
        entries_created=entries_created,
    )

    return {
        "topic_path": topic_path,
        "entries_created": entries_created,
        "input_tokens": 0,
        "output_tokens": 0,
        "skipped": False,
    }
