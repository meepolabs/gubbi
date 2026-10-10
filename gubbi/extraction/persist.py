"""Persistence of an extraction run: topic, deduped entries, embeddings, completion.

Every function here runs on the caller's connection inside the caller's
transaction, so a failure anywhere rolls the whole run back. Embeddings are
computed by the caller before the connection is acquired; nothing here does
CPU-bound work or calls a model.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol

import structlog
from gubbi_common.audit.actions import Action
from gubbi_common.audit.targets import TargetKind

from gubbi.audit import record_audit
from gubbi.constants import EXTRACTION_VERSION
from gubbi.extraction.dedupe import DedupeAction, DedupeCandidate, DedupeDecision, classify
from gubbi.storage.exceptions import TopicNotFoundError
from gubbi.storage.repositories import conversations as conv_repo
from gubbi.storage.repositories import entries as entry_repo
from gubbi.storage.repositories import extraction_jobs
from gubbi.storage.repositories import topics as topic_repo
from gubbi.tools.constants import MAX_ENTRY_CONTENT_CHARS, MAX_ENTRY_REASONING_CHARS
from gubbi.validation import reject_tool_call_syntax, sanitize_freetext, sanitize_label

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    import asyncpg

    from gubbi.crypto.cipher import ContentCipher
    from gubbi.extraction.llm.provider import LLMUsage
    from gubbi.extraction.pipeline import ExtractedCandidate

__all__: list[str] = [
    "EmbeddingStore",
    "ExtractionToPersist",
    "PersistResult",
    "ensure_topic",
    "persist_extraction",
]

logger = structlog.get_logger(__name__)

_MAX_TOPIC_TITLE_CHARS = 100


class EmbeddingStore(Protocol):
    """Anything that stores a precomputed entry embedding on a connection."""

    async def save_by_vector(
        self, conn: asyncpg.Connection, entry_id: int, embedding: list[float]
    ) -> None:
        """Upsert ``embedding`` as the stored embedding of ``entry_id``."""
        ...


@dataclass(frozen=True)
class ExtractionToPersist:
    """Everything one extraction run writes, gathered before the connection is taken.

    ``embeddings[i]`` is the embedding of ``candidates[i].content``. ``job_id`` is
    None when the run has no job row to complete.
    """

    user_id: UUID
    conversation_id: int
    job_id: UUID | None
    extraction_attempt_id: str
    client: str | None
    topic_path: str
    topic_title: str
    candidates: Sequence[ExtractedCandidate]
    embeddings: Sequence[Sequence[float]]
    content_hash: str
    usage: LLMUsage
    llm_calls: int
    chunks: int
    cents_spent: int

    def __post_init__(self) -> None:
        """Reject a run whose embeddings do not pair one-to-one with its candidates."""
        if len(self.embeddings) != len(self.candidates):
            msg = (
                f"expected one embedding per candidate: {len(self.candidates)} candidates,"
                f" {len(self.embeddings)} embeddings"
            )
            raise ValueError(msg)


@dataclass(frozen=True)
class PersistResult:
    """What a run wrote: the topic, the inserted entries, and the stored entries it linked."""

    topic_id: int
    topic_created: bool
    entry_ids: tuple[int, ...]
    linked_entry_ids: tuple[int, ...]
    suppressed: int


async def ensure_topic(conn: asyncpg.Connection, path: str, title: str) -> tuple[int, bool]:
    """Return ``(topic_id, created)`` for ``path``, creating the topic if it is missing.

    ``title`` is sanitized as a label before it is stored; when nothing
    survives, the title is derived from the last segment of ``path``.

    The INSERT runs in its own savepoint: when a concurrent transaction creates
    the same path first, the unique violation rolls back only that savepoint,
    and the winner's id is read back on a transaction that is still usable.
    """
    try:
        return await topic_repo.get_id(conn, path), False
    except TopicNotFoundError:
        pass
    safe_title = sanitize_label(title, max_len=_MAX_TOPIC_TITLE_CHARS) or _title_from_path(path)
    try:
        async with conn.transaction():
            topic_id = await topic_repo.create(conn, path, title=safe_title)
    except topic_repo.TopicAlreadyExists:
        return await topic_repo.get_id(conn, path), False
    return topic_id, True


def _title_from_path(path: str) -> str:
    """``"cars/toyota-corolla"`` -> ``"Toyota corolla"``."""
    return path.rsplit("/", 1)[-1].replace("-", " ").capitalize()[:_MAX_TOPIC_TITLE_CHARS]


async def persist_extraction(
    conn: asyncpg.Connection,
    cipher: ContentCipher,
    embedding_service: EmbeddingStore,
    run: ExtractionToPersist,
) -> PersistResult:
    """Write one extraction run and mark its conversation and job complete.

    Candidates are deduped against each other and the user's stored entries of
    the same event day. A candidate repeating a stored entry inserts nothing and
    links that entry to the conversation if it has no source conversation yet; one
    repeating an earlier candidate inserts nothing. A near match is inserted with
    ``duplicate_of_entry_id`` naming the entry it resembles. Inserted entries carry
    the run's client and conversation and get their embedding stored.

    Candidate text is sanitized and capped as the MCP write path stores it
    before dedupe compares it; a candidate left empty or carrying tool-call
    syntax is dropped. Concurrent runs for the same user and event day are
    serialized from the neighbour read to commit, so two runs cannot both
    miss each other's entry and insert it twice.
    """
    run, dropped = _sanitized(run)
    if dropped:
        await logger.warning(
            "extraction_candidates_dropped",
            conversation_id=run.conversation_id,
            candidates_dropped=dropped,
        )
    topic_id, topic_created = await ensure_topic(conn, run.topic_path, run.topic_title)
    await entry_repo.lock_dedupe_days(
        conn, run.user_id, [candidate.entry_date for candidate in run.candidates], run.client
    )
    decisions = await _classify(conn, cipher, run)
    entry_ids, linked = await _write_candidates(conn, cipher, embedding_service, run, decisions)
    await conv_repo.mark_processed(conn, run.conversation_id)
    await conv_repo.set_extraction_state(
        conn, run.conversation_id, run.content_hash, EXTRACTION_VERSION
    )
    await _record_completion(conn, run, topic_created=topic_created, entries_created=len(entry_ids))
    result = PersistResult(
        topic_id=topic_id,
        topic_created=topic_created,
        entry_ids=tuple(entry_ids),
        linked_entry_ids=tuple(linked),
        suppressed=sum(d.action is DedupeAction.SUPPRESS for d in decisions),
    )
    await logger.info(
        "extraction_persisted",
        conversation_id=run.conversation_id,
        entries_created=len(result.entry_ids),
        entries_linked=len(result.linked_entry_ids),
        candidates_suppressed=result.suppressed,
        topic_created=topic_created,
    )
    return result


def _sanitized_candidate(candidate: ExtractedCandidate) -> ExtractedCandidate | None:
    """The candidate as it will be stored, or None when it must not be stored."""
    content = sanitize_freetext(candidate.content, max_len=MAX_ENTRY_CONTENT_CHARS)
    why = sanitize_freetext(candidate.why, max_len=MAX_ENTRY_REASONING_CHARS)
    if not content.strip():
        return None
    try:
        reject_tool_call_syntax(content)
        reject_tool_call_syntax(why)
    except ValueError:
        return None
    tags = tuple(dict.fromkeys(label for tag in candidate.tags if (label := sanitize_label(tag))))
    return replace(candidate, content=content, why=why, tags=tags)


def _sanitized(run: ExtractionToPersist) -> tuple[ExtractionToPersist, int]:
    """``run`` with every candidate sanitized, and how many were dropped as unstorable."""
    kept = [
        (safe, embedding)
        for candidate, embedding in zip(run.candidates, run.embeddings, strict=True)
        if (safe := _sanitized_candidate(candidate)) is not None
    ]
    sanitized = replace(
        run,
        candidates=tuple(candidate for candidate, _ in kept),
        embeddings=tuple(embedding for _, embedding in kept),
    )
    return sanitized, len(run.candidates) - len(kept)


async def _classify(
    conn: asyncpg.Connection, cipher: ContentCipher, run: ExtractionToPersist
) -> list[DedupeDecision]:
    neighbours = await entry_repo.list_dedupe_neighbours(
        conn,
        cipher,
        dates=[candidate.entry_date for candidate in run.candidates],
        client=run.client,
        exclude_conversation_id=run.conversation_id,
    )
    candidates = [
        DedupeCandidate(content=candidate.content, event_date=candidate.entry_date, embedding=emb)
        for candidate, emb in zip(run.candidates, run.embeddings, strict=True)
    ]
    return classify(candidates, neighbours, run.client)


async def _write_candidates(
    conn: asyncpg.Connection,
    cipher: ContentCipher,
    embedding_service: EmbeddingStore,
    run: ExtractionToPersist,
    decisions: Sequence[DedupeDecision],
) -> tuple[list[int], list[int]]:
    """Insert or link each candidate in input order; return (inserted ids, linked ids)."""
    inserted_by_index: dict[int, int] = {}
    linked: list[int] = []
    for index, decision in enumerate(decisions):
        if decision.action is DedupeAction.SUPPRESS:
            if decision.existing_entry_id is not None and await entry_repo.link_to_conversation(
                conn, decision.existing_entry_id, run.conversation_id
            ):
                linked.append(decision.existing_entry_id)
            continue
        duplicate_of = (
            inserted_by_index[decision.earlier_candidate]
            if decision.earlier_candidate is not None
            else decision.existing_entry_id
        )
        entry_id = await _insert_entry(conn, cipher, run, index, duplicate_of)
        await embedding_service.save_by_vector(conn, entry_id, list(run.embeddings[index]))
        await entry_repo.mark_indexed(conn, entry_id)
        inserted_by_index[index] = entry_id
    return list(inserted_by_index.values()), linked


async def _insert_entry(
    conn: asyncpg.Connection,
    cipher: ContentCipher,
    run: ExtractionToPersist,
    index: int,
    duplicate_of_entry_id: int | None,
) -> int:
    candidate = run.candidates[index]
    return await entry_repo.append(
        conn,
        cipher,
        topic=run.topic_path,
        content=candidate.content,
        reasoning=candidate.why or None,
        tags=candidate.tags,
        date=candidate.entry_date.isoformat(),
        client=run.client,
        duplicate_of_entry_id=duplicate_of_entry_id,
        source_conversation_id=run.conversation_id,
    )


async def _record_completion(
    conn: asyncpg.Connection,
    run: ExtractionToPersist,
    *,
    topic_created: bool,
    entries_created: int,
) -> None:
    await record_audit(
        conn,
        actor_type="user",
        actor_id=str(run.user_id),
        action=Action.CONVERSATION_EXTRACTED,
        target_kind=TargetKind.CONVERSATION,
        target_id=str(run.conversation_id),
        metadata={
            "via": "extraction-worker",
            "entries_created": entries_created,
            "topics_touched": 1,
            "extraction_attempt_id": run.extraction_attempt_id,
        },
    )
    if run.job_id is None:
        return
    updated = await extraction_jobs.mark_completed(
        conn,
        run.job_id,
        topics_created=1 if topic_created else 0,
        entries_created=entries_created,
        cents_spent=run.cents_spent,
        input_tokens=run.usage.input,
        output_tokens=run.usage.output,
        cached_input_tokens=run.usage.cached_input,
        reasoning_tokens=run.usage.reasoning,
        llm_calls=run.llm_calls,
        chunks=run.chunks,
    )
    if updated:
        await record_audit(
            conn,
            action="extraction_job.completed",
            actor_type="user",
            actor_id=str(run.user_id),
            target_kind=TargetKind.EXTRACTION_JOB,
            target_id=str(run.job_id),
            metadata={"conversation_id": run.conversation_id, "entries_created": entries_created},
        )
