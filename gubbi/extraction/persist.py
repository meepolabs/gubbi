"""Persistence of an extraction run: topic, deduped entries, embeddings, completion.

Every function here runs on the caller's connection inside the caller's
transaction, so a failure anywhere rolls the whole run back. Embeddings are
computed by the caller before the connection is acquired; nothing here does
CPU-bound work or calls a model.
"""

from __future__ import annotations

from dataclasses import dataclass
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

    The INSERT runs in its own savepoint: when a concurrent transaction creates
    the same path first, the unique violation rolls back only that savepoint,
    and the winner's id is read back on a transaction that is still usable.
    """
    try:
        return await topic_repo.get_id(conn, path), False
    except TopicNotFoundError:
        pass
    try:
        async with conn.transaction():
            topic_id = await topic_repo.create(conn, path, title=title)
    except topic_repo.TopicAlreadyExists:
        return await topic_repo.get_id(conn, path), False
    return topic_id, True


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
    """
    topic_id, topic_created = await ensure_topic(conn, run.topic_path, run.topic_title)
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
