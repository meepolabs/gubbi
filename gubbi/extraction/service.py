import pathlib

from gubbi.extraction.chunk_extraction import (
    CHUNK_CATEGORIZE_SCHEMA,
    CHUNK_ENTRIES_SCHEMA,
    CHUNK_EXTRACTION_SCHEMA,
    TRIAGE_SCHEMA,
    ChunkExtraction,
    TriageVerdict,
    parse_chunk_categorization,
    parse_chunk_entries,
    parse_chunk_extraction,
    parse_triage,
)
from gubbi.extraction.llm.provider import LLMMessage, LLMProvider, LLMResponse, LLMUsage


class ExtractionService:
    def __init__(self, llm_provider: LLMProvider) -> None:
        self._llm = llm_provider
        self._prompts_dir = pathlib.Path(__file__).resolve().parent / "prompts"

    async def extract_chunk(
        self,
        chunk_text: str,
        existing_topics: list[str],
        prior_entries_text: str,
        *,
        time_budget: float | None = None,
    ) -> ChunkExtraction:
        """Pick a topic and extract significant entries from one transcript chunk.

        ``chunk_text`` is the serialized chunk (``[index] [date] role: text``
        lines); ``prior_entries_text`` lists entries already extracted from
        earlier chunks so the model does not repeat them. An empty
        ``entries`` tuple means the chunk holds nothing worth recording.
        ``time_budget`` bounds the provider call in seconds, retries included.

        Raises:
            LLMSchemaError: the model output does not match the chunk schema.
        """
        system_prompt = self._read_prompt("extract_single.md")
        user_msg: LLMMessage = {
            "role": "user",
            "content": _chunk_user_content(chunk_text, existing_topics, prior_entries_text),
        }
        response = await self._llm.complete(
            [user_msg], system_prompt, CHUNK_EXTRACTION_SCHEMA, time_budget=time_budget
        )
        return parse_chunk_extraction(
            response.content,
            _usage_of(response),
            finish_reason=response.finish_reason,
            attempts=response.attempts,
            unmetered_attempts=response.unmetered_attempts,
        )

    async def categorize_chunk(
        self, chunk_text: str, existing_topics: list[str], *, time_budget: float | None = None
    ) -> ChunkExtraction:
        """Pick a topic and summarize the conversation from one transcript chunk.

        The first call of the two-call extraction; the result carries no entries.

        Raises:
            LLMSchemaError: the model output does not match the categorize schema.
        """
        system_prompt = self._read_prompt("categorize_chunk.md")
        user_msg: LLMMessage = {
            "role": "user",
            "content": _categorize_user_content(chunk_text, existing_topics),
        }
        response = await self._llm.complete(
            [user_msg], system_prompt, CHUNK_CATEGORIZE_SCHEMA, time_budget=time_budget
        )
        return parse_chunk_categorization(
            response.content,
            _usage_of(response),
            finish_reason=response.finish_reason,
            attempts=response.attempts,
            unmetered_attempts=response.unmetered_attempts,
        )

    async def extract_chunk_entries(
        self,
        chunk_text: str,
        topic_path: str,
        prior_entries_text: str,
        *,
        time_budget: float | None = None,
    ) -> ChunkExtraction:
        """Extract significant entries from one transcript chunk under a known topic.

        The per-chunk call of the two-call extraction; the result carries
        ``topic_path`` and an empty title and summary. Inputs are as for
        :meth:`extract_chunk`.

        Raises:
            LLMSchemaError: the model output does not match the entries schema.
        """
        system_prompt = self._read_prompt("extract_chunk_entries.md")
        user_msg: LLMMessage = {
            "role": "user",
            "content": _entries_user_content(chunk_text, topic_path, prior_entries_text),
        }
        response = await self._llm.complete(
            [user_msg], system_prompt, CHUNK_ENTRIES_SCHEMA, time_budget=time_budget
        )
        return parse_chunk_entries(
            response.content,
            _usage_of(response),
            topic_path=topic_path,
            finish_reason=response.finish_reason,
            attempts=response.attempts,
            unmetered_attempts=response.unmetered_attempts,
        )

    async def triage(self, chunk_text: str, *, time_budget: float | None = None) -> TriageVerdict:
        """Ask whether one transcript chunk holds anything worth recording.

        Raises:
            LLMSchemaError: the model output does not match the triage schema.
        """
        system_prompt = self._read_prompt("triage.md")
        user_msg: LLMMessage = {"role": "user", "content": f"{_TRANSCRIPT_HEADING}{chunk_text}"}
        response = await self._llm.complete(
            [user_msg], system_prompt, TRIAGE_SCHEMA, time_budget=time_budget
        )
        return parse_triage(
            response.content,
            _usage_of(response),
            finish_reason=response.finish_reason,
            attempts=response.attempts,
            unmetered_attempts=response.unmetered_attempts,
        )

    @property
    def provider_name(self) -> str:
        """Class name of the wrapped provider, for telemetry."""
        return type(self._llm).__name__

    @property
    def model_name(self) -> str:
        """Model the wrapped provider is configured for, or ``unknown``."""
        model = getattr(self._llm, "_model", None)
        return model if isinstance(model, str) and model else _UNKNOWN_MODEL

    def estimate_cost_cents(self, usage: LLMUsage) -> int:
        """Estimate the cost of ``usage`` in whole cents at the provider's prices."""
        return round(self._llm.estimate_cost_cents(usage.input, usage.output))

    def _read_prompt(self, filename: str) -> str:
        path = self._prompts_dir / filename
        return path.read_text(encoding="utf-8")


_UNKNOWN_MODEL = "unknown"
_NONE_LABEL = "(none)"
_TRANSCRIPT_HEADING = "## Transcript\n"


def _usage_of(response: LLMResponse) -> LLMUsage:
    return LLMUsage(
        input=response.input_tokens,
        output=response.output_tokens,
        cached_input=response.cached_input_tokens,
        reasoning=response.reasoning_tokens,
    )


def _topics_section(existing_topics: list[str]) -> str:
    topics = "\n".join(f"- {topic}" for topic in existing_topics) or _NONE_LABEL
    return f"## Existing topics\n{topics}\n\n"


def _prior_section(prior_entries_text: str) -> str:
    prior = prior_entries_text or _NONE_LABEL
    return f"## Entries already recorded from earlier parts of this conversation\n{prior}\n\n"


def _chunk_user_content(
    chunk_text: str, existing_topics: list[str], prior_entries_text: str
) -> str:
    return (
        _topics_section(existing_topics)
        + _prior_section(prior_entries_text)
        + f"{_TRANSCRIPT_HEADING}{chunk_text}"
    )


def _categorize_user_content(chunk_text: str, existing_topics: list[str]) -> str:
    return _topics_section(existing_topics) + f"{_TRANSCRIPT_HEADING}{chunk_text}"


def _entries_user_content(chunk_text: str, topic_path: str, prior_entries_text: str) -> str:
    return (
        f"## Topic\n{topic_path}\n\n"
        + _prior_section(prior_entries_text)
        + f"{_TRANSCRIPT_HEADING}{chunk_text}"
    )
