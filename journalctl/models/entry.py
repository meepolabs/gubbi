import re
from datetime import datetime

from pydantic import BaseModel, field_validator

# Validation patterns
# Matches 1-2 level paths: "work", "work/acme", "hobbies/my-project"
# Prevents path traversal, requires lowercase alphanumeric with hyphens
TOPIC_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*(?:/[a-z0-9]+(?:-[a-z0-9]+)*)?$")
# Matches titles 1-100 chars, alphanumeric with spaces/hyphens/underscores
TITLE_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9 _-]{0,98}[a-zA-Z0-9]$|^[a-zA-Z0-9]$")
# Matches any non-alphanumeric sequence for slug conversion
SLUG_PATTERN = re.compile(r"[^a-z0-9]+")
# Date format: YYYY-MM-DD
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def validate_topic(value: str) -> str:
    """Validate topic path. Prevents path traversal.

    Valid: 'work/acme', 'hobbies', 'projects/my-app'
    Invalid: '../etc/passwd', '/absolute', 'CAPS', 'a/b/c/d'
    """
    if not TOPIC_PATTERN.match(value):
        msg = (
            f"Invalid topic '{value}'. Must be 1-2 levels of "
            "lowercase alphanumeric with hyphens (e.g. 'work/acme')."
        )
        raise ValueError(msg)
    return value


def validate_title(value: str) -> str:
    """Validate conversation title."""
    if not TITLE_PATTERN.match(value):
        msg = (
            f"Invalid title '{value}'. Must be alphanumeric with "
            "spaces, hyphens, underscores (max 100 chars)."
        )
        raise ValueError(msg)
    return value


def validate_date(value: str) -> str:
    """Validate date string as YYYY-MM-DD.

    Also checks that the date is a real calendar date
    (rejects 2024-13-01 or 2024-02-30).
    """
    if not DATE_PATTERN.match(value):
        msg = f"Invalid date '{value}'. Must be YYYY-MM-DD format."
        raise ValueError(msg)
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        msg = f"Invalid calendar date '{value}'."
        raise
    return value


def slugify(text: str) -> str:
    """Convert text to a URL-safe slug for filenames."""
    slug = text.lower().strip()
    slug = SLUG_PATTERN.sub("-", slug)
    return slug.strip("-")


class TopicMeta(BaseModel):
    """YAML frontmatter for a topic file."""

    topic: str
    title: str
    description: str = ""
    tags: list[str] = []
    created: str  # YYYY-MM-DD
    updated: str  # YYYY-MM-DD
    entry_count: int = 0

    @field_validator("topic")
    @classmethod
    def check_topic(cls, v: str) -> str:
        return validate_topic(v)


class ConversationMeta(BaseModel):
    """YAML frontmatter for a conversation archive."""

    type: str = "conversation"
    source: str = "claude"
    title: str
    topic: str
    tags: list[str] = []
    created: str  # YYYY-MM-DD
    updated: str  # YYYY-MM-DD
    summary: str = ""
    participants: list[str] = []
    message_count: int = 0
    thread: str | None = None
    thread_seq: int | None = None

    @field_validator("topic")
    @classmethod
    def check_topic(cls, v: str) -> str:
        return validate_topic(v)


class Entry(BaseModel):
    """A single dated entry parsed from a topic file."""

    index: int  # 1-based position in the file
    date: str  # YYYY-MM-DD or YYYY-MM-DD HH:MM
    tags: list[str] = []
    content: str


class TopicInfo(BaseModel):
    """Summary info for journal_list_topics."""

    topic: str
    title: str
    description: str
    tags: list[str]
    entry_count: int
    created: str
    updated: str


class SearchResult(BaseModel):
    """A single search result from FTS5."""

    file_path: str
    doc_type: str  # 'topic' or 'conversation'
    topic: str
    title: str
    snippet: str
    rank: float
    date: str


class ConversationInfo(BaseModel):
    """Summary info for journal_list_conversations."""

    topic: str
    title: str
    filename: str
    summary: str
    source: str
    created: str
    updated: str
    message_count: int
    thread: str | None = None
    thread_seq: int | None = None


class Message(BaseModel):
    """A single message in a conversation."""

    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str | None = None


class TimelineEntry(BaseModel):
    """A single entry in a timeline view."""

    date: str
    topic: str
    summary: str
    source_path: str
    tags: list[str] = []
