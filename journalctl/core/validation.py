"""Input validation and sanitization utilities."""

import re
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

# Sanitization
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SAFE_LABEL = re.compile(r"[^a-zA-Z0-9 ._-]")


def sanitize_label(value: str, max_len: int = 50) -> str:
    """Sanitize a short label (source, tag) for safe use in frontmatter.

    Strips control characters, restricts to alphanumeric + space/dot/hyphen,
    and enforces a length limit. Returns empty string if nothing remains.
    """
    value = _CONTROL_CHARS.sub("", value).strip()
    value = _SAFE_LABEL.sub("", value)
    return value[:max_len]


def sanitize_freetext(value: str, max_len: int = 1_000_000) -> str:
    """Sanitize free-text content.

    Strips control characters (null bytes, escape codes, etc.) but
    preserves tabs, newlines, and carriage returns for markdown.
    """
    return _CONTROL_CHARS.sub("", value)[:max_len]


_TOOL_CALL_PATTERNS = (
    re.compile(r"<parameter\s+name="),
    re.compile(r"</parameter>"),
    re.compile(r"<parameter\b"),
)


def reject_tool_call_syntax(text: str) -> None:
    """Raise ValueError if text contains unparsed tool-call XML fragments."""
    for pattern in _TOOL_CALL_PATTERNS:
        if pattern.search(text):
            raise ValueError(
                "Input contains unparsed tool-call syntax; "
                "client likely failed to emit a valid tool call."
            )


# Validation patterns
# Matches 1-2 level paths: "work", "work/acme", "hobbies/my-project"
# Prevents path traversal, requires lowercase alphanumeric with hyphens
TOPIC_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*(?:/[a-z0-9]+(?:-[a-z0-9]+)*)?$")
# Matches titles 1-100 chars after sanitization
TITLE_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9 _-]{0,98}[a-zA-Z0-9]$|^[a-zA-Z0-9]$")
# Chars to strip from titles before validation
_TITLE_STRIP = re.compile(r"[^a-zA-Z0-9 _-]")
# Matches any non-alphanumeric sequence for slug conversion
SLUG_PATTERN = re.compile(r"[^a-z0-9]+")
# Date format: YYYY-MM-DD
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def validate_topic(value: str) -> str:
    """Validate topic path. Prevents path traversal. Auto-lowercases input.

    Valid: 'work/acme', 'hobbies', 'projects/my-app', 'Work/Acme' (lowercased)
    Invalid: '../etc/passwd', '/absolute', 'a/b/c/d'
    """
    value = value.lower()
    if not TOPIC_PATTERN.match(value):
        msg = (
            f"Invalid topic '{value}'. Must be 1-2 levels of "
            "lowercase alphanumeric with hyphens (e.g. 'work/acme')."
        )
        raise ValueError(msg)
    return value


def validate_title(value: str) -> str:
    """Sanitize and validate a conversation title.

    Strips leading/trailing whitespace, removes disallowed punctuation
    (e.g. colons, parentheses, apostrophes), and enforces 1-100 chars.
    """
    value = value.strip()
    value = _TITLE_STRIP.sub("", value).strip()
    # Collapse multiple spaces
    value = re.sub(r" {2,}", " ", value)
    value = value[:100]
    if not value or not TITLE_PATTERN.match(value):
        msg = (
            "Title could not be sanitized to a valid value. "
            "Use alphanumeric characters with spaces, hyphens, or underscores."
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
        raise ValueError(msg) from None
    return value


def local_today(timezone: str = "UTC") -> str:
    """Return today's date in the given IANA timezone as YYYY-MM-DD.

    Falls back to UTC if the timezone string is invalid.
    """
    try:
        tz = ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, KeyError):
        tz = ZoneInfo("UTC")
    return datetime.now(tz).date().isoformat()


def is_future_date(value: str, timezone: str = "UTC") -> bool:
    """Return True if the YYYY-MM-DD string is after today in the given timezone."""
    try:
        tz = ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, KeyError):
        tz = ZoneInfo("UTC")
    today = datetime.now(tz).date()
    return datetime.strptime(value, "%Y-%m-%d").date() > today


def slugify(text: str) -> str:
    """Convert text to a URL-safe slug for filenames."""
    slug = text.lower().strip()
    slug = SLUG_PATTERN.sub("-", slug)
    return slug.strip("-")
