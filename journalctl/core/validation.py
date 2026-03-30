"""Input validation and sanitization utilities."""

import re
from datetime import datetime

# Sanitization
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SAFE_LABEL = re.compile(r"[^a-zA-Z0-9 ._-]")


def sanitize_label(value: str, max_len: int = 50) -> str:
    """Sanitize a short label (source, tag) for safe use in frontmatter.

    Strips control characters, restricts to alphanumeric + space/dot/hyphen,
    and enforces a length limit.  Returns 'unknown' if the result is empty.
    """
    value = _CONTROL_CHARS.sub("", value).strip()
    value = _SAFE_LABEL.sub("", value)
    return value[:max_len] or "unknown"


def sanitize_freetext(value: str, max_len: int = 1_000_000) -> str:
    """Sanitize free-text content.

    Strips control characters (null bytes, escape codes, etc.) but
    preserves tabs, newlines, and carriage returns for markdown.
    """
    return _CONTROL_CHARS.sub("", value)[:max_len]


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
