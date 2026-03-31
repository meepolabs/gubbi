"""Domain exceptions for the storage layer.

These are raised by storage methods to signal "not found" conditions.
Using dedicated exception classes (rather than built-in KeyError/IndexError)
prevents accidental masking of infrastructure errors — e.g., a sqlite3.Row
dict access failure has the same type as a bare KeyError but a very different
meaning.
"""


class TopicNotFoundError(LookupError):
    """Raised when a topic path does not exist in the database."""


class ConversationNotFoundError(LookupError):
    """Raised when a conversation ID or title/topic pair does not exist."""
