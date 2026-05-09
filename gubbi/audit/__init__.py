"""Audit package -- DEC-061 / TASK-02.20.

Re-exports from :mod:`gubbi.audit.sql` (``record_audit``, ``Action``)
and :mod:`gubbi.audit.decorator` (``@audited`` and constants).
"""

from __future__ import annotations

# sql module -- record_audit + Action
from gubbi.audit.decorator import (
    ACTION_CONVERSATION_SAVED,  # noqa: F401
    ACTION_ENTRY_CREATED,  # noqa: F401
    ACTION_ENTRY_DELETED,  # noqa: F401
    ACTION_ENTRY_UPDATED,  # noqa: F401
    ACTION_TOPIC_CREATED,  # noqa: F401
    _extract_target_id,  # noqa: F401
    _result_is_success,  # noqa: F401
    audited,  # noqa: F401
)
from gubbi.audit.sql import Action, record_audit  # noqa: F401

__all__: list[str] = [
    "Action",
    "ACTION_CONVERSATION_SAVED",
    "ACTION_ENTRY_CREATED",
    "ACTION_ENTRY_DELETED",
    "ACTION_ENTRY_UPDATED",
    "ACTION_TOPIC_CREATED",
    "_extract_target_id",
    "_result_is_success",
    "audited",
    "record_audit",
]
