"""Audit package.

Re-exports from :mod:`gubbi.audit.sql` (``record_audit``, ``Action``)
and :mod:`gubbi.audit.decorator` (``@audited`` and constants).
"""

from __future__ import annotations

# sql module -- record_audit + Action
from gubbi.audit.decorator import (
    ACTION_CONVERSATION_SAVED,
    ACTION_ENTRY_CREATED,
    ACTION_ENTRY_DELETED,
    ACTION_ENTRY_UPDATED,
    ACTION_TOPIC_CREATED,
    _extract_target_id,
    _result_is_success,
    audited,
)
from gubbi.audit.sql import Action, record_audit

__all__: list[str] = [
    "ACTION_CONVERSATION_SAVED",
    "ACTION_ENTRY_CREATED",
    "ACTION_ENTRY_DELETED",
    "ACTION_ENTRY_UPDATED",
    "ACTION_TOPIC_CREATED",
    "Action",
    "_extract_target_id",
    "_result_is_success",
    "audited",
    "record_audit",
]
