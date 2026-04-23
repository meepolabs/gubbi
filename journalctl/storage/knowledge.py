"""File-based knowledge store — not backed by PostgreSQL."""

from __future__ import annotations

import re
from pathlib import Path

from journalctl.storage.constants import MAX_KNOWLEDGE_FILE_SIZE

_KNOWLEDGE_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")


def read(data_dir: Path, name: str) -> str:
    """Read a knowledge file (e.g. user-profile). File-based, not DB."""
    if not _KNOWLEDGE_NAME_PATTERN.match(name):
        raise ValueError(f"Invalid knowledge file name: {name!r}")
    base = (data_dir / "knowledge").resolve()
    path = (data_dir / "knowledge" / f"{name}.md").resolve()
    if not path.is_relative_to(base):
        raise ValueError(f"Knowledge file path escapes knowledge directory: {name!r}")
    if not path.exists():
        return ""
    if path.stat().st_size > MAX_KNOWLEDGE_FILE_SIZE:
        limit_mb = MAX_KNOWLEDGE_FILE_SIZE // (1024 * 1024)
        raise ValueError(f"Knowledge file '{name}' exceeds size limit ({limit_mb} MB)")
    return path.read_text(encoding="utf-8")
