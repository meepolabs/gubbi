"""Journal-data REST endpoints under ``/api/v1``.

This package holds the read/write REST surface for journal resources
(topics, entries, search, timeline, conversations, stats). It is named
``web`` to avoid clashing with the storage repository modules
(``gubbi/storage/repositories/{topics,entries,...}.py``).

Shared conventions every resource router reuses live in sibling modules:

* ``schemas`` -- the generic ``PaginatedList`` base and the
  ``decryption_failed`` per-item convention.
* ``responses`` -- ``private_no_store_response`` so any endpoint returning
  decrypted content sets the right ``Cache-Control`` header in one place.
* ``decryption`` -- ``decrypt_field`` maps a row + cipher to
  ``(value, decryption_failed)``, surfacing the ``[decryption failed]``
  sentinel instead of raising.
* ``pagination`` -- the shared ``OffsetQuery`` param; each router declares its
  own ``LimitParam`` cap (over-max -> 422 via Pydantic ``Query`` constraints).
* ``errors`` -- ``topic_not_found`` mapping to FastAPI's default 404 envelope.

A new resource router plugs in by: importing ``PaginatedList`` for its list
shape, ``private_no_store_response`` when it returns decrypted content,
``decrypt_field`` for any encrypted column, the ``pagination`` builders for
its ``limit`` / ``offset`` params, and the ``errors`` helpers for 404 mapping.
See ``topics`` for the reference implementation.
"""

from __future__ import annotations

__all__: list[str] = []
