"""AST scan / text-scan: ensure no bare ``frozenset({"journal"})`` literal
survives in ``journalctl/middleware/auth.py``.

The post-fix code uses ``frozenset({"journal:read","journal:write"})``
(the audit-equivalent default) only as a fallback constant; the regression
test asserts the BARE-WORD ``"journal"`` is no longer being used as a
single-token scope literal in the middleware auth module.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_AUTH_PATH = Path(__file__).resolve().parents[2] / "journalctl" / "middleware" / "auth.py"


@pytest.mark.unit
def test_no_bare_journal_scope_literal() -> None:
    """Assert no ``frozenset({"journal"})`` literal remains in auth.py source.

    The H-1 fix replaced all uses of ``frozenset({"journal"})`` with
    ``frozenset({"journal:read","journal:write"})`` or a config-driven
    frozenset. A bare ``"journal"`` as a single-token scope literal
    would mean the drift has re-introduced the old hardcoded scope.

    The Hydra Mode 3 path at line ~219 reads scopes from
    ``claims.scope.split()`` which may include ``"journal"`` as one of
    several scopes -- that path is correct and is NOT targeted by this
    guard. This test catches any literal ``frozenset({"journal"})``
    (with or without the extra spaces/newlines in the set literal).
    """
    src = _AUTH_PATH.read_text(encoding="utf-8")
    # Match the specific pattern frozenset({"journal"}) with optional whitespace
    assert 'frozenset({"journal"})' not in src, (
        'Bare frozenset({"journal"}) literal found in auth.py -- '
        "the H-1 fix should have replaced all uses with "
        'frozenset({"journal:read","journal:write"}) or a config-driven frozenset.'
    )
