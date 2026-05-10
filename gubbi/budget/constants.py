"""Budget constants. Cross-repo invariant (see docstring)."""

from __future__ import annotations

from typing import Final

# MUST equal gubbi-cloud/gubbi_cloud/gateway/budget_middleware.py PRE_CHARGE_CENTS.
# Bumping this requires bumping both repos in lockstep. The cross-repo
# equality test in tests/unit/test_budget_constants.py guards drift.
PRE_CHARGE_CENTS: Final[int] = 50
