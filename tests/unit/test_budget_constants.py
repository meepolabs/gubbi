"""Unit test: PRE_CHARGE_CENTS cross-repo invariant.

gubbi and gubbi-cloud must agree on the pre-charge amount.
The gubbi venv cannot import gubbi-cloud directly, so this test
asserts the local constant equals the expected literal value (50).
If gubbi-cloud bumps PRE_CHARGE_CENTS, this test will fail until
gubbi/gubbi/budget/constants.py is updated in lockstep.

Cross-repo coupling: gubbi-cloud/gubbi_cloud/gateway/budget_middleware.py
"""

from __future__ import annotations

from gubbi.budget import PRE_CHARGE_CENTS


def test_pre_charge_cents_equals_fifty() -> None:
    """PRE_CHARGE_CENTS must equal 50 -- cross-repo invariant with gubbi-cloud."""
    assert PRE_CHARGE_CENTS == 50, (
        f"PRE_CHARGE_CENTS drifted: got {PRE_CHARGE_CENTS}, expected 50. "
        "Bump both gubbi and gubbi-cloud in lockstep when changing this value."
    )
