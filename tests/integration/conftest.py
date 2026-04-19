"""Integration-scope conftest — re-exports tenant fixtures for pytest auto-discovery.

Fixtures defined in ``tests/fixtures/tenants.py`` are not auto-discovered
by pytest because that module is not a conftest. Importing them here,
at conftest scope, lets every test in ``tests/integration/`` request them
by parameter name without any module-level import (which would shadow
the fixture function with the same name and trigger ruff F811).
"""

from tests.fixtures.tenants import (  # noqa: F401 — re-exported for pytest discovery
    TenantSeed,
    seed_for,
    seeded_a,
    seeded_b,
    tenant_a,
    tenant_b,
)
