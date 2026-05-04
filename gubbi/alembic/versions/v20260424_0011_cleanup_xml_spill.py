"""Marker migration -- companion script ran on bunsamosa 2026-04-30, then deleted.

Migration 0011 was originally written to clean up XML-spill data from
``entries.reasoning``, but that column was dropped by migration 0008.
The actual spill lived in ``entries.reasoning_encrypted`` (BYTEA,
AES-GCM via ContentCipher), which an alembic migration cannot touch
because it lacks the encryption key and runtime app context.

A companion one-shot script
``deployment/scripts/cleanup_encrypted_xml_spill.py`` was used to
decrypt, trim ``<parameter`` tails, and re-encrypt affected rows. It
ran on bunsamosa 2026-04-30 (matched 9 entries; audit_log carries 9
``cleanup_xml_spill_v2`` rows from that run). Then deleted from the
repo since bunsamosa was the only deployment.

If a future fresh deployment imports the bunsamosa data dump, the
spill is already cleaned at rest; nothing to re-run. If a fresh
deployment seeds entirely from current gubbi source-of-truth
(no legacy import), the spill never existed and there is nothing to
clean. So the script no longer has a use case in any going-forward
deploy.

Keeping this migration in the chain so revision IDs stay stable and
``alembic upgrade`` from any older snapshot still arrives at the
correct head.
"""

import logging

logger = logging.getLogger(__name__)

revision = "0011_cleanup_xml_spill"
down_revision = "0010_audit_log"
branch_labels = None
depends_on = None


def upgrade() -> None:
    logger.info(
        "Migration 0011 is a no-op marker: companion script "
        "deployment/scripts/cleanup_encrypted_xml_spill.py ran on "
        "bunsamosa 2026-04-30 and was deleted from the repo."
    )


def downgrade() -> None:
    pass
