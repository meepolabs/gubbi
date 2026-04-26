"""Marker migration -- superseded by deployment/cleanup_encrypted_xml_spill.py.

Migration 0011 was originally written to clean up XML-spill data from
``entries.reasoning``, but that column was dropped by migration 0008.
The actual spill data lives in ``entries.reasoning_encrypted`` (BYTEA,
AES-GCM via ContentCipher), which an alembic migration has no access to
because it lacks the encryption key and the runtime app context needed to
construct a ContentCipher.

Operators should instead use::

    cd journalctl
    JOURNAL_ENCRYPTION_MASTER_KEY_V1="base64-key..." \\
        JOURNAL_DB_ADMIN_URL="postgresql://admin@.../journal" \\
        poetry run python deployment/cleanup_encrypted_xml_spill.py

to decrypt, trim, and re-encrypt affected rows.
"""

import logging

logger = logging.getLogger(__name__)

revision = "0011_cleanup_xml_spill"
down_revision = "0010_audit_log"
branch_labels = None
depends_on = None


def upgrade() -> None:
    logger.info(
        "Migration 0011 is a no-op: original cleanup target (entries.reasoning) "
        "was dropped by migration 0008, and the actual encrypted spill data "
        "must be cleaned via deployment/cleanup_encrypted_xml_spill.py "
        "which has access to ContentCipher."
    )


def downgrade() -> None:
    pass
