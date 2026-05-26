#!/bin/bash
set -euo pipefail

# Entrypoint script
# Starts as root, fixes permissions on mounted volumes, then drops
# to non-root appuser via gosu. The application never runs as root.

# Detect UID/GID of the mounted journal directory
MOUNT_UID=$(stat -c '%u' /app/journal 2>/dev/null || echo "1000")
MOUNT_GID=$(stat -c '%g' /app/journal 2>/dev/null || echo "1000")

# Update appuser to match the mount owner
if [ "$MOUNT_UID" != "0" ]; then
    usermod -u "$MOUNT_UID" appuser 2>/dev/null || true
    groupmod -g "$MOUNT_GID" appuser 2>/dev/null || true
fi

# Ensure journal subdirectories exist before appuser needs them
mkdir -p /app/journal/knowledge /app/journal/conversations_json /app/logs

# Fix ownership of app-internal directories (including ONNX model volume)
chown -R appuser:appuser /src /app/journal /app/logs /home/appuser/.cache 2>/dev/null || true

# Run database migrations as appuser BEFORE starting gunicorn.
# Alembic resolves the DSN from JOURNAL_DB_MIGRATION_URL (preferred) or
# JOURNAL_DB_ADMIN_URL (fallback); both are provided by the deployment
# secret store at deploy time. Idempotent: alembic skips already-applied revisions.
# Failure exits the entrypoint non-zero, which fails the container HEALTHCHECK
# and aborts the deploy after deploy_timeout.
echo "[entrypoint] running alembic upgrade head..."
gosu appuser python -m alembic -c alembic.ini upgrade head

# Verify DB invariants (GRANTs / RLS / policies / triggers / otel_ro)
# AFTER migrations succeed and BEFORE gunicorn starts. Same fail-fast
# guarantee: any invariant violation aborts the container, fails the
# HEALTHCHECK, and aborts the deploy. psql is available in the
# image (postgresql-client installed in the Dockerfile alongside gosu).
echo "[entrypoint] running verify-db-invariants.sh..."
gosu appuser /src/deployment/scripts/verify-db-invariants.sh

# Pre-download ONNX model as appuser before gunicorn workers start.
# Without --preload, each worker would try to download concurrently.
# Running EmbeddingService() here serializes the download to disk cache
# so all workers find the model already present on startup.
# Exit with non-zero status on failure so Docker can restart the container
# rather than starting a degraded server with no embedding capability.
gosu appuser python -c "
from gubbi.storage.embedding_service import EmbeddingService
EmbeddingService()
" 2>&1

# Drop the migration-only superuser DSN from the runtime env so long-lived
# gunicorn workers cannot read it from os.environ. Migrations + verify are
# both done by this point; gunicorn only needs JOURNAL_DB_ADMIN_URL +
# JOURNAL_DB_APP_URL. Defense-in-depth against in-process RCE escalating to
# total DB ownership via the `journal` superuser.
unset JOURNAL_DB_MIGRATION_URL

# Drop privileges and run the CMD
exec gosu appuser "$@"
