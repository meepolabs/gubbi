#!/bin/bash
set -e

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

# Fix ownership of app-internal directories (including ONNX model volume)
chown -R appuser:appuser /src /app/logs /home/appuser/.cache 2>/dev/null || true

# Pre-download ONNX model as appuser before gunicorn workers start.
# Without --preload, each worker would try to download concurrently.
# Running EmbeddingService() here serializes the download to disk cache
# so all workers find the model already present on startup.
# Exit with non-zero status on failure so Docker can restart the container
# rather than starting a degraded server with no embedding capability.
gosu appuser python -c "
from journalctl.storage.embedding_service import EmbeddingService
EmbeddingService()
" 2>&1

# Drop privileges and run the CMD
exec gosu appuser "$@"
