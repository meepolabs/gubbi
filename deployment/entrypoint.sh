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
# Without this, both workers race to download simultaneously and one gets
# a corrupted archive. Running once here serializes the download.
gosu appuser python -c "
from mcp_memory_service.embeddings.onnx_embeddings import ONNXEmbeddingModel
ONNXEmbeddingModel()
" 2>&1 || true

# Drop privileges and run the CMD
exec gosu appuser "$@"
