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

# Fix ownership of app-internal directories
chown -R appuser:appuser /src /app/logs 2>/dev/null || true

# Drop privileges and run the CMD
exec gosu appuser "$@"
