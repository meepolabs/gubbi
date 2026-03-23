#!/bin/bash
# Daily cron job: generate timeline files + git sync
# Install: 0 2 * * * /path/to/journalctl/scripts/daily_sync.sh
#
# This script runs OUTSIDE the server container.
# The server only reads/writes markdown + FTS5 index.
# Git and timeline generation happen here.

set -euo pipefail

JOURNAL_ROOT="${JOURNAL_ROOT:?Set JOURNAL_ROOT to your journal content path}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[$(date)] Starting daily sync..."

# 1. Generate static timeline files for MkDocs
if command -v python3 &>/dev/null; then
    python3 "$SCRIPT_DIR/generate_timeline.py" "$JOURNAL_ROOT" || {
        echo "[$(date)] Timeline generation failed, continuing with git sync"
    }
fi

# 2. Git sync
cd "$JOURNAL_ROOT"

if [ -d ".git" ]; then
    git add -A
    git diff --cached --quiet || {
        git commit -m "daily sync $(date +%Y-%m-%d)"
        echo "[$(date)] Committed changes"
    }
    git push origin main || {
        echo "[$(date)] Push failed, will retry tomorrow"
    }
else
    echo "[$(date)] No git repo at $JOURNAL_ROOT, skipping git sync"
fi

echo "[$(date)] Daily sync complete"
