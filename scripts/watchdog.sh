#!/usr/bin/env bash
# Auto-restart crane vision main.py on RDKx5
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

while true; do
    echo "[watchdog] Starting scripts/main.py at $(date)"
    exit_code=0
    python3 scripts/main.py || exit_code=$?
    echo "[watchdog] main.py exited ($exit_code), restarting in 2s..."
    sleep 2
done
