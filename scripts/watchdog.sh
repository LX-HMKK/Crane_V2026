#!/usr/bin/env bash
# Auto-restart crane vision main_arm64.py on RDKx5
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

while true; do
    echo "[watchdog] Starting scripts/main_arm64.py at $(date)"
    if python3 scripts/main_arm64.py; then
        echo "[watchdog] main_arm64.py exited normally; stopping watchdog."
        exit 0
    else
        exit_code=$?
    fi
    echo "[watchdog] main_arm64.py exited ($exit_code), restarting in 2s..."
    sleep 2
done
