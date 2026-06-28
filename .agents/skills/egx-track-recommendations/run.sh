#!/bin/bash
# EGX Track Recommendations — monitoring + JSON processing
#
# This skill is triggered by a Base44 automation at 9:15 AM Cairo time.
# It runs monitor.py which:
#   1. Checks if today's GitHub Actions workflow succeeded
#   2. If succeeded: clones repo, reads recommendations JSON, outputs it
#   3. If failed: sends Telegram alert + triggers workflow_dispatch retry
#   4. Waits 20 min, checks retry — sends final alert if still failed
#
# Usage: run.sh [json_date_YYYY-MM-DD]  (date is optional, for backward compat)
#
# Environment variables (must be set in sandbox):
#   GITHUB_TOKEN       — GitHub PAT
#   TELEGRAM_BOT_TOKEN — Telegram bot token
#   TELEGRAM_CHAT_ID   — Telegram chat ID (default: 7534010234)

set -euo pipefail

SCRIPT_DIR="/app/egx-bot"

echo "=== EGX Track & Monitor — $(TZ='Africa/Cairo' date '+%Y-%m-%d %H:%M') Cairo ==="

# Run the monitoring script
# It handles everything: GitHub check, Telegram alerts, retry, JSON processing
python3 "${SCRIPT_DIR}/monitor.py" 2>&1

# Exit code from monitor.py
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Monitoring completed successfully"
else
    echo "⚠️ Monitoring completed with exit code $EXIT_CODE"
fi

exit 0
