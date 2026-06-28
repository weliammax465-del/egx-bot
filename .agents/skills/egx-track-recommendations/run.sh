#!/bin/bash
# EGX Track Recommendations — processes daily recommendations JSON from GitHub
# and saves to RecommendationHistory entity for performance tracking.
#
# Usage: run.sh [json_date_YYYY-MM-DD]
# If no date given, uses today's date (Cairo timezone).

set -euo pipefail

REPO_URL="https://github.com/weliammax465-del/egx-bot.git"
CLONE_DIR="/tmp/egx-bot-track"
TODAY=$(TZ="Africa/Cairo" date +%Y-%m-%d)
TARGET_DATE="${1:-$TODAY}"

echo "=== EGX Track Recommendations ==="
echo "Target date: $TARGET_DATE"

# Clone or pull the repo
if [ -d "$CLONE_DIR" ]; then
  echo "Pulling latest changes..."
  cd "$CLONE_DIR"
  git pull --quiet
else
  echo "Cloning repo..."
  git clone --depth 1 "$REPO_URL" "$CLONE_DIR" --quiet
  cd "$CLONE_DIR"
fi

# Find the recommendations JSON file
JSON_FILE="data/recommendations_${TARGET_DATE}.json"
if [ ! -f "$JSON_FILE" ]; then
  echo "ERROR: No recommendations file found for $TARGET_DATE"
  echo "Looking for recent files..."
  ls -t data/recommendations_*.json 2>/dev/null | head -3
  exit 1
fi

echo "Found: $JSON_FILE"
cat "$JSON_FILE"
