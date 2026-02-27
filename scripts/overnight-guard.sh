#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Default: dry-run guardrail checks every 5 minutes.
python3 scripts/terminal_sentinel.py overnight \
  --source ghostty \
  --interval 300 \
  --max-ai-sessions 4 \
  --kill-waiting-ai-after 120 \
  --write-log
