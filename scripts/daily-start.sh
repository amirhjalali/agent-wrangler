#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[workflow] generating snapshot"
python3 scripts/workflow_agent.py snapshot --include-linear || true

echo "[workflow] running health checks"
python3 scripts/workflow_agent.py doctor || true

echo "[workflow] top focus"
python3 scripts/workflow_agent.py focus --limit 5
