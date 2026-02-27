#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

exec "$SCRIPT_DIR/agent-wrangler" program daemon --interval 600 --write-report "$@"
