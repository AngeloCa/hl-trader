#!/usr/bin/env bash
# ─────────────────────────────────────────────────────
#  HYPE/USDC Trading Bot — launcher
#  Usage:
#    ./run.sh          # dry-run (safe, no real orders)
#    ./run.sh live     # live trading (requires .env)
# ─────────────────────────────────────────────────────
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Load .env if present
if [[ -f .env ]]; then
  export $(grep -v '^#' .env | xargs)
fi

# Override DRY_RUN based on argument
if [[ "${1:-}" == "live" ]]; then
  export DRY_RUN=false
  echo "⚠️  LIVE MODE — real orders will be placed"
  read -p "Type YES to confirm: " confirm
  [[ "$confirm" == "YES" ]] || { echo "Aborted."; exit 1; }
else
  export DRY_RUN=true
  echo "🔵 DRY-RUN mode — no real orders"
fi

# Install deps if venv missing
if [[ ! -d ../.venv ]]; then
  python3 -m venv ../.venv
fi
../.venv/bin/pip install -q -r requirements.txt

exec ../.venv/bin/python bot.py
