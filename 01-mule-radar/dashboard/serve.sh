#!/usr/bin/env bash
# Serve the analyst console on http://localhost:${PORT:-8050}
# Rebuilds index.html from the latest results first.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 dashboard/build.py results/demo.json
PORT=${PORT:-8050}
echo "MuleRadar console: http://localhost:$PORT"
exec python3 -m http.server "$PORT" --bind 127.0.0.1 --directory dashboard
