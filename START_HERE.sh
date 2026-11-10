#!/usr/bin/env bash
set -euo pipefail
FINSEC_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$FINSEC_ROOT"
python3 finsec.py verify
exec python3 finsec.py demo --port "${FINSEC_DEMO_PORT:-8765}"
