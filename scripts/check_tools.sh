#!/usr/bin/env bash
set -euo pipefail
FINSEC_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$FINSEC_ROOT"
FINSEC_PYTHON="${FINSEC_PYTHON:-python3.11}"
export FINSEC_PYTHON
"$FINSEC_PYTHON" -m unittest tests.test_crypto_sandbox tests.test_agent_workspace tests.test_pi_crypto_tools -v
node --test pi_crypto_agent/agent_described.test.mjs
node submission_tools/tool_loop_smoke.mjs
