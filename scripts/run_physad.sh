#!/usr/bin/env bash
set -euo pipefail
# Usage: bash scripts/run_physad.sh MANIFEST WORK_DIR [OLLAMA_HOST]
MANIFEST="${1:?provide an external manifest.json}"
WORK_DIR="${2:?provide an external work directory}"
HOST="${3:-http://127.0.0.1:11434}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python -m mustard experiment --manifest "$MANIFEST" --config "$REPO_ROOT/configs/paper.json" \
  --work "$WORK_DIR" --host "$HOST"
