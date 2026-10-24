#!/usr/bin/env bash
set -euo pipefail
# Model inference only. Paid explanation judging is a separate command.
MANIFEST="${1:?provide an external manifest.json}"
WORK_ROOT="${2:?provide an external work root}"
HOST="${3:-http://127.0.0.1:11434}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for variant in raw trajectory raw_flow trajectory_flow full; do
  python -m mustard experiment --manifest "$MANIFEST" \
    --config "$REPO_ROOT/configs/ablations/$variant.json" \
    --work "$WORK_ROOT/$variant" --host "$HOST"
done
