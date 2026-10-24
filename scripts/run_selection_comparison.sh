#!/usr/bin/env bash
set -euo pipefail
MANIFEST="${1:?provide an external manifest.json}"
WORK_ROOT="${2:?provide an external work root}"
HOST="${3:-http://127.0.0.1:11434}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for sampler in uniform keyframe; do
  python -m mustard experiment --manifest "$MANIFEST" \
    --config "$REPO_ROOT/configs/baselines/qwen_$sampler.json" \
    --work "$WORK_ROOT/qwen_$sampler" --host "$HOST"
done
