#!/usr/bin/env bash
set -euo pipefail

MAX_PARALLEL="${MAX_PARALLEL:-2}"
RESULTS_DIR="${RESULTS_DIR:-results}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

find "$ROOT/jobs" -maxdepth 1 -name 'tinyllama_piper_faster_whisper_parallel_*.job' -print0 |
  sort -z |
  xargs -0 -P "$MAX_PARALLEL" -I{} \
    python -m coop_navigation_sds.batch \
      --job-file "{}" \
      --results-dir "$RESULTS_DIR" \
      --progress
