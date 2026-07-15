#!/usr/bin/env bash
set -Eeuo pipefail

# Compatibility wrapper. The finalization workflow now prepares all
# thesis-relevant result folders together, including large1, before pushing.

ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
exec bash "${ROOT}/scripts/cluster_finalize_relevant_results.sh"
