#!/usr/bin/env bash
set -Eeuo pipefail

# Submit wider pressure-grid follow-up arrays after the current cluster results
# have been pushed. This script intentionally targets only
# jobs/agent_b_llm/userlm_pressure_grid and does not touch existing jobs.

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv-linux/bin/python}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/results}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-1}"
MAX_CONDITIONS_PER_ARRAY="${MAX_CONDITIONS_PER_ARRAY:-8}"
TIERS="${TIERS:-small medium large}"

cd "${PROJECT_ROOT}"

echo "[1/2] Preview wider UserLM pressure-grid follow-up"
"${PYTHON_BIN}" scripts/submit_agent_b_model_jobs.py \
  --root jobs/agent_b_llm/userlm_pressure_grid \
  --provider transformers \
  --tier ${TIERS} \
  --python-bin "${PYTHON_BIN}" \
  --results-dir "${RESULTS_ROOT}" \
  --array-concurrency "${ARRAY_CONCURRENCY}" \
  --max-conditions-per-array "${MAX_CONDITIONS_PER_ARRAY}" \
  --dry-run

echo "[2/2] Submit wider UserLM pressure-grid follow-up"
"${PYTHON_BIN}" scripts/submit_agent_b_model_jobs.py \
  --root jobs/agent_b_llm/userlm_pressure_grid \
  --provider transformers \
  --tier ${TIERS} \
  --python-bin "${PYTHON_BIN}" \
  --results-dir "${RESULTS_ROOT}" \
  --array-concurrency "${ARRAY_CONCURRENCY}" \
  --max-conditions-per-array "${MAX_CONDITIONS_PER_ARRAY}"
