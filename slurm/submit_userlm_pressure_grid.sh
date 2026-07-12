#!/usr/bin/env bash
set -euo pipefail

# Compatibility wrapper for the isolated pressure-grid submitter. It does not
# inspect, cancel, or mutate already running Slurm jobs. It only submits new
# arrays for the job files under jobs/agent_b_llm/userlm_pressure_grid.

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv-linux/bin/python}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/results}"
MAX_CONDITIONS_PER_ARRAY="${MAX_CONDITIONS_PER_ARRAY:-8}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-1}"

cd "${PROJECT_ROOT}"

echo "[1/3] Pressure-grid dry run"
"${PYTHON_BIN}" scripts/submit_agent_b_model_jobs.py \
  --root jobs/agent_b_llm/userlm_pressure_grid \
  --provider transformers \
  --tier small medium large \
  --python-bin "${PYTHON_BIN}" \
  --results-dir "${RESULTS_ROOT}" \
  --array-concurrency "${ARRAY_CONCURRENCY}" \
  --max-conditions-per-array "${MAX_CONDITIONS_PER_ARRAY}" \
  --dry-run

echo "[2/3] Submit pressure-grid arrays"
"${PYTHON_BIN}" scripts/submit_agent_b_model_jobs.py \
  --root jobs/agent_b_llm/userlm_pressure_grid \
  --provider transformers \
  --tier small medium large \
  --python-bin "${PYTHON_BIN}" \
  --results-dir "${RESULTS_ROOT}" \
  --array-concurrency "${ARRAY_CONCURRENCY}" \
  --max-conditions-per-array "${MAX_CONDITIONS_PER_ARRAY}"

echo "[3/3] Current queue"
squeue -u "${USER}" -o "%.18i %.9P %.48j %.8T %.12M %.12l %.8C %.10m %.20R"
