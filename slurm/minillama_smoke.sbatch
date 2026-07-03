#!/usr/bin/env bash
#SBATCH --job-name=mn-smoke
#SBATCH --array=0-1%1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/results/slurm/smoke}"
cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" scripts/run_slurm_condition.py run \
  --grid slurm/minillama_smoke.json \
  --results-dir "${RESULTS_ROOT}" \
  --model-device cpu
