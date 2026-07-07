#!/usr/bin/env bash
#SBATCH --job-name=mn-grid-cpu
#SBATCH --array=0-15%8
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/slurm-%x-%A_%a.out
#SBATCH --error=slurm/logs/slurm-%x-%A_%a.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/results/slurm/cpu}"
cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" scripts/run_slurm_condition.py run \
  --grid slurm/minillama_grid.json \
  --results-dir "${RESULTS_ROOT}" \
  --model-device cpu
