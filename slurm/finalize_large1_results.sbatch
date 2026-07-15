#!/usr/bin/env bash
#SBATCH --job-name=finalize-large1-results
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/finalize-large1-results-%j.out
#SBATCH --error=slurm/logs/finalize-large1-results-%j.err

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv-linux/bin/python}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/results}"

cd "${PROJECT_ROOT}"
mkdir -p slurm/logs

echo "start=$(date --iso-8601=seconds)"
echo "host=$(hostname)"
echo "project=${PROJECT_ROOT}"
echo "python=${PYTHON_BIN}"
echo "results=${RESULTS_ROOT}"

PROJECT_ROOT="${PROJECT_ROOT}" \
PYTHON_BIN="${PYTHON_BIN}" \
RESULTS_ROOT="${RESULTS_ROOT}" \
STABILITY_SECONDS="${STABILITY_SECONDS:-120}" \
bash scripts/cluster_finalize_large1_results.sh

echo "done=$(date --iso-8601=seconds)"
