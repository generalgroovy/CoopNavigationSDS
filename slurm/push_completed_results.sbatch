#!/usr/bin/env bash
#SBATCH --job-name=completed-results-push
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/completed-results-push-%j.out
#SBATCH --error=slurm/logs/completed-results-push-%j.err

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv-linux/bin/python}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/results}"

cd "${PROJECT_ROOT}"
mkdir -p slurm/logs

echo "start=$(date --iso-8601=seconds)"
echo "project=${PROJECT_ROOT}"
echo "python=${PYTHON_BIN}"
echo "results=${RESULTS_ROOT}"

PROJECT_ROOT="${PROJECT_ROOT}" \
PYTHON_BIN="${PYTHON_BIN}" \
RESULTS_ROOT="${RESULTS_ROOT}" \
bash scripts/cluster_push_completed_results.sh

echo "done=$(date --iso-8601=seconds)"
