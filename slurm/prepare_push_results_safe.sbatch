#!/usr/bin/env bash
#SBATCH --job-name=prepare-push-results
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=slurm/logs/prepare-push-results-%j.out
#SBATCH --error=slurm/logs/prepare-push-results-%j.err

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv-linux/bin/python}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/results}"
GIT_KEY_PATH="${GIT_KEY_PATH:-${PROJECT_ROOT}/key2}"

cd "${PROJECT_ROOT}"

echo "Project: ${PROJECT_ROOT}"
echo "Python: ${PYTHON_BIN}"
echo "Results: ${RESULTS_ROOT}"
echo "Git key: ${GIT_KEY_PATH}"

bash scripts/cluster_prepare_push_results_safe.sh all
