#!/usr/bin/env bash
#SBATCH --job-name=analyze-push-completed
#SBATCH --partition=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:30:00
#SBATCH --output=slurm/logs/analyze-push-completed-%j.out
#SBATCH --error=slurm/logs/analyze-push-completed-%j.err

set -Eeuo pipefail

ROOT="${PROJECT_ROOT:-$(pwd)}"
cd "${ROOT}"
mkdir -p slurm/logs

echo "[1/3] Environment"
date
hostname
echo "PROJECT_ROOT=${ROOT}"
echo "PYTHON_BIN=${PYTHON_BIN:-${ROOT}/.venv-linux/bin/python}"
echo "RESULTS_ROOT=${RESULTS_ROOT:-${ROOT}/results}"

echo "[2/3] Run clean pull, analysis, and push workflow"
bash scripts/cluster_clean_pull_analyze_push.sh

echo "[3/3] Done"
date
