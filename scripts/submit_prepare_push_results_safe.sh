#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv-linux/bin/python}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/results}"
GIT_KEY_PATH="${GIT_KEY_PATH:-${PROJECT_ROOT}/key2}"

cd "${PROJECT_ROOT}"
mkdir -p slurm/logs

sbatch \
  --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",PYTHON_BIN="${PYTHON_BIN}",RESULTS_ROOT="${RESULTS_ROOT}",GIT_KEY_PATH="${GIT_KEY_PATH}" \
  slurm/prepare_push_results_safe.sbatch
