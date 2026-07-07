#!/usr/bin/env bash
#SBATCH --job-name=tab-large-cpu
#SBATCH --array=0-83%1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:59:00
#SBATCH --output=slurm/logs/slurm-%x-%A_%a.out
#SBATCH --error=slurm/logs/slurm-%x-%A_%a.err

set -euo pipefail

# Force CPU-only runtime before Python imports torch/transformers.
export CUDA_VISIBLE_DEVICES=""
export NVIDIA_VISIBLE_DEVICES="void"
export ACCELERATE_USE_CPU="true"
export PYTORCH_NVML_BASED_CUDA_CHECK="0"

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv-linux/bin/python}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/results}"
JOB_FILE="${JOB_FILE:?Set JOB_FILE to one jobs/agent_b_llm/transformers_speech_grid/large/*.job file.}"
CONDITION_INDEX="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

cd "${PROJECT_ROOT}"
mkdir -p slurm/logs

"${PYTHON_BIN}" -u -m coop_navigation_sds.batch \
  --job-file "${JOB_FILE}" \
  --results-dir "${RESULTS_ROOT}" \
  --condition-start "${CONDITION_INDEX}" \
  --condition-count 1 \
  --model-device cpu \
  --agent-a-model-device cpu \
  --tts-device cpu \
  --asr-device cpu \
  --progress
