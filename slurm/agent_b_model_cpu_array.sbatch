#!/usr/bin/env bash
#SBATCH --job-name=agent-b-model
#SBATCH --array=0-7%1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=18G
#SBATCH --time=01:29:00
#SBATCH --output=slurm/logs/slurm-%x-%A_%a.out
#SBATCH --error=slurm/logs/slurm-%x-%A_%a.err

set -euo pipefail

# CPU arrays must not probe cluster GPUs. Some GPU nodes expose old NVIDIA
# drivers, and PyTorch can fail during import/preflight even when models are
# explicitly configured for CPU inference.
export CUDA_VISIBLE_DEVICES=""
export NVIDIA_VISIBLE_DEVICES="void"
export ACCELERATE_USE_CPU="true"
export PYTORCH_NVML_BASED_CUDA_CHECK="0"

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv-linux/bin/python}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/results}"
JOB_FILE="${JOB_FILE:?Set JOB_FILE to one Agent B .job file.}"
START_OLLAMA="${START_OLLAMA:-0}"
CONDITION_INDEX="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

cd "${PROJECT_ROOT}"
mkdir -p slurm/logs

extra_args=()
if [[ "${START_OLLAMA}" == "1" ]]; then
  command -v ollama >/dev/null 2>&1 || {
    echo "ERROR: ollama executable is required for this Agent B model." >&2
    exit 2
  }
  port=$((20000 + (${SLURM_JOB_ID:-0} % 20000) + (CONDITION_INDEX % 1000)))
  export OLLAMA_MODELS="${OLLAMA_MODELS:-${PROJECT_ROOT}/.model-providers/agent_b/linux/ollama}"
  export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:${port}}"
  mkdir -p "${OLLAMA_MODELS}"
  ollama serve > "slurm/logs/ollama-${SLURM_JOB_ID}-${CONDITION_INDEX}.log" 2>&1 &
  OLLAMA_PID=$!
  trap 'kill "${OLLAMA_PID}" 2>/dev/null || true' EXIT
  sleep "${OLLAMA_STARTUP_WAIT_SEC:-10}"
  extra_args+=(--model-base-url "http://${OLLAMA_HOST}/api")
fi

"${PYTHON_BIN}" -u -m coop_navigation_sds.batch \
  --job-file "${JOB_FILE}" \
  --results-dir "${RESULTS_ROOT}" \
  --condition-start "${CONDITION_INDEX}" \
  --condition-count 1 \
  --model-device cpu \
  --agent-a-model-device cpu \
  --tts-device cpu \
  --asr-device cpu \
  --progress \
  "${extra_args[@]}"
