#!/usr/bin/env bash
#SBATCH --job-name=u8b-small1-cpu
#SBATCH --array=0-7%1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=01:29:00
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
JOB_FILE="${JOB_FILE:-jobs/agent_b_llm/userlm_speech_grid/small/01-llama3.2-1b.job}"
CONDITION_INDEX="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

cd "${PROJECT_ROOT}"

command -v ollama >/dev/null 2>&1 || {
  echo "ERROR: ollama executable is required for Agent B llama3.2:1b." >&2
  exit 2
}

export OLLAMA_MODELS="${OLLAMA_MODELS:-${PROJECT_ROOT}/.model-providers/agent_b/linux/ollama}"
export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:$((11435 + CONDITION_INDEX))}"
mkdir -p "${OLLAMA_MODELS}" slurm/logs
ollama serve > "slurm/logs/ollama-${SLURM_JOB_ID}-${CONDITION_INDEX}.log" 2>&1 &
OLLAMA_PID=$!
trap 'kill "${OLLAMA_PID}" 2>/dev/null || true' EXIT
sleep "${OLLAMA_STARTUP_WAIT_SEC:-10}"

"${PYTHON_BIN}" -u -m coop_navigation_sds.batch \
  --job-file "${JOB_FILE}" \
  --results-dir "${RESULTS_ROOT}" \
  --condition-start "${CONDITION_INDEX}" \
  --condition-count 1 \
  --model-base-url "http://${OLLAMA_HOST}/api" \
  --model-device cpu \
  --agent-a-model-device cpu \
  --tts-device cpu \
  --asr-device cpu \
  --progress
