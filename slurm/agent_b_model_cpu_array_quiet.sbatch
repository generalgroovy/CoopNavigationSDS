#!/usr/bin/env bash
#SBATCH --job-name=agent-b-model-quiet
#SBATCH --array=0-7%1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=18G
#SBATCH --time=01:29:00
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --error=slurm/logs/%x-%A_%a.err

set -Eeuo pipefail

export CUDA_VISIBLE_DEVICES=""
export NVIDIA_VISIBLE_DEVICES="void"
export ACCELERATE_USE_CPU="true"
export PYTORCH_NVML_BASED_CUDA_CHECK="0"

# Avoid ONNX Runtime affinity spam on Slurm CPU sets.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export ONNXRUNTIME_THREAD_POOL_SIZE="${ONNXRUNTIME_THREAD_POOL_SIZE:-1}"
export ORT_LOG_SEVERITY_LEVEL="${ORT_LOG_SEVERITY_LEVEL:-3}"

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv-linux/bin/python}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/results}"
JOB_FILE="${JOB_FILE:?Set JOB_FILE to one Agent B .job file.}"
VALID_CONDITIONS_ONLY="${VALID_CONDITIONS_ONLY:-1}"
CONDITION_INDEX="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

cd "${PROJECT_ROOT}"
mkdir -p slurm/logs

test -x "${PYTHON_BIN}" || { echo "ERROR: PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2; exit 2; }
test -f "${JOB_FILE}" || { echo "ERROR: JOB_FILE is missing: ${JOB_FILE}" >&2; exit 2; }

condition_args=()
if [[ "${VALID_CONDITIONS_ONLY}" == "1" ]]; then
  condition_args+=(--valid-conditions-only)
else
  condition_args+=(--no-valid-conditions-only)
fi

echo "START $(date --iso-8601=seconds) job=${SLURM_JOB_ID:-none}_${CONDITION_INDEX} file=${JOB_FILE}"

"${PYTHON_BIN}" -u -m coop_navigation_sds.batch \
  --job-file "${JOB_FILE}" \
  --results-dir "${RESULTS_ROOT}" \
  --condition-start "${CONDITION_INDEX}" \
  --condition-count 1 \
  "${condition_args[@]}" \
  --model-device cpu \
  --agent-a-model-device cpu \
  --tts-device cpu \
  --asr-device cpu \
  --no-update-coverage-registry \
  --no-require-complete-speech-performance-coverage \
  --progress

echo "DONE $(date --iso-8601=seconds) job=${SLURM_JOB_ID:-none}_${CONDITION_INDEX}"
