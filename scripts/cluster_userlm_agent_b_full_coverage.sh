#!/usr/bin/env bash
# Prepare local assets and submit UserLM Agent A / Agent B model Slurm arrays.
# The non-Ollama Transformers grid is the default because it runs without a
# service daemon and keeps every submitted model on the same condition coverage.
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv-linux/bin/python}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/results}"
MODEL_ROOT="${MODEL_ROOT:-${PROJECT_ROOT}/jobs/agent_b_llm/userlm_transformers_speech_grid}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-1}"
ARRAY_CHUNKS="${ARRAY_CHUNKS:-4}"
ASSET_TIMEOUT_SECONDS="${ASSET_TIMEOUT_SECONDS:-1200}"
INCLUDE_OLLAMA="${INCLUDE_OLLAMA:-0}"
SELECTED_TIERS="${SELECTED_TIERS:-small medium large}"
SPEECH_ASSETS="${SPEECH_ASSETS:-piper faster_whisper}"
HF_MAX_WORKERS="${HF_MAX_WORKERS:-1}"
MODEL_PROFILES="${MODEL_PROFILES:-tinyllama_1b_transformers qwen2_5_0_5b_transformers qwen2_5_1_5b_transformers phi3_mini_4k_transformers qwen2_5_7b_transformers falcon3_7b_transformers}"
MODEL_DOWNLOAD_TIMEOUT_SECONDS="${MODEL_DOWNLOAD_TIMEOUT_SECONDS:-14400}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"

cd "${PROJECT_ROOT}"

step_total=9
step_index=0

step() {
  step_index=$((step_index + 1))
  printf '\n[%02d/%02d] %s\n' "${step_index}" "${step_total}" "$1"
}

require_file() {
  if [[ ! -e "$1" ]]; then
    printf 'ERROR: required path is missing: %s\n' "$1" >&2
    exit 2
  fi
}

run_python() {
  printf '+ %s' "${PYTHON_BIN}"
  printf ' %q' "$@"
  printf '\n'
  "${PYTHON_BIN}" "$@"
}

run_python_with_model_timeout() {
  if command -v timeout >/dev/null 2>&1; then
    printf '+ timeout --foreground %q %s' "${MODEL_DOWNLOAD_TIMEOUT_SECONDS}" "${PYTHON_BIN}"
    printf ' %q' "$@"
    printf '\n'
    timeout --foreground "${MODEL_DOWNLOAD_TIMEOUT_SECONDS}" "${PYTHON_BIN}" "$@"
  else
    echo "WARNING: coreutils timeout is unavailable; running without external model-download timeout." >&2
    run_python "$@"
  fi
}

model_profile_args=()
for profile in ${MODEL_PROFILES}; do
  model_profile_args+=(--profile "${profile}")
done

action="${1:-all}"
case "${action}" in
  preview-small) action="preview"; SELECTED_TIERS="small" ;;
  preview-medium) action="preview"; SELECTED_TIERS="medium" ;;
  preview-large) action="preview"; SELECTED_TIERS="large" ;;
  submit-small) action="submit"; SELECTED_TIERS="small" ;;
  submit-medium) action="submit"; SELECTED_TIERS="medium" ;;
  submit-large) action="submit"; SELECTED_TIERS="large" ;;
esac
case "${action}" in
  prepare|preview|submit|refresh|all) ;;
  *)
    cat >&2 <<'USAGE'
Usage: scripts/cluster_userlm_agent_b_full_coverage.sh [prepare|preview|submit|refresh|all|preview-small|preview-medium|preview-large|submit-small|submit-medium|submit-large]

Environment overrides:
  PROJECT_ROOT=/path/to/CoopNavigationSDS
  PYTHON_BIN=/path/to/.venv-linux/bin/python
  RESULTS_ROOT=/path/to/results
  MODEL_ROOT=/path/to/jobs/agent_b_llm/userlm_transformers_speech_grid
  SELECTED_TIERS="small medium large"
  SPEECH_ASSETS="piper faster_whisper"
  HF_MAX_WORKERS=1
  MODEL_PROFILES="tinyllama_1b_transformers qwen2_5_0_5b_transformers qwen2_5_1_5b_transformers phi3_mini_4k_transformers qwen2_5_7b_transformers falcon3_7b_transformers"
  MODEL_DOWNLOAD_TIMEOUT_SECONDS=14400
  ARRAY_CONCURRENCY=1
  ARRAY_CHUNKS=4
  ASSET_TIMEOUT_SECONDS=1200
  INCLUDE_OLLAMA=0|1
USAGE
    exit 2
    ;;
esac

step "Verify project paths"
require_file "${PYTHON_BIN}"
require_file "${MODEL_ROOT}"
mkdir -p "${RESULTS_ROOT}" "${PROJECT_ROOT}/slurm/logs"
printf 'Project: %s\nPython:  %s\nResults: %s\nJobs:    %s\n' \
  "${PROJECT_ROOT}" "${PYTHON_BIN}" "${RESULTS_ROOT}" "${MODEL_ROOT}"
printf 'Tiers:   %s\n' "${SELECTED_TIERS}"
printf 'Array chunks per model: %s\n' "${ARRAY_CHUNKS}"
printf 'Speech assets: %s\n' "${SPEECH_ASSETS}"
printf 'Model profiles: %s\n' "${MODEL_PROFILES}"
printf 'Hugging Face workers: %s\n' "${HF_MAX_WORKERS}"
printf 'Model download timeout: %s seconds\n' "${MODEL_DOWNLOAD_TIMEOUT_SECONDS}"
printf 'Hugging Face Xet disabled: %s\n' "${HF_HUB_DISABLE_XET}"

if [[ "${action}" == "prepare" || "${action}" == "all" ]]; then
  step "Prepare selected speech assets with fail-fast progress"
  run_python -u scripts/prepare_test_environment.py \
    --only-assets ${SPEECH_ASSETS} \
    --fail-fast \
    --asset-timeout-seconds "${ASSET_TIMEOUT_SECONDS}"

  step "Prepare UserLM Agent A model assets"
  run_python_with_model_timeout -u scripts/setup_transformers_agent_b_models.py \
    --profile userlm_8b_transformers \
    --download \
    --max-workers "${HF_MAX_WORKERS}"

  step "Prepare all non-Ollama Transformers Agent B model assets"
  run_python_with_model_timeout -u scripts/setup_transformers_agent_b_models.py \
    "${model_profile_args[@]}" \
    --download \
    --max-workers "${HF_MAX_WORKERS}"

  step "Verify project-local asset readiness"
  run_python -u scripts/prepare_test_environment.py \
    --check \
    --only-assets ${SPEECH_ASSETS} \
    --fail-fast
  run_python -u scripts/setup_transformers_agent_b_models.py \
    --profile userlm_8b_transformers \
    "${model_profile_args[@]}" \
    --json

  if [[ "${INCLUDE_OLLAMA}" == "1" ]]; then
    step "Prepare Ollama Agent B models"
    command -v ollama >/dev/null 2>&1 || {
      echo "ERROR: INCLUDE_OLLAMA=1 but ollama is not available on PATH." >&2
      exit 2
    }
    run_python -u scripts/setup_agent_b_models.py --tier small --tier medium --tier large --pull
  else
    step "Skip Ollama models"
    echo "INCLUDE_OLLAMA=0, so only service-free Transformers Agent B jobs are prepared."
  fi
else
  step "Skip preparation"
  echo "Action ${action} does not prepare assets."
  step "Skip UserLM preparation"
  echo "Action ${action} does not prepare Agent A assets."
  step "Skip Agent B preparation"
  echo "Action ${action} does not prepare Agent B assets."
  step "Skip readiness verification"
  echo "Action ${action} does not verify readiness."
  step "Skip optional Ollama preparation"
  echo "Action ${action} does not prepare Ollama assets."
fi

if [[ "${action}" == "preview" || "${action}" == "submit" || "${action}" == "all" ]]; then
  step "Preview UserLM Agent A / Agent B arrays sorted by Agent B size"
  run_python -u scripts/submit_agent_b_model_jobs.py \
    --root "${MODEL_ROOT}" \
    --provider transformers \
    --tier ${SELECTED_TIERS} \
    "${model_profile_args[@]}" \
    --results-dir "${RESULTS_ROOT}" \
    --python-bin "${PYTHON_BIN}" \
    --array-concurrency "${ARRAY_CONCURRENCY}" \
    --array-chunks "${ARRAY_CHUNKS}" \
    --dry-run

  if [[ "${action}" == "submit" || "${action}" == "all" ]]; then
    step "Submit independent fail-fast Slurm arrays"
    run_python -u scripts/submit_agent_b_model_jobs.py \
      --root "${MODEL_ROOT}" \
      --provider transformers \
      --tier ${SELECTED_TIERS} \
      "${model_profile_args[@]}" \
      --results-dir "${RESULTS_ROOT}" \
      --python-bin "${PYTHON_BIN}" \
      --array-concurrency "${ARRAY_CONCURRENCY}" \
      --array-chunks "${ARRAY_CHUNKS}"
  else
    step "Skip Slurm submission"
    echo "Preview mode does not call sbatch."
  fi
else
  step "Skip Slurm preview"
  echo "Action ${action} does not preview Slurm jobs."
  step "Skip Slurm submission"
  echo "Action ${action} does not submit Slurm jobs."
fi

if [[ "${action}" == "refresh" || "${action}" == "all" ]]; then
  step "Refresh coverage and phase-wise comparison outputs"
  run_python -u scripts/update_experiment_coverage.py --results-dir "${RESULTS_ROOT}"
  run_python -u -m coop_navigation_sds.ResultsAndArtifacts.comparison \
    "${RESULTS_ROOT}" \
    --output "${RESULTS_ROOT}/comparison" \
    --include-partial
else
  step "Skip result analysis refresh"
  echo "Run '${BASH_SOURCE[0]} refresh' after Slurm jobs finish."
fi

echo
echo "DONE | ${action} completed"
