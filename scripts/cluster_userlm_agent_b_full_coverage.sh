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
ARRAY_CHUNKS="${ARRAY_CHUNKS:-1}"
MAX_CONDITIONS_PER_ARRAY="${MAX_CONDITIONS_PER_ARRAY:-14}"
ASSET_TIMEOUT_SECONDS="${ASSET_TIMEOUT_SECONDS:-1200}"
INCLUDE_OLLAMA="${INCLUDE_OLLAMA:-0}"
SELECTED_TIERS="${SELECTED_TIERS:-small medium large}"
SPEECH_ASSETS="${SPEECH_ASSETS:-piper faster_whisper}"
HF_MAX_WORKERS="${HF_MAX_WORKERS:-1}"
MODEL_PROFILES="${MODEL_PROFILES:-tinyllama_1b_transformers qwen2_5_0_5b_transformers smollm2_360m_transformers smollm2_1_7b_transformers qwen2_5_1_5b_transformers phi3_mini_4k_transformers gemma2_2b_it_transformers qwen3_4b_instruct_transformers qwen2_5_7b_transformers mistral_7b_transformers llama3_1_8b_transformers falcon3_7b_transformers}"
MODEL_DOWNLOAD_TIMEOUT_SECONDS="${MODEL_DOWNLOAD_TIMEOUT_SECONDS:-14400}"
GIT_REMOTE="${GIT_REMOTE:-origin}"
GIT_BRANCH="${GIT_BRANCH:-main}"
GIT_KEY_PATH="${GIT_KEY_PATH:-${PROJECT_ROOT}/key2}"
RESULTS_COMMIT_MESSAGE="${RESULTS_COMMIT_MESSAGE:-Add cluster results and Slurm logs}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"

cd "${PROJECT_ROOT}"

configure_git_line_endings() {
  git config core.autocrlf false
  git config core.eol lf
}

step_total=10
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

print_result_summary() {
  run_python - <<'PY'
import csv
import json
from pathlib import Path

root = Path("results")
manifest_path = root / "general" / "analysis_manifest.json"
coverage_path = root / "experiment_coverage_summary.json"
inventory_path = root / "general" / "run_inventory.csv"

if coverage_path.is_file():
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    print(
        "Coverage registry: "
        f"completed_runs={coverage.get('completed_run_count')} "
        f"completed_planned={coverage.get('completed_planned_configuration_count')} "
        f"coverage={coverage.get('coverage_percentage')}%"
    )
else:
    print("Coverage registry: missing")

if manifest_path.is_file():
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(
        "Partial-aware comparison: "
        f"runs={manifest.get('discovered_run_count')} "
        f"observed={manifest.get('observed_condition_count')} "
        f"completed={manifest.get('completed_condition_count')} "
        f"sources={manifest.get('source_file_count')}"
    )
else:
    print("Partial-aware comparison: missing")

if not inventory_path.is_file():
    print("Run inventory: missing")
    raise SystemExit(0)

rows = list(csv.DictReader(inventory_path.open(encoding="utf-8")))
groups = {}
for row in rows:
    key = (
        row.get("agent_a_type") or "",
        row.get("agent_b_llm_size") or "",
        row.get("agent_b_model") or "",
    )
    item = groups.setdefault(key, {
        "runs": 0,
        "planned": 0,
        "observed": 0,
        "completed": 0,
        "interrupted": 0,
        "not_started": 0,
        "success": 0,
        "states": set(),
    })
    item["runs"] += 1
    item["states"].add(row.get("run_state") or "")
    for source, target in (
        ("planned_condition_count", "planned"),
        ("observed_condition_count", "observed"),
        ("completed_condition_count", "completed"),
        ("interrupted_condition_count", "interrupted"),
        ("not_started_condition_count", "not_started"),
        ("successful_condition_count", "success"),
    ):
        item[target] += int(float(row.get(source) or 0))

print("Run inventory by Agent B:")
for key, item in sorted(
    groups.items(),
    key=lambda value: (
        {"small": 0, "medium": 1, "large": 2}.get(value[0][1], 9),
        value[0][0],
        value[0][2],
    ),
):
    print(
        f"  {key[0]} | {key[1]} | {key[2]} | "
        f"runs={item['runs']} planned={item['planned']} observed={item['observed']} "
        f"completed={item['completed']} interrupted={item['interrupted']} "
        f"not_started={item['not_started']} success={item['success']} "
        f"states={','.join(sorted(item['states']))}"
    )
PY
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
  prepare|preview|submit|refresh|push-results|all) ;;
  *)
    cat >&2 <<'USAGE'
Usage: scripts/cluster_userlm_agent_b_full_coverage.sh [prepare|preview|submit|refresh|push-results|all|preview-small|preview-medium|preview-large|submit-small|submit-medium|submit-large]

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
  MAX_CONDITIONS_PER_ARRAY=14
  ARRAY_CHUNKS=1
  ASSET_TIMEOUT_SECONDS=1200
  INCLUDE_OLLAMA=0|1
  GIT_KEY_PATH=/path/to/key2
  GIT_REMOTE=origin
  GIT_BRANCH=main
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
printf 'Maximum conditions per submitted array: %s\n' "${MAX_CONDITIONS_PER_ARRAY}"
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
    --max-conditions-per-array "${MAX_CONDITIONS_PER_ARRAY}" \
    --dry-run

  if [[ "${action}" == "submit" || "${action}" == "all" ]]; then
    step "Submit independent Slurm arrays with fail-fast preflight"
    run_python -u scripts/submit_agent_b_model_jobs.py \
      --root "${MODEL_ROOT}" \
      --provider transformers \
      --tier ${SELECTED_TIERS} \
      "${model_profile_args[@]}" \
      --results-dir "${RESULTS_ROOT}" \
      --python-bin "${PYTHON_BIN}" \
      --array-concurrency "${ARRAY_CONCURRENCY}" \
      --array-chunks "${ARRAY_CHUNKS}" \
      --max-conditions-per-array "${MAX_CONDITIONS_PER_ARRAY}"
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

if [[ "${action}" == "refresh" || "${action}" == "push-results" || "${action}" == "all" ]]; then
  step "Refresh coverage and phase-wise comparison outputs"
  run_python -u scripts/update_experiment_coverage.py --results-dir "${RESULTS_ROOT}"
  run_python -u -m coop_navigation_sds.ResultsAndArtifacts.comparison \
    "${RESULTS_ROOT}" \
    --output "${RESULTS_ROOT}/general" \
    --include-partial
  print_result_summary
else
  step "Skip result analysis refresh"
  echo "Run '${BASH_SOURCE[0]} refresh' after Slurm jobs finish."
fi

if [[ "${action}" == "push-results" ]]; then
  step "Commit and push refreshed results and Slurm logs"
  require_file "${GIT_KEY_PATH}"
  configure_git_line_endings
  git config user.name "${GIT_AUTHOR_NAME:-generalgroovy}"
  git config user.email "${GIT_AUTHOR_EMAIL:-generalgroovy@users.noreply.github.com}"
  eval "$(ssh-agent -s)"
  ssh-add "${GIT_KEY_PATH}"
  git add "${RESULTS_ROOT}"
  git add -f "${PROJECT_ROOT}/slurm/logs"
  git status --short
  if git diff --cached --quiet; then
    echo "No staged result/log changes to commit."
  else
    git commit -m "${RESULTS_COMMIT_MESSAGE}"
  fi
  GIT_SSH_COMMAND="ssh -i ${GIT_KEY_PATH} -o IdentitiesOnly=yes -o HostName=ssh.github.com -o Port=443" \
    git fetch "${GIT_REMOTE}" "${GIT_BRANCH}"
  GIT_SSH_COMMAND="ssh -i ${GIT_KEY_PATH} -o IdentitiesOnly=yes -o HostName=ssh.github.com -o Port=443" \
    git rebase "${GIT_REMOTE}/${GIT_BRANCH}"
  GIT_SSH_COMMAND="ssh -i ${GIT_KEY_PATH} -o IdentitiesOnly=yes -o HostName=ssh.github.com -o Port=443" \
    git push "${GIT_REMOTE}" HEAD:"${GIT_BRANCH}"
else
  step "Skip result push"
  echo "Action ${action} does not commit or push results."
fi

echo
echo "DONE | ${action} completed"
