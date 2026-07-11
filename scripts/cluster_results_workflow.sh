#!/usr/bin/env bash
# Cluster-side helper for inspecting running Slurm jobs, refreshing completed
# experiment results, pushing results/logs, and submitting the remaining
# UserLM-Agent-A / Transformer-Agent-B model tiers.
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv-linux/bin/python}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/results}"
GIT_KEY_PATH="${GIT_KEY_PATH:-${PROJECT_ROOT}/key2}"
GIT_REMOTE="${GIT_REMOTE:-origin}"
GIT_BRANCH="${GIT_BRANCH:-main}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-1}"
MAX_CONDITIONS_PER_ARRAY="${MAX_CONDITIONS_PER_ARRAY:-14}"

SMALL_PROFILES="${SMALL_PROFILES:-tinyllama_1b_transformers qwen2_5_0_5b_transformers}"
MEDIUM_PROFILES="${MEDIUM_PROFILES:-qwen2_5_1_5b_transformers phi3_mini_4k_transformers}"
LARGE_PROFILES="${LARGE_PROFILES:-qwen2_5_7b_transformers falcon3_7b_transformers}"

cd "${PROJECT_ROOT}"

configure_git_line_endings() {
  git config core.autocrlf false
  git config core.eol lf
}

submit_or_preview_tier() {
  local tier="$1"
  local profiles="$2"
  local dry_run_flag="$3"
  local command=(
    "${PYTHON_BIN}" -u scripts/submit_agent_b_model_jobs.py
    --root "${PROJECT_ROOT}/jobs/agent_b_llm/userlm_transformers_speech_grid"
    --provider transformers
    --tier "${tier}"
    --results-dir "${RESULTS_ROOT}"
    --python-bin "${PYTHON_BIN}"
    --array-concurrency "${ARRAY_CONCURRENCY}"
    --max-conditions-per-array "${MAX_CONDITIONS_PER_ARRAY}"
  )
  for profile in ${profiles}; do
    command+=(--profile "${profile}")
  done
  if [[ "${dry_run_flag}" == "dry-run" ]]; then
    command+=(--dry-run)
  fi
  printf '+'
  printf ' %q' "${command[@]}"
  printf '\n'
  "${command[@]}"
}

refresh_results() {
  "${PYTHON_BIN}" -u scripts/update_experiment_coverage.py --results-dir "${RESULTS_ROOT}"
  "${PYTHON_BIN}" -u -m coop_navigation_sds.ResultsAndArtifacts.comparison \
    "${RESULTS_ROOT}" \
    --output "${RESULTS_ROOT}/comparison" \
    --include-partial
  "${PYTHON_BIN}" - "${RESULTS_ROOT}" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest_path = root / "comparison" / "analysis_manifest.json"
inventory_path = root / "comparison" / "run_inventory.csv"
coverage_path = root / "experiment_coverage_summary.json"

if coverage_path.is_file():
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    print(
        "coverage:",
        f"completed_runs={coverage.get('completed_run_count')}",
        f"planned_completed={coverage.get('completed_planned_configuration_count')}",
        f"coverage={coverage.get('coverage_percentage')}%",
    )
if manifest_path.is_file():
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(
        "comparison:",
        f"runs={manifest.get('discovered_run_count')}",
        f"observed={manifest.get('observed_condition_count')}",
        f"completed={manifest.get('completed_condition_count')}",
        f"sources={manifest.get('source_file_count')}",
    )
if inventory_path.is_file():
    rows = list(csv.DictReader(inventory_path.open(encoding="utf-8")))
    groups = {}
    for row in rows:
        key = (
            row.get("agent_a_type") or "",
            row.get("agent_b_llm_size") or "",
            row.get("agent_b_model") or "",
        )
        item = groups.setdefault(key, {"planned": 0, "observed": 0, "completed": 0, "success": 0})
        for source, target in (
            ("planned_condition_count", "planned"),
            ("observed_condition_count", "observed"),
            ("completed_condition_count", "completed"),
            ("successful_condition_count", "success"),
        ):
            item[target] += int(float(row.get(source) or 0))
    order = {"small": 0, "medium": 1, "large": 2}
    print("by model:")
    for key, item in sorted(groups.items(), key=lambda value: (order.get(value[0][1], 9), value[0][2])):
        print(
            f"  {key[0]} | {key[1]} | {key[2]} | "
            f"planned={item['planned']} observed={item['observed']} "
            f"completed={item['completed']} success={item['success']}"
        )
PY
}

push_results() {
  refresh_results
  if [[ ! -f "${GIT_KEY_PATH}" ]]; then
    echo "ERROR: Git key missing: ${GIT_KEY_PATH}" >&2
    exit 2
  fi
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
    git commit -m "${RESULTS_COMMIT_MESSAGE:-Add cluster experiment results}"
  fi
  GIT_SSH_COMMAND="ssh -i ${GIT_KEY_PATH} -o IdentitiesOnly=yes -o HostName=ssh.github.com -o Port=443" \
    git push "${GIT_REMOTE}" HEAD:"${GIT_BRANCH}"
}

usage() {
  cat <<'USAGE'
Usage: bash scripts/cluster_results_workflow.sh ACTION

Actions:
  squeue          Show detailed current Slurm jobs for this user.
  refresh         Rebuild result coverage and comparison documents.
  push-results    Refresh, commit results plus slurm/logs, and push.
  preview-small   Print small-model Slurm arrays only.
  preview-medium  Print medium-model Slurm arrays only.
  preview-large   Print large-model Slurm arrays only.
  submit-small    Submit the two configured small Agent B models.
  submit-medium   Submit the two configured medium Agent B models.
  submit-large    Submit the two configured large Agent B models.

Default model coverage:
  small:  tinyllama_1b_transformers qwen2_5_0_5b_transformers
  medium: qwen2_5_1_5b_transformers phi3_mini_4k_transformers
  large:  qwen2_5_7b_transformers falcon3_7b_transformers

Environment overrides:
  PROJECT_ROOT, PYTHON_BIN, RESULTS_ROOT, GIT_KEY_PATH, GIT_REMOTE, GIT_BRANCH
  ARRAY_CONCURRENCY, MAX_CONDITIONS_PER_ARRAY
  SMALL_PROFILES, MEDIUM_PROFILES, LARGE_PROFILES
USAGE
}

action="${1:-}"
case "${action}" in
  squeue)
    squeue -u "${USER}" -o "%.18i %.9P %.48j %.8T %.12M %.12l %.8C %.10m %.20R"
    ;;
  refresh)
    refresh_results
    ;;
  push-results)
    push_results
    ;;
  preview-small)
    submit_or_preview_tier small "${SMALL_PROFILES}" dry-run
    ;;
  preview-medium)
    submit_or_preview_tier medium "${MEDIUM_PROFILES}" dry-run
    ;;
  preview-large)
    submit_or_preview_tier large "${LARGE_PROFILES}" dry-run
    ;;
  submit-small)
    submit_or_preview_tier small "${SMALL_PROFILES}" submit
    ;;
  submit-medium)
    submit_or_preview_tier medium "${MEDIUM_PROFILES}" submit
    ;;
  submit-large)
    submit_or_preview_tier large "${LARGE_PROFILES}" submit
    ;;
  -h|--help|help|"")
    usage
    ;;
  *)
    echo "ERROR: unknown action: ${action}" >&2
    usage >&2
    exit 2
    ;;
esac
