#!/usr/bin/env bash
set -Eeuo pipefail

# Finalize all thesis-relevant result folders after experiment writers are
# quiet. Integrity rules:
# - raw run folders are never edited;
# - derived analysis is regenerated only after a stable result snapshot;
# - only selected thesis-scope model folders and compact analysis files are
#   staged;
# - oversized generated aggregates are excluded because raw evidence remains
#   sufficient to regenerate them.

ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv-linux/bin/python}"
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT}/results}"
REMOTE="${REMOTE:-origin}"
BRANCH="${BRANCH:-main}"
GIT_KEY_PATH="${GIT_KEY_PATH:-${ROOT}/key2}"
STABILITY_SECONDS="${STABILITY_SECONDS:-120}"
ALLOW_ACTIVE_SLURM_JOBS="${ALLOW_ACTIVE_SLURM_JOBS:-0}"
MAX_GITHUB_FILE_BYTES="${MAX_GITHUB_FILE_BYTES:-95000000}"
COMMIT_MESSAGE="${COMMIT_MESSAGE:-Finalize relevant cluster results and analysis}"

RELEVANT_RESULT_DIRS="${RELEVANT_RESULT_DIRS:-01-small-tinyllama-1.1b 01-small-qwen2.5-0.5b 02-medium-qwen2.5-1.5b 02-medium-phi3-mini 03-large-qwen2.5-7b}"

cd "${ROOT}"

step() {
  printf '\n[%s] %s\n' "$(date --iso-8601=seconds)" "$*"
}

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

run_python() {
  printf '+ %s' "${PYTHON_BIN}" >&2
  printf ' %q' "$@" >&2
  printf '\n' >&2
  "${PYTHON_BIN}" "$@"
}

active_slurm_jobs() {
  local own_job="${SLURM_JOB_ID:-}"
  squeue -u "$USER" -h -o "%A|%j|%T|%R" \
    | awk -F'|' -v own="${own_job}" '
        own != "" && $1 == own { next }
        $2 ~ /finalize|completed-results|push-results|prep-push|prepare-push/ { next }
        { print }
      '
}

result_snapshot() {
  run_python - "${RESULTS_ROOT}" ${RELEVANT_RESULT_DIRS} <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
selected = sys.argv[2:]
paths = [root / name for name in selected]
summaries = []
size = 0
for path in paths:
    if not path.exists():
        continue
    summaries.extend(path.rglob("run_summary.json"))
    size += sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
latest = max((path.stat().st_mtime_ns for path in summaries), default=0)
print(f"{len(summaries)}|{latest}|{size}")
PY
}

print_snapshot() {
  local label="$1"
  local snapshot="$2"
  IFS='|' read -r count latest size <<<"${snapshot}"
  echo "${label}_relevant_run_summary_count=${count}"
  echo "${label}_latest_run_summary_mtime_ns=${latest}"
  echo "${label}_relevant_results_size_bytes=${size}"
}

preflight() {
  step "1/9 Preflight"
  test -x "${PYTHON_BIN}" || fail "Python binary is not executable: ${PYTHON_BIN}"
  test -d "${RESULTS_ROOT}" || fail "Results directory not found: ${RESULTS_ROOT}"
  test -f "${GIT_KEY_PATH}" || fail "Git key not found: ${GIT_KEY_PATH}"
  mkdir -p slurm/logs "${RESULTS_ROOT}/_transfer"

  if [[ -e .git/index.lock || -d .git/rebase-merge || -d .git/rebase-apply ]]; then
    find .git -maxdepth 2 \( -name '*.lock' -o -name 'rebase-merge' -o -name 'rebase-apply' \) -print -ls >&2
    fail "Git lock or rebase state exists"
  fi
  if pgrep -u "$USER" -f "git (add|commit|rebase|merge|push|gc|pack-objects)" >/dev/null; then
    ps -u "$USER" -o pid,etime,pcpu,pmem,cmd \
      | grep -E "git (add|commit|rebase|merge|push|gc|pack-objects)" \
      | grep -v grep >&2 || true
    fail "another Git process is active"
  fi

  git config gc.auto 0
  git config core.autocrlf false
  git config core.eol lf
  git config user.name "${GIT_AUTHOR_NAME:-generalgroovy}"
  git config user.email "${GIT_AUTHOR_EMAIL:-generalgroovy@users.noreply.github.com}"
  export GIT_SSH_COMMAND="ssh -i ${GIT_KEY_PATH} -o IdentitiesOnly=yes -o HostName=ssh.github.com -o Port=443"

  echo "Project: ${ROOT}"
  echo "Python:  ${PYTHON_BIN}"
  echo "Results: ${RESULTS_ROOT}"
  echo "Remote:  ${REMOTE}/${BRANCH}"
  echo "Relevant result dirs: ${RELEVANT_RESULT_DIRS}"
  echo "Stability wait: ${STABILITY_SECONDS}s"
}

require_queue_quiet() {
  step "2/9 Verify experiment writers are quiet"
  local active
  active="$(active_slurm_jobs || true)"
  if [[ -n "${active}" && "${ALLOW_ACTIVE_SLURM_JOBS}" != "1" ]]; then
    echo "${active}" >&2
    fail "active Slurm jobs remain. Wait for experiment arrays to finish before final analysis/push."
  fi
  if [[ -n "${active}" ]]; then
    echo "WARNING: active Slurm jobs ignored because ALLOW_ACTIVE_SLURM_JOBS=1"
    echo "${active}"
  else
    echo "slurm_queue=quiet_for_final_analysis"
  fi
}

require_stable_results() {
  step "3/9 Verify relevant result evidence is stable"
  local before after
  before="$(result_snapshot)"
  print_snapshot "before" "${before}"
  sleep "${STABILITY_SECONDS}"
  after="$(result_snapshot)"
  print_snapshot "after" "${after}"
  [[ "${before}" == "${after}" ]] || fail "relevant result evidence changed during stability window"
}

refresh_analysis() {
  step "4/9 Refresh derived analysis from stable evidence"
  run_python -u scripts/update_experiment_coverage.py --results-dir "${RESULTS_ROOT}"
  run_python -u -m coop_navigation_sds.ResultsAndArtifacts.comparison \
    "${RESULTS_ROOT}" \
    --output "${RESULTS_ROOT}/comparison"
  run_python -u -m coop_navigation_sds.ResultsAndArtifacts.comparison \
    "${RESULTS_ROOT}" \
    --output "${RESULTS_ROOT}/general" \
    --include-partial
}

stage_relevant_results_and_analysis() {
  step "5/9 Stage selected result folders and compact analysis together"
  git reset --quiet

  for name in ${RELEVANT_RESULT_DIRS}; do
    if [[ -d "${RESULTS_ROOT}/${name}" ]]; then
      echo "stage result_dir=${RESULTS_ROOT}/${name}"
      git add -f -- "${RESULTS_ROOT}/${name}"
    else
      echo "skip missing result_dir=${RESULTS_ROOT}/${name}"
    fi
  done

  git add -f -- \
    "${RESULTS_ROOT}/experiment_coverage_summary.json" \
    "${RESULTS_ROOT}/experiment_coverage_conditions.csv" \
    "${RESULTS_ROOT}/experiment_coverage_runs.csv" \
    "${RESULTS_ROOT}/experiment_coverage_matrix.csv" \
    "${RESULTS_ROOT}/experiment_case_coverage.csv" \
    "${RESULTS_ROOT}/agent_model_combination_coverage.csv" \
    "${RESULTS_ROOT}/agent_model_combination_coverage.html" \
    "${RESULTS_ROOT}/general" \
    "${RESULTS_ROOT}/comparison" \
    2>/dev/null || true
}

unstage_excluded() {
  step "6/9 Exclude oversized/generated nonessential files"
  run_python - "${MAX_GITHUB_FILE_BYTES}" <<'PY'
from pathlib import Path
import subprocess
import sys

limit = int(sys.argv[1])
always_exclude = {
    "results/comparison/combined_metrics_long.csv",
    "results/comparison/metric_deltas.csv",
    "results/comparison/metric_summary.csv",
}
staged = subprocess.check_output(["git", "diff", "--cached", "--name-only"], text=True).splitlines()
for name in staged:
    path = Path(name)
    excluded = name.startswith("slurm/logs/") or name in always_exclude
    oversized = path.exists() and path.is_file() and path.stat().st_size > limit
    if excluded or oversized:
        reason = "excluded" if excluded else f"oversized:{path.stat().st_size}"
        print(f"unstage {reason}: {name}", flush=True)
        subprocess.run(["git", "restore", "--staged", "--", name], check=False)

bad = []
for name in subprocess.check_output(["git", "diff", "--cached", "--name-only"], text=True).splitlines():
    path = Path(name)
    if path.exists() and path.is_file() and path.stat().st_size > limit:
        bad.append((name, path.stat().st_size))
if bad:
    for name, size in bad:
        print(f"ERROR: staged file exceeds limit: {name} {size}", flush=True)
    raise SystemExit(2)
print("staged_oversized_check=passed")
PY
}

commit_and_push() {
  step "7/9 Commit coherent result snapshot"
  if git diff --cached --quiet; then
    echo "No relevant result or analysis changes staged."
  else
    git commit -m "${COMMIT_MESSAGE}"
  fi

  step "8/9 Rebase and push"
  git fetch "${REMOTE}" "${BRANCH}"
  git rebase --autostash "${REMOTE}/${BRANCH}"
  git push --progress "${REMOTE}" "HEAD:${BRANCH}"
}

final_state() {
  step "9/9 Final state"
  git status --short --branch
  git log --oneline --decorate -5
}

preflight
require_queue_quiet
require_stable_results
git fetch "${REMOTE}" "${BRANCH}"
refresh_analysis
stage_relevant_results_and_analysis
unstage_excluded
commit_and_push
final_state
