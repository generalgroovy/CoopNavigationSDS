#!/usr/bin/env bash
set -Eeuo pipefail

# Finalize completed large1/Qwen2.5-7B results after all experiment writers are
# quiet. The script never edits raw run folders. It regenerates derived analysis
# only from a stable result snapshot, commits/pushes model-wise result folders
# and compact analysis, and excludes oversized generated aggregates.

ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv-linux/bin/python}"
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT}/results}"
REMOTE="${REMOTE:-origin}"
BRANCH="${BRANCH:-main}"
GIT_KEY_PATH="${GIT_KEY_PATH:-${ROOT}/key2}"
STABILITY_SECONDS="${STABILITY_SECONDS:-120}"
ALLOW_ACTIVE_SLURM_JOBS="${ALLOW_ACTIVE_SLURM_JOBS:-0}"
REFRESH_ANALYSIS="${REFRESH_ANALYSIS:-1}"
FORCE_REFRESH="${FORCE_REFRESH:-1}"
MAX_GITHUB_FILE_BYTES="${MAX_GITHUB_FILE_BYTES:-95000000}"

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
        $2 ~ /finalize-large1|completed-results|push-results|prep-push|prepare-push/ { next }
        { print }
      '
}

result_snapshot() {
  run_python - "${RESULTS_ROOT}" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
summaries = list(root.rglob("run_summary.json"))
count = len(summaries)
latest = max((path.stat().st_mtime_ns for path in summaries), default=0)
size = sum(path.stat().st_size for path in root.rglob("*") if path.is_file()) if root.is_dir() else 0
print(f"{count}|{latest}|{size}")
PY
}

print_snapshot() {
  local label="$1"
  local snapshot="$2"
  IFS='|' read -r count latest size <<<"${snapshot}"
  echo "${label}_run_summary_count=${count}"
  echo "${label}_latest_run_summary_mtime_ns=${latest}"
  echo "${label}_results_size_bytes=${size}"
}

preflight() {
  step "1/7 Preflight"
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
  echo "Stability wait: ${STABILITY_SECONDS}s"
}

require_queue_quiet() {
  step "2/7 Verify Slurm writers are quiet"
  local active
  active="$(active_slurm_jobs || true)"
  if [[ -n "${active}" && "${ALLOW_ACTIVE_SLURM_JOBS}" != "1" ]]; then
    echo "${active}" >&2
    fail "active Slurm jobs remain. Wait for large1/experiment arrays to finish, or set ALLOW_ACTIVE_SLURM_JOBS=1 only for a deliberate partial push without analysis."
  fi
  if [[ -n "${active}" ]]; then
    echo "WARNING: active Slurm jobs ignored because ALLOW_ACTIVE_SLURM_JOBS=1"
    echo "${active}"
  else
    echo "slurm_queue=quiet_for_final_analysis"
  fi
}

require_stable_results() {
  step "3/7 Verify result evidence is stable"
  local before after
  before="$(result_snapshot)"
  print_snapshot "before" "${before}"
  sleep "${STABILITY_SECONDS}"
  after="$(result_snapshot)"
  print_snapshot "after" "${after}"
  [[ "${before}" == "${after}" ]] || fail "result evidence changed during stability window; postpone finalization until all writers are finished"
}

refresh_and_push() {
  step "4/7 Fetch remote"
  git fetch "${REMOTE}" "${BRANCH}"

  step "5/7 Refresh analysis and commit/push completed results"
  PROJECT_ROOT="${ROOT}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  RESULTS_ROOT="${RESULTS_ROOT}" \
  REMOTE="${REMOTE}" \
  BRANCH="${BRANCH}" \
  GIT_KEY_PATH="${GIT_KEY_PATH}" \
  REFRESH_ANALYSIS="${REFRESH_ANALYSIS}" \
  FORCE_REFRESH="${FORCE_REFRESH}" \
  MAX_GITHUB_FILE_BYTES="${MAX_GITHUB_FILE_BYTES}" \
  COMMIT_PREFIX="Finalize completed large1 cluster results" \
  bash scripts/cluster_push_completed_results.sh
}

write_final_manifest() {
  step "6/7 Final compact manifest"
  run_python - "${RESULTS_ROOT}" <<'PY'
from pathlib import Path
import json
import sys

root = Path(sys.argv[1])
payload = {
    "run_summary_count": len(list(root.rglob("run_summary.json"))),
    "coverage_summary_exists": (root / "experiment_coverage_summary.json").is_file(),
    "general_analysis_manifest_exists": (root / "general" / "analysis_manifest.json").is_file(),
    "comparison_analysis_manifest_exists": (root / "comparison" / "analysis_manifest.json").is_file(),
}
print(json.dumps(payload, indent=2, sort_keys=True))
PY
}

final_state() {
  step "7/7 Final state"
  git status --short --branch
  git log --oneline --decorate -5
}

preflight
require_queue_quiet
require_stable_results
refresh_and_push
write_final_manifest
final_state
