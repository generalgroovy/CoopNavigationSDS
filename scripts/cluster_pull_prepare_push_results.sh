#!/usr/bin/env bash
set -Eeuo pipefail

# Safely update the cluster checkout, then push completed result folders.
# Default behavior refuses to pull while this project's Slurm jobs are active,
# because running jobs may read code/configuration from the same checkout.

ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv-linux/bin/python}"
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT}/results}"
REMOTE="${REMOTE:-origin}"
BRANCH="${BRANCH:-main}"
GIT_KEY_PATH="${GIT_KEY_PATH:-${ROOT}/key2}"
ALLOW_WHILE_JOBS_RUNNING="${ALLOW_WHILE_JOBS_RUNNING:-0}"
REFRESH_ANALYSIS="${REFRESH_ANALYSIS:-0}"
MAX_GITHUB_FILE_BYTES="${MAX_GITHUB_FILE_BYTES:-95000000}"

cd "${ROOT}"

step() {
  printf '\n[%s] %s\n' "$(date --iso-8601=seconds)" "$*"
}

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

active_project_jobs() {
  command -v squeue >/dev/null 2>&1 || return 1
  squeue -u "$USER" -h -o "%i|%j|%Z|%T" \
    | awk -F'|' -v root="${ROOT}" '
      $4 ~ /RUNNING|PENDING|CONFIGURING|COMPLETING/ && $3 == root && $2 !~ /push|prep|prepare|coverage|analysis/ {
        print
      }'
}

step "1/7 Preflight"
test -x "${PYTHON_BIN}" || fail "Python binary is not executable: ${PYTHON_BIN}"
test -d "${RESULTS_ROOT}" || fail "Results directory not found: ${RESULTS_ROOT}"
test -f "${GIT_KEY_PATH}" || fail "Git SSH key not found: ${GIT_KEY_PATH}"
test -x "scripts/cluster_push_completed_results.sh" || chmod +x scripts/cluster_push_completed_results.sh

if pgrep -u "$USER" -f "git (add|commit|rebase|merge|push|gc|pack-objects)" >/dev/null; then
  ps -u "$USER" -o pid,etime,pcpu,pmem,cmd \
    | grep -E "git (add|commit|rebase|merge|push|gc|pack-objects)" \
    | grep -v grep >&2 || true
  fail "another Git process is active"
fi
if [[ -e .git/index.lock || -d .git/rebase-merge || -d .git/rebase-apply ]]; then
  find .git -maxdepth 2 \( -name '*.lock' -o -name 'rebase-merge' -o -name 'rebase-apply' \) -print -ls >&2
  fail "Git lock or rebase state exists"
fi

running_jobs="$(active_project_jobs || true)"
if [[ -n "${running_jobs}" && "${ALLOW_WHILE_JOBS_RUNNING}" != "1" ]]; then
  echo "${running_jobs}" >&2
  fail "project Slurm jobs are active; wait for them to finish or set ALLOW_WHILE_JOBS_RUNNING=1"
fi

step "2/7 Configure Git"
export GIT_SSH_COMMAND="ssh -i ${GIT_KEY_PATH} -o IdentitiesOnly=yes -o HostName=ssh.github.com -o Port=443"
git config gc.auto 0
git config core.autocrlf false
git config core.eol lf
git config user.name "${GIT_AUTHOR_NAME:-generalgroovy}"
git config user.email "${GIT_AUTHOR_EMAIL:-generalgroovy@users.noreply.github.com}"

step "3/7 Pull latest project code"
git fetch "${REMOTE}" "${BRANCH}"
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Uncommitted local changes before pull:" >&2
  git status --short >&2
  fail "commit or stash local changes before pulling"
fi
git rebase "${REMOTE}/${BRANCH}"

step "4/7 Result inventory"
"${PYTHON_BIN}" - "${RESULTS_ROOT}" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
runs = list(root.rglob("run_summary.json"))
print(f"run_summary_count={len(runs)}")
if runs:
    newest = max(runs, key=lambda path: path.stat().st_mtime)
    print(f"newest_run_summary={newest}")
PY

step "5/7 Push completed result folders"
REFRESH_ANALYSIS="${REFRESH_ANALYSIS}" \
MAX_GITHUB_FILE_BYTES="${MAX_GITHUB_FILE_BYTES}" \
PROJECT_ROOT="${ROOT}" \
PYTHON_BIN="${PYTHON_BIN}" \
RESULTS_ROOT="${RESULTS_ROOT}" \
REMOTE="${REMOTE}" \
BRANCH="${BRANCH}" \
GIT_KEY_PATH="${GIT_KEY_PATH}" \
bash scripts/cluster_push_completed_results.sh

step "6/7 Final result inventory"
"${PYTHON_BIN}" - "${RESULTS_ROOT}" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
print(f"run_summary_count={len(list(root.rglob('run_summary.json')))}")
PY

step "7/7 Done"
git status --short --branch
git log --oneline --decorate -5
