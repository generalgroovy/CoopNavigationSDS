#!/usr/bin/env bash
set -Eeuo pipefail

# Clean cluster workflow:
# 1. fail fast on Git locks or concurrent Git operations,
# 2. optionally preserve local result changes before pulling,
# 3. pull/rebase latest code,
# 4. run completed-dialogue metric/success analysis,
# 5. push only compact analysis outputs and safe result evidence.

ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv-linux/bin/python}"
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT}/results}"
REMOTE="${REMOTE:-origin}"
BRANCH="${BRANCH:-main}"
GIT_KEY_PATH="${GIT_KEY_PATH:-${ROOT}/key2}"
MAX_GITHUB_FILE_BYTES="${MAX_GITHUB_FILE_BYTES:-95000000}"
PRESERVE_LOCAL_RESULTS_BEFORE_PULL="${PRESERVE_LOCAL_RESULTS_BEFORE_PULL:-1}"
PUSH_RAW_RESULT_FOLDERS="${PUSH_RAW_RESULT_FOLDERS:-0}"

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

active_git_processes() {
  pgrep -u "$USER" -f "git (add|commit|rebase|merge|push|gc|pack-objects)" >/dev/null
}

ensure_git_ready() {
  if active_git_processes; then
    ps -u "$USER" -o pid,etime,pcpu,pmem,cmd \
      | grep -E "git (add|commit|rebase|merge|push|gc|pack-objects)" \
      | grep -v grep >&2 || true
    fail "another Git process is active"
  fi
  if [[ -e .git/index.lock || -d .git/rebase-merge || -d .git/rebase-apply ]]; then
    find .git -maxdepth 2 \( -name '*.lock' -o -name 'rebase-merge' -o -name 'rebase-apply' \) -print -ls >&2
    fail "Git lock or rebase state exists"
  fi
}

configure_git() {
  export GIT_SSH_COMMAND="ssh -i ${GIT_KEY_PATH} -o IdentitiesOnly=yes -o HostName=ssh.github.com -o Port=443"
  git config gc.auto 0
  git config core.autocrlf false
  git config core.eol lf
  git config user.name "${GIT_AUTHOR_NAME:-generalgroovy}"
  git config user.email "${GIT_AUTHOR_EMAIL:-generalgroovy@users.noreply.github.com}"
}

unstage_for_github_limits() {
  run_python - "${MAX_GITHUB_FILE_BYTES}" <<'PY'
import subprocess
import sys
from pathlib import Path

limit = int(sys.argv[1])
always_exclude = {
    "results/comparison/combined_metrics_long.csv",
    "results/comparison/metric_deltas.csv",
    "results/comparison/metric_summary.csv",
}
staged = subprocess.check_output(["git", "diff", "--cached", "--name-only"], text=True).splitlines()
for name in staged:
    path = Path(name)
    oversized = path.exists() and path.is_file() and path.stat().st_size > limit
    excluded = name.startswith("slurm/logs/") or name in always_exclude
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
        print(f"ERROR: staged file exceeds GitHub-safe limit: {name} {size}", flush=True)
    raise SystemExit(2)
PY
}

commit_if_staged() {
  local message="$1"
  if git diff --cached --quiet; then
    echo "nothing staged for commit"
    return 0
  fi
  git commit -m "${message}"
}

stage_existing_result_evidence() {
  [[ -d "${RESULTS_ROOT}" ]] || return 0
  git add -f -- "${RESULTS_ROOT}"
  unstage_for_github_limits
}

stage_completed_analysis_outputs() {
  git add -f -- \
    results/general/completed_dialogue_run_outcomes.csv \
    results/general/completed_dialogue_outcome_summary.csv \
    results/general/completed_metric_indicator_means.csv \
    results/general/metric_success_correlations_completed.csv \
    results/general/completed_metric_success_summary.md \
    results/general/completed_metric_success_manifest.json \
    docs/THESIS_WRITING_AID.md \
    scripts/analyze_completed_metric_success.py \
    scripts/cluster_clean_pull_analyze_push.sh \
    scripts/submit_completed_metric_analysis_push.sh \
    slurm/analyze_push_completed_metrics.sbatch \
    2>/dev/null || true
  unstage_for_github_limits
}

step "1/9 Preflight"
test -x "${PYTHON_BIN}" || fail "Python binary is not executable: ${PYTHON_BIN}"
test -d "${RESULTS_ROOT}" || fail "Results directory not found: ${RESULTS_ROOT}"
test -f "${GIT_KEY_PATH}" || fail "Git SSH key not found: ${GIT_KEY_PATH}"
ensure_git_ready
configure_git
echo "Project: ${ROOT}"
echo "Python:  ${PYTHON_BIN}"
echo "Results: ${RESULTS_ROOT}"
echo "Remote:  ${REMOTE}/${BRANCH}"

step "2/9 Inventory before pull"
run_python - "${RESULTS_ROOT}" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
print(f"run_summary_count={len(list(root.rglob('run_summary.json')))}")
print(f"conditions_count={len(list(root.rglob('conditions.jsonl')))}")
PY

step "3/9 Preserve local result evidence before pull"
if [[ "${PRESERVE_LOCAL_RESULTS_BEFORE_PULL}" == "1" ]]; then
  stage_existing_result_evidence
  commit_if_staged "Preserve cluster result evidence before analysis pull"
else
  echo "preserve_local_results_before_pull=skipped"
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Remaining local changes after result-preservation step:" >&2
  git status --short >&2
  fail "refusing to pull with non-result local changes"
fi

step "4/9 Pull latest code"
git fetch "${REMOTE}" "${BRANCH}"
git rebase "${REMOTE}/${BRANCH}"

step "5/9 Run completed-dialogue metric/success analysis"
run_python -u scripts/analyze_completed_metric_success.py \
  --results-dir "${RESULTS_ROOT}" \
  --output-dir "${RESULTS_ROOT}/general"

step "6/9 Stage compact completed-dialogue analysis"
stage_completed_analysis_outputs
commit_if_staged "Add completed-dialogue metric success analysis"

step "7/9 Optionally stage raw result folders"
if [[ "${PUSH_RAW_RESULT_FOLDERS}" == "1" ]]; then
  stage_existing_result_evidence
  commit_if_staged "Push completed cluster result folders"
else
  echo "raw_result_folder_push=skipped (PUSH_RAW_RESULT_FOLDERS=0)"
fi

step "8/9 Rebase and push"
git fetch "${REMOTE}" "${BRANCH}"
git rebase --autostash "${REMOTE}/${BRANCH}"
git push --progress "${REMOTE}" "HEAD:${BRANCH}"

step "9/9 Final state"
git status --short --branch
git log --oneline --decorate -5
