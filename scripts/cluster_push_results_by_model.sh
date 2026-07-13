#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REMOTE="${REMOTE:-origin}"
BRANCH="${BRANCH:-main}"
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT}/results}"
GIT_KEY_PATH="${GIT_KEY_PATH:-${ROOT}/key2}"
MAX_GITHUB_FILE_BYTES="${MAX_GITHUB_FILE_BYTES:-95000000}"
MODEL_FILTER="${MODEL_FILTER:-}"

cd "${ROOT}"

step() {
  printf '\n[%s] %s\n' "$(date --iso-8601=seconds)" "$*"
}

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

git_quiet() {
  git "$@" >/dev/null
}

step "Preflight"
test -d "${RESULTS_ROOT}" || fail "results root not found: ${RESULTS_ROOT}"
test -f "${GIT_KEY_PATH}" || fail "git key not found: ${GIT_KEY_PATH}"
if pgrep -u "$USER" -f "git (add|commit|rebase|merge|push|gc|pack-objects)" >/dev/null; then
  ps -u "$USER" -o pid,etime,pcpu,pmem,cmd | grep -E "git (add|commit|rebase|merge|push|gc|pack-objects)" | grep -v grep >&2 || true
  fail "another git process is active"
fi
if [[ -e .git/index.lock || -d .git/rebase-merge || -d .git/rebase-apply ]]; then
  find .git -maxdepth 2 \( -name '*.lock' -o -name 'rebase-merge' -o -name 'rebase-apply' \) -print -ls >&2
  fail "git lock or rebase state exists"
fi

export GIT_SSH_COMMAND="ssh -i ${GIT_KEY_PATH} -o IdentitiesOnly=yes -o HostName=ssh.github.com -o Port=443"
git config gc.auto 0
git fetch "${REMOTE}" "${BRANCH}"
git rebase "${REMOTE}/${BRANCH}"

step "Discover model result folders"
mapfile -t MODEL_DIRS < <(
  {
    find "${RESULTS_ROOT}/agent_b" -mindepth 2 -maxdepth 3 -type d 2>/dev/null || true
    find "${RESULTS_ROOT}" -mindepth 1 -maxdepth 2 -type d -name '[0-9][0-9]-*' 2>/dev/null || true
  } | while read -r directory; do
    find "$directory" -name run_summary.json -print -quit | grep -q . && printf '%s\n' "$directory"
  done | sort -u
)

if [[ -n "${MODEL_FILTER}" ]]; then
  mapfile -t MODEL_DIRS < <(printf '%s\n' "${MODEL_DIRS[@]}" | grep -i -- "${MODEL_FILTER}" || true)
fi

(( ${#MODEL_DIRS[@]} > 0 )) || fail "no model result folders found"
printf 'model folders: %s\n' "${#MODEL_DIRS[@]}"
printf '  %s\n' "${MODEL_DIRS[@]}"

unstage_oversized_and_excess() {
  python3 - "$MAX_GITHUB_FILE_BYTES" <<'PY'
import subprocess
import sys
from pathlib import Path

limit = int(sys.argv[1])
excess_names = {
    "results/comparison/combined_metrics_long.csv",
    "results/comparison/metric_deltas.csv",
}
staged = subprocess.check_output(["git", "diff", "--cached", "--name-only"], text=True).splitlines()
for name in staged:
    path = Path(name)
    if name.startswith("slurm/logs/"):
        subprocess.run(["git", "restore", "--staged", "--", name], check=False)
        continue
    if name in excess_names:
        subprocess.run(["git", "restore", "--staged", "--", name], check=False)
        subprocess.run(["git", "rm", "--cached", "--ignore-unmatch", "--", name], check=False)
        continue
    if path.exists() and path.is_file() and path.stat().st_size > limit:
        print(f"unstage oversized: {name} ({path.stat().st_size} bytes)", flush=True)
        subprocess.run(["git", "restore", "--staged", "--", name], check=False)

bad = []
for name in subprocess.check_output(["git", "diff", "--cached", "--name-only"], text=True).splitlines():
    path = Path(name)
    if path.exists() and path.is_file() and path.stat().st_size > limit:
        bad.append((name, path.stat().st_size))
if bad:
    for name, size in bad:
        print(f"ERROR: staged file over limit: {name} {size}", flush=True)
    raise SystemExit(2)
PY
}

commit_and_push_if_needed() {
  local message="$1"
  if git diff --cached --quiet; then
    echo "nothing staged"
    return 0
  fi
  git commit -m "$message"
  git pull --rebase "${REMOTE}" "${BRANCH}"
  git push --progress "${REMOTE}" "HEAD:${BRANCH}"
}

index=0
for directory in "${MODEL_DIRS[@]}"; do
  index=$((index + 1))
  relative="${directory#${ROOT}/}"
  label="$(echo "$relative" | tr '/\\' '__' | tr -cd '[:alnum:]_.-' | cut -c1-80)"
  step "Model ${index}/${#MODEL_DIRS[@]}: ${relative}"
  git add -f -- "$relative"
  unstage_oversized_and_excess
  commit_and_push_if_needed "Push results for ${label}"
done

step "Stage compact top-level analysis only"
git add -f -- \
  results/experiment_coverage_summary.json \
  results/experiment_coverage_conditions.csv \
  results/experiment_coverage_runs.csv \
  results/experiment_coverage_matrix.csv \
  results/experiment_case_coverage.csv \
  results/agent_model_combination_coverage.csv \
  results/agent_model_combination_coverage.html \
  results/general \
  results/comparison/*.csv \
  results/comparison/*.html \
  2>/dev/null || true
unstage_oversized_and_excess
commit_and_push_if_needed "Push compact result analysis"

step "Done"
git status --short --branch
git log --oneline --decorate -5
