#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv-linux/bin/python}"
GIT_KEY_PATH="${GIT_KEY_PATH:-${ROOT}/key2}"
REMOTE="${REMOTE:-origin}"
BRANCH="${BRANCH:-main}"
MAX_GITHUB_FILE_BYTES="${MAX_GITHUB_FILE_BYTES:-95000000}"

cd "${ROOT}"

step() {
  printf '\n[%s] %s\n' "$(date --iso-8601=seconds)" "$*"
}

step "Preflight"
test -x "${PYTHON_BIN}" || { echo "ERROR: Python not executable: ${PYTHON_BIN}" >&2; exit 2; }
test -d results || { echo "ERROR: results directory missing" >&2; exit 2; }
if pgrep -u "$USER" -f "git (add|commit|rebase|merge|push|gc|pack-objects)" >/dev/null; then
  echo "ERROR: another git process is active for this user; inspect before pushing." >&2
  ps -u "$USER" -o pid,etime,pcpu,pmem,cmd | grep -E "git (add|commit|rebase|merge|push|gc|pack-objects)" | grep -v grep >&2 || true
  exit 3
fi
if [[ -e .git/index.lock || -d .git/rebase-merge || -d .git/rebase-apply ]]; then
  echo "ERROR: git lock or rebase state exists; clean it manually after verifying no git process is active." >&2
  find .git -maxdepth 2 \( -name '*.lock' -o -name 'rebase-merge' -o -name 'rebase-apply' \) -print -ls >&2
  exit 4
fi

step "Configure SSH and disable repository auto-gc"
export GIT_SSH_COMMAND="ssh -i ${GIT_KEY_PATH} -o IdentitiesOnly=yes -o HostName=ssh.github.com -o Port=443"
git config gc.auto 0
git fetch "${REMOTE}" "${BRANCH}"

step "Refresh compact Slurm summaries"
"${PYTHON_BIN}" -u scripts/dedupe_slurm_logs.py --log-dir slurm/logs --output-dir results/general

step "Stage results and compact cluster state"
git add -f results scripts slurm README.md docs 2>/dev/null || git add -f results scripts slurm README.md
git restore --staged -- slurm/logs 2>/dev/null || true
git add -f results/general/slurm_error_summary.txt \
  results/general/slurm_error_signature_summary.csv \
  results/general/slurm_log_group_summary.csv \
  results/general/slurm_log_samples 2>/dev/null || true

step "Remove oversized generated aggregate files from the index"
"${PYTHON_BIN}" - <<'PY'
import os
import subprocess
from pathlib import Path

limit = int(os.environ.get("MAX_GITHUB_FILE_BYTES", "95000000"))
generated_patterns = {
    "results/comparison/combined_metrics_long.csv",
    "results/comparison/metric_deltas.csv",
}
staged = subprocess.check_output(["git", "diff", "--cached", "--name-only"], text=True).splitlines()
too_large = []
for name in staged:
    path = Path(name)
    if path.exists() and path.is_file() and path.stat().st_size > limit:
        too_large.append(name)
for name in too_large:
    subprocess.run(["git", "restore", "--staged", "--", name], check=False)
    if name in generated_patterns:
        subprocess.run(["git", "rm", "--cached", "--ignore-unmatch", "--", name], check=False)
if too_large:
    print("excluded_over_limit=" + ",".join(too_large), flush=True)
PY

step "Verify no staged file exceeds GitHub hard limit"
"${PYTHON_BIN}" - <<'PY'
import os
import subprocess
from pathlib import Path
limit = int(os.environ.get("MAX_GITHUB_FILE_BYTES", "95000000"))
bad = []
for name in subprocess.check_output(["git", "diff", "--cached", "--name-only"], text=True).splitlines():
    path = Path(name)
    if path.exists() and path.is_file() and path.stat().st_size > limit:
        bad.append((name, path.stat().st_size))
if bad:
    for name, size in bad:
        print(f"ERROR: staged file too large: {name} {size}", flush=True)
    raise SystemExit(5)
print("staged oversized files: none", flush=True)
PY

step "Commit if needed"
if git diff --cached --quiet; then
  echo "Nothing staged; results already represented in git."
else
  git commit -m "${COMMIT_MESSAGE:-Push compact cluster results and analysis}"
fi

step "Rebase and push"
git rebase "${REMOTE}/${BRANCH}"
git push --progress "${REMOTE}" "HEAD:${BRANCH}"

step "Done"
git log --oneline --decorate -3
