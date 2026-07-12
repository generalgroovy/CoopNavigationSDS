#!/usr/bin/env bash
set -Eeuo pipefail

# Slurm-safe result refresh and Git push helper.
# Integrity rules:
# - never deletes or edits canonical run folders;
# - regenerates only derived coverage/comparison views;
# - stages results and slurm/logs;
# - excludes generated files over GitHub's 100 MB limit from the commit;
# - preserves raw per-run evidence, which is sufficient to regenerate excluded
#   aggregate comparison files.

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv-linux/bin/python}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/results}"
GIT_KEY_PATH="${GIT_KEY_PATH:-${PROJECT_ROOT}/key2}"
GIT_REMOTE="${GIT_REMOTE:-origin}"
GIT_BRANCH="${GIT_BRANCH:-main}"
MAX_GITHUB_FILE_MB="${MAX_GITHUB_FILE_MB:-95}"
COMMIT_MESSAGE="${COMMIT_MESSAGE:-Refresh cluster results and logs}"

cd "${PROJECT_ROOT}"
mkdir -p slurm/logs

step() {
  printf '\n[%s] %s\n' "$(date --iso-8601=seconds)" "$1"
}

require_path() {
  [[ -e "$1" ]] || {
    echo "ERROR: missing required path: $1" >&2
    exit 2
  }
}

run_python() {
  printf '+ %s' "${PYTHON_BIN}"
  printf ' %q' "$@"
  printf '\n'
  "${PYTHON_BIN}" "$@"
}

configure_git() {
  git config core.autocrlf false
  git config core.eol lf
  git config user.name "${GIT_AUTHOR_NAME:-generalgroovy}"
  git config user.email "${GIT_AUTHOR_EMAIL:-generalgroovy@users.noreply.github.com}"
}

preflight_git_state() {
  step "Git state preflight"
  require_path "${RESULTS_ROOT}"
  if [[ -d .git/rebase-merge || -d .git/rebase-apply ]]; then
    echo "ERROR: repository is in a rebase state. Finish or abort the rebase before pushing results." >&2
    exit 4
  fi
  if [[ -e .git/index.lock ]]; then
    echo "ERROR: .git/index.lock exists. Check for active git processes before removing stale locks." >&2
    exit 4
  fi
  git status --short --branch
}

configure_ssh() {
  require_path "${GIT_KEY_PATH}"
  export GIT_SSH_COMMAND="ssh -i ${GIT_KEY_PATH} -o IdentitiesOnly=yes -o HostName=ssh.github.com -o Port=443"
  eval "$(ssh-agent -s)"
  ssh-add "${GIT_KEY_PATH}"
}

print_run_summary() {
  run_python - "${RESULTS_ROOT}" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
run_summaries = list(root.rglob("run_summary.json")) if root.is_dir() else []
print(f"run_summary files: {len(run_summaries)}")
if root.is_dir():
    total = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    print(f"results size MB: {total / (1024 * 1024):.1f}")

manifest = root / "general" / "analysis_manifest.json"
if manifest.is_file():
    data = json.loads(manifest.read_text(encoding="utf-8"))
    print(
        "analysis manifest:",
        f"runs={data.get('discovered_run_count')}",
        f"observed={data.get('observed_condition_count')}",
        f"completed={data.get('completed_condition_count')}",
    )

inventory = root / "general" / "run_inventory.csv"
if inventory.is_file():
    rows = list(csv.DictReader(inventory.open(encoding="utf-8")))
    print(f"inventory rows: {len(rows)}")
PY
}

refresh_results() {
  step "Refresh coverage registry"
  run_python -u scripts/update_experiment_coverage.py --results-dir "${RESULTS_ROOT}"

  step "Refresh finalized-run comparison views"
  if find "${RESULTS_ROOT}" -name run_summary.json -print -quit | grep -q .; then
    run_python -u -m coop_navigation_sds.ResultsAndArtifacts.comparison \
      "${RESULTS_ROOT}" \
      --output "${RESULTS_ROOT}/comparison"
  else
    echo "No finalized run_summary.json files found; skipping finalized comparison."
  fi

  step "Refresh partial-aware general views"
  run_python -u -m coop_navigation_sds.ResultsAndArtifacts.comparison \
    "${RESULTS_ROOT}" \
    --output "${RESULTS_ROOT}/general" \
    --include-partial

  step "Result summary"
  print_run_summary
}

stage_results_without_oversized_generated_files() {
  step "Stage results and Slurm logs"
  git add -A "${RESULTS_ROOT}" slurm/logs

  step "Unstage generated files over ${MAX_GITHUB_FILE_MB} MB"
  local threshold_bytes=$((MAX_GITHUB_FILE_MB * 1024 * 1024))
  local oversized
  oversized="$(
    python - "${threshold_bytes}" <<'PY' || true
import os
import subprocess
import sys

threshold = int(sys.argv[1])
raw = subprocess.check_output(["git", "diff", "--cached", "--name-only", "-z"])
for item in raw.split(b"\0"):
    if not item:
        continue
    path = item.decode("utf-8", errors="replace")
    try:
        size = os.path.getsize(path)
    except OSError:
        continue
    if size > threshold:
        print(path)
PY
  )"
  if [[ -n "${oversized}" ]]; then
    echo "${oversized}" | while IFS= read -r path; do
      [[ -n "${path}" ]] || continue
      echo "UNSTAGE oversized generated file: ${path}"
      git restore --staged -- "${path}" || git rm --cached --ignore-unmatch -- "${path}"
    done
  else
    echo "No staged file exceeds ${MAX_GITHUB_FILE_MB} MB."
  fi

  step "Verify staged file sizes"
  python - "${threshold_bytes}" <<'PY'
import os
import subprocess
import sys

threshold = int(sys.argv[1])
too_large = []
raw = subprocess.check_output(["git", "diff", "--cached", "--name-only", "-z"])
for item in raw.split(b"\0"):
    if not item:
        continue
    path = item.decode("utf-8", errors="replace")
    try:
        size = os.path.getsize(path)
    except OSError:
        continue
    if size > threshold:
        too_large.append((path, size))
if too_large:
    print("ERROR: staged files still exceed limit:", file=sys.stderr)
    for path, size in too_large:
        print(f"  {path}: {size / (1024 * 1024):.1f} MB", file=sys.stderr)
    raise SystemExit(3)
print("Staged file-size check passed.")
PY
}

commit_and_push() {
  configure_git
  configure_ssh
  preflight_git_state

  step "Fetch remote before committing"
  git fetch "${GIT_REMOTE}" "${GIT_BRANCH}"

  stage_results_without_oversized_generated_files

  step "Commit staged changes"
  if git diff --cached --quiet; then
    echo "No staged changes to commit."
  else
    git commit -m "${COMMIT_MESSAGE}"
  fi

  step "Rebase onto ${GIT_REMOTE}/${GIT_BRANCH}"
  git fetch "${GIT_REMOTE}" "${GIT_BRANCH}"
  git rebase "${GIT_REMOTE}/${GIT_BRANCH}"

  step "Push"
  git push --progress "${GIT_REMOTE}" HEAD:"${GIT_BRANCH}"
}

case "${1:-all}" in
  refresh)
    refresh_results
    ;;
  push)
    commit_and_push
    ;;
  all)
    refresh_results
    commit_and_push
    ;;
  *)
    echo "Usage: bash scripts/cluster_prepare_push_results_safe.sh [refresh|push|all]" >&2
    exit 2
    ;;
esac
