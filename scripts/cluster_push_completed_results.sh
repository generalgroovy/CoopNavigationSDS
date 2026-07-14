#!/usr/bin/env bash
set -Eeuo pipefail

# Prepare and push already completed run results without touching running Slurm
# jobs. Raw Slurm logs are intentionally not staged. Derived analysis is off by
# default because running experiment jobs can write new evidence while analysis
# reads the result tree. Set REFRESH_ANALYSIS=1 only after the queue is quiet.

ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv-linux/bin/python}"
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT}/results}"
REMOTE="${REMOTE:-origin}"
BRANCH="${BRANCH:-main}"
GIT_KEY_PATH="${GIT_KEY_PATH:-${ROOT}/key2}"
MAX_GITHUB_FILE_BYTES="${MAX_GITHUB_FILE_BYTES:-95000000}"
FORCE_REFRESH="${FORCE_REFRESH:-0}"
REFRESH_ANALYSIS="${REFRESH_ANALYSIS:-0}"
COMMIT_PREFIX="${COMMIT_PREFIX:-Push completed cluster results}"

cd "${ROOT}"

step() {
  printf '\n[%s] %s\n' "$(date --iso-8601=seconds)" "$*"
}

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

run_python() {
  printf '+ %s' "${PYTHON_BIN}"
  printf ' %q' "$@"
  printf '\n'
  "${PYTHON_BIN}" "$@"
}

preflight() {
  step "1/8 Preflight"
  test -x "${PYTHON_BIN}" || fail "Python binary is not executable: ${PYTHON_BIN}"
  test -d "${RESULTS_ROOT}" || fail "Results directory not found: ${RESULTS_ROOT}"
  test -f "${GIT_KEY_PATH}" || fail "Git SSH key not found: ${GIT_KEY_PATH}"
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
  echo "Running jobs are not cancelled, modified, or inspected beyond optional status output."
}

print_run_state() {
  step "2/8 Completed result inventory"
  run_python - "${RESULTS_ROOT}" <<'PY'
from pathlib import Path
import json
import sys

root = Path(sys.argv[1])
runs = sorted(root.rglob("run_summary.json"))
print(f"run_summary_count={len(runs)}")
if runs:
    newest = max(path.stat().st_mtime for path in runs)
    print(f"newest_run_summary_mtime={newest:.0f}")
summary = root / "experiment_coverage_summary.json"
if summary.is_file():
    data = json.loads(summary.read_text(encoding="utf-8"))
    print(f"coverage_updated_at={data.get('updated_at_utc')}")
    print(f"completed_run_count={data.get('completed_run_count')}")
    print(f"completed_selected={data.get('completed_planned_configuration_count')}/{data.get('planned_configuration_count')}")
PY
}

analysis_is_stale() {
  [[ "${FORCE_REFRESH}" == "1" ]] && return 0
  [[ "${REFRESH_ANALYSIS}" != "1" ]] && return 1
  "${PYTHON_BIN}" - "${RESULTS_ROOT}" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
run_summaries = list(root.rglob("run_summary.json"))
if not run_summaries:
    raise SystemExit(1)
targets = [
    root / "experiment_coverage_summary.json",
    root / "general" / "analysis_manifest.json",
    root / "comparison" / "analysis_manifest.json",
]
if any(not target.is_file() for target in targets):
    raise SystemExit(0)
newest_run = max(path.stat().st_mtime for path in run_summaries)
oldest_target = min(path.stat().st_mtime for path in targets)
raise SystemExit(0 if newest_run > oldest_target else 1)
PY
}

refresh_if_needed() {
  step "3/8 Refresh compact analysis if stale"
  if [[ "${REFRESH_ANALYSIS}" != "1" ]]; then
    echo "analysis_refresh=skipped (REFRESH_ANALYSIS=0; safe while Slurm jobs are running)"
    return 0
  fi
  if analysis_is_stale; then
    echo "analysis_refresh=needed"
    run_python -u scripts/update_experiment_coverage.py --results-dir "${RESULTS_ROOT}"
    run_python -u -m coop_navigation_sds.ResultsAndArtifacts.comparison \
      "${RESULTS_ROOT}" \
      --output "${RESULTS_ROOT}/comparison"
    run_python -u -m coop_navigation_sds.ResultsAndArtifacts.comparison \
      "${RESULTS_ROOT}" \
      --output "${RESULTS_ROOT}/general" \
      --include-partial
  else
    echo "analysis_refresh=skipped"
  fi
}

stage_one_path() {
  local path="$1"
  [[ -e "${path}" ]] || return 0
  git add -f -- "${path}"
  unstage_excluded
  if git diff --cached --quiet; then
    echo "nothing staged for ${path}"
    return 0
  fi
  git commit -m "${COMMIT_PREFIX}: ${path#${ROOT}/}"
  git pull --rebase "${REMOTE}" "${BRANCH}"
  git push --progress "${REMOTE}" "HEAD:${BRANCH}"
}

unstage_excluded() {
  "${PYTHON_BIN}" - "${MAX_GITHUB_FILE_BYTES}" <<'PY'
import subprocess
import sys
from pathlib import Path

limit = int(sys.argv[1])
always_exclude = {
    "results/comparison/combined_metrics_long.csv",
    "results/comparison/metric_deltas.csv",
}
staged = subprocess.check_output(["git", "diff", "--cached", "--name-only"], text=True).splitlines()
for name in staged:
    path = Path(name)
    excluded = name.startswith("slurm/logs/") or name in always_exclude
    oversized = path.exists() and path.is_file() and path.stat().st_size > limit
    if not (excluded or oversized):
        continue
    reason = "excluded" if excluded else f"oversized:{path.stat().st_size}"
    print(f"unstage {reason}: {name}", flush=True)
    subprocess.run(["git", "restore", "--staged", "--", name], check=False)
    subprocess.run(["git", "rm", "--cached", "--ignore-unmatch", "--", name], check=False)

bad = []
for name in subprocess.check_output(["git", "diff", "--cached", "--name-only"], text=True).splitlines():
    path = Path(name)
    if path.exists() and path.is_file() and path.stat().st_size > limit:
        bad.append((name, path.stat().st_size))
if bad:
    for name, size in bad:
        print(f"ERROR: staged file exceeds limit: {name} {size}", flush=True)
    raise SystemExit(2)
PY
}

discover_model_paths() {
  run_python - "${RESULTS_ROOT}" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
paths = set()
for summary in root.rglob("run_summary.json"):
    relative = summary.parent.relative_to(root)
    parts = relative.parts
    if not parts:
        continue
    if parts[0] == "agent_b" and len(parts) >= 3:
        paths.add(root / parts[0] / parts[1] / parts[2])
    elif parts[0][:2].isdigit() and len(parts) >= 1:
        paths.add(root / parts[0])
for path in sorted(paths):
    print(path)
PY
}

commit_model_results() {
  step "4/8 Commit and push completed run folders model-wise"
  mapfile -t model_paths < <(discover_model_paths)
  if (( ${#model_paths[@]} == 0 )); then
    echo "No completed model result folders discovered."
    return 0
  fi
  local index=0
  for path in "${model_paths[@]}"; do
    index=$((index + 1))
    echo "model_path ${index}/${#model_paths[@]}: ${path}"
    stage_one_path "${path}"
  done
}

commit_compact_analysis() {
  step "5/8 Commit and push compact analysis tables"
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
    results/comparison/*.csv.gz \
    results/comparison/*.html \
    results/comparison/*.json \
    2>/dev/null || true
  unstage_excluded
  if git diff --cached --quiet; then
    echo "No compact analysis changes staged."
  else
    git commit -m "${COMMIT_PREFIX}: compact analysis"
    git pull --rebase "${REMOTE}" "${BRANCH}"
    git push --progress "${REMOTE}" "HEAD:${BRANCH}"
  fi
}

final_state() {
  step "6/8 Final state"
  git status --short --branch
  git log --oneline --decorate -5

  step "7/8 Running Slurm jobs left untouched"
  squeue -u "$USER" -o "%.18i %.12P %.32j %.8T %.10M %.10l %.6C %.8m %.24R" || true

  step "8/8 Done"
}

preflight
print_run_state
git fetch "${REMOTE}" "${BRANCH}"
refresh_if_needed
commit_model_results
commit_compact_analysis
final_state
