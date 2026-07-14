#!/usr/bin/env bash
set -euo pipefail

# Safely align a cluster checkout and submit only uncovered selected thesis
# conditions. This script does not delete results. It reads the existing
# coverage CSV and delegates condition selection to
# scripts/submit_remaining_selected_no_large2.py.

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv-linux/bin/python}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/results}"
REMOTE="${REMOTE:-origin}"
BRANCH="${BRANCH:-main}"
GIT_KEY_PATH="${GIT_KEY_PATH:-${PROJECT_ROOT}/key2}"
PARTITION="${PARTITION:-standard}"
MAX_CONDITIONS_PER_ARRAY="${MAX_CONDITIONS_PER_ARRAY:-12}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-1}"
SUBMIT="${SUBMIT:-0}"
CANCEL_PROJECT_ARRAYS="${CANCEL_PROJECT_ARRAYS:-0}"

cd "${PROJECT_ROOT}"

step() {
  printf '\n[%s] %s\n' "$(date --iso-8601=seconds)" "$*"
}

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

step "1/8 Preflight"
test -x "${PYTHON_BIN}" || fail "Python binary is not executable: ${PYTHON_BIN}"
test -d "${RESULTS_ROOT}" || fail "Results directory does not exist: ${RESULTS_ROOT}"
test -f "${RESULTS_ROOT}/experiment_coverage_conditions.csv" || fail "Coverage file missing. Prepare/push/pull compact results first."
test -f "scripts/submit_remaining_selected_no_large2.py" || fail "Missing submit helper. Pull latest project code first."
test -f "slurm/agent_b_model_cpu_array.sbatch" || fail "Missing Slurm wrapper."

export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export ONNXRUNTIME_THREAD_POOL_SIZE="${ONNXRUNTIME_THREAD_POOL_SIZE:-1}"
export ORT_LOG_SEVERITY_LEVEL="${ORT_LOG_SEVERITY_LEVEL:-3}"

echo "Project: ${PROJECT_ROOT}"
echo "Python:  ${PYTHON_BIN}"
echo "Results: ${RESULTS_ROOT}"
echo "Mode:    $([[ "${SUBMIT}" == "1" ]] && echo submit || echo preview)"

step "2/8 Configure Git SSH if key exists"
if [[ -f "${GIT_KEY_PATH}" ]]; then
  export GIT_SSH_COMMAND="ssh -i ${GIT_KEY_PATH} -o IdentitiesOnly=yes -o HostName=ssh.github.com -o Port=443"
  echo "Git key: ${GIT_KEY_PATH}"
else
  echo "Git key not found; continuing without overriding SSH: ${GIT_KEY_PATH}"
fi

step "3/8 Detect active Git operations"
if pgrep -u "$USER" -f "git (add|commit|rebase|merge|push|gc|pack-objects)" >/dev/null; then
  ps -u "$USER" -o pid,etime,pcpu,pmem,cmd | grep -E "git (add|commit|rebase|merge|push|gc|pack-objects)" | grep -v grep >&2 || true
  fail "another Git process is active; do not submit while repository state may be changing"
fi
if [[ -e .git/index.lock || -d .git/rebase-merge || -d .git/rebase-apply ]]; then
  find .git -maxdepth 2 \( -name '*.lock' -o -name 'rebase-merge' -o -name 'rebase-apply' \) -print -ls >&2
  fail "Git lock/rebase state exists; resolve it before submitting more jobs"
fi

step "4/8 Pull code only when working tree is clean"
git fetch "${REMOTE}" "${BRANCH}"
if git diff --quiet && git diff --cached --quiet; then
  git merge --ff-only "${REMOTE}/${BRANCH}"
else
  echo "Working tree has local changes. Skipping merge to protect local results/state."
  git status --short
fi

step "5/8 Optional cancellation of old project arrays"
if [[ "${CANCEL_PROJECT_ARRAYS}" == "1" ]]; then
  echo "Cancelling only matching CoopNavigationSDS remaining-coverage arrays."
  squeue -u "$USER" -h -o "%A %j" \
    | awk '$2 ~ /^(rem-|std-main|std-pres|fix-main|fix-pres|main-|pres-)/ {print $1}' \
    | sort -u \
    | while read -r job_id; do
        [[ -n "${job_id}" ]] || continue
        echo "scancel ${job_id}"
        scancel "${job_id}" || true
      done
else
  echo "No jobs cancelled. Set CANCEL_PROJECT_ARRAYS=1 to cancel old matching arrays first."
fi

step "6/8 Current Slurm queue"
squeue -u "$USER" -o "%.18i %.12P %.32j %.8T %.10M %.10l %.6C %.8m %.24R" || true

step "7/8 Preview uncovered selected conditions"
"${PYTHON_BIN}" -u scripts/submit_remaining_selected_no_large2.py \
  --coverage-file "${RESULTS_ROOT}/experiment_coverage_conditions.csv" \
  --results-dir "${RESULTS_ROOT}" \
  --python-bin "${PYTHON_BIN}" \
  --partition "${PARTITION}" \
  --max-conditions-per-array "${MAX_CONDITIONS_PER_ARRAY}" \
  --array-concurrency "${ARRAY_CONCURRENCY}"

step "8/8 Submit if requested"
if [[ "${SUBMIT}" == "1" ]]; then
  "${PYTHON_BIN}" -u scripts/submit_remaining_selected_no_large2.py \
    --coverage-file "${RESULTS_ROOT}/experiment_coverage_conditions.csv" \
    --results-dir "${RESULTS_ROOT}" \
    --python-bin "${PYTHON_BIN}" \
    --partition "${PARTITION}" \
    --max-conditions-per-array "${MAX_CONDITIONS_PER_ARRAY}" \
    --array-concurrency "${ARRAY_CONCURRENCY}" \
    --submit
  echo
  echo "Submitted. Monitor with:"
  echo '  squeue -u "$USER" -o "%.18i %.12P %.32j %.8T %.10M %.10l %.6C %.8m %.24R"'
  echo '  tail -n 80 slurm/logs/slurm-*.err'
else
  echo
  echo "Preview only. To submit:"
  echo "  SUBMIT=1 PARTITION=${PARTITION} MAX_CONDITIONS_PER_ARRAY=${MAX_CONDITIONS_PER_ARRAY} ARRAY_CONCURRENCY=${ARRAY_CONCURRENCY} bash scripts/cluster_unify_submit_remaining.sh"
  echo
  echo "To cancel old matching arrays before submitting:"
  echo "  CANCEL_PROJECT_ARRAYS=1 SUBMIT=1 bash scripts/cluster_unify_submit_remaining.sh"
fi
