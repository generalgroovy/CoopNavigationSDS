#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
JOB_ID="${1:-${JOB_ID:-}}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-120}"
COUNT_RUNS="${COUNT_RUNS:-0}"

cd "${ROOT}"

if [[ -z "${JOB_ID}" ]]; then
  JOB_ID="$(
    squeue -u "$USER" -h -o "%A %j" \
      | awk '$2 ~ /finalize-large1/ {print $1; exit}'
  )"
fi

if [[ -z "${JOB_ID}" ]]; then
  echo "ERROR: no job id passed and no finalize-large1 job found." >&2
  echo "Usage: JOB_ID=<id> bash scripts/monitor_finalize_large1_results.sh" >&2
  exit 2
fi

while true; do
  clear || true
  echo "=== finalize-large1 monitor $(date --iso-8601=seconds) ==="
  echo "job_id=${JOB_ID}"
  echo

  echo "[1] Slurm state"
  squeue -j "${JOB_ID}" -o "%.18i %.12P %.32j %.8T %.10M %.10l %.6C %.8m %.24R" || true
  echo

  stdout="$(ls -t slurm/logs/finalize-large1-results-"${JOB_ID}".out 2>/dev/null | head -1 || true)"
  stderr="$(ls -t slurm/logs/finalize-large1-results-"${JOB_ID}".err 2>/dev/null | head -1 || true)"

  echo "[2] Logs"
  echo "stdout=${stdout:-missing}"
  echo "stderr=${stderr:-missing}"
  if [[ -n "${stdout}" && -f "${stdout}" ]]; then
    stat -c "stdout_size=%s stdout_mtime=%y" "${stdout}" 2>/dev/null || true
    echo "--- latest stdout stage ---"
    grep -E "^\[[0-9]{4}-|^[0-9]+/[0-9]+|^before_|^after_|^slurm_queue=|^analysis_refresh=|^model_path|^run_summary_count|^Project:" "${stdout}" \
      | tail -30 || tail -30 "${stdout}"
  fi
  if [[ -n "${stderr}" && -s "${stderr}" ]]; then
    echo "--- latest stderr ---"
    tail -40 "${stderr}"
  else
    echo "stderr=empty-or-missing"
  fi
  echo

  echo "[3] Active related processes"
  ps -u "$USER" -o pid,etime,pcpu,pmem,stat,cmd \
    | grep -E "cluster_finalize_large1_results|cluster_push_completed_results|update_experiment_coverage|ResultsAndArtifacts.comparison|git (add|commit|pull|rebase|push|pack-objects)" \
    | grep -v grep || true
  echo

  if [[ "${COUNT_RUNS}" == "1" ]]; then
    echo "[4] Run count"
    find results -name run_summary.json | wc -l
  else
    echo "[4] Run count skipped (COUNT_RUNS=0)"
  fi
  echo

  if ! squeue -j "${JOB_ID}" -h >/dev/null 2>&1; then
    echo "Job is no longer in squeue. Final log tail above."
    break
  fi
  echo "Sleeping ${INTERVAL_SECONDS}s. Ctrl+C stops monitor only."
  sleep "${INTERVAL_SECONDS}"
done
