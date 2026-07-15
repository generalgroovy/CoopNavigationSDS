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
      | awk '$2 ~ /finalize-relevant/ {print $1; exit}'
  )"
fi

if [[ -z "${JOB_ID}" ]]; then
  echo "ERROR: no job id passed and no finalize-relevant job found." >&2
  exit 2
fi

while true; do
  clear || true
  echo "=== finalize-relevant monitor $(date --iso-8601=seconds) ==="
  echo "job_id=${JOB_ID}"
  echo
  squeue -j "${JOB_ID}" -o "%.18i %.12P %.32j %.8T %.10M %.10l %.6C %.8m %.24R" || true
  echo

  stdout="$(ls -t slurm/logs/finalize-relevant-results-"${JOB_ID}".out 2>/dev/null | head -1 || true)"
  stderr="$(ls -t slurm/logs/finalize-relevant-results-"${JOB_ID}".err 2>/dev/null | head -1 || true)"
  echo "stdout=${stdout:-missing}"
  echo "stderr=${stderr:-missing}"
  if [[ -n "${stdout}" && -f "${stdout}" ]]; then
    stat -c "stdout_size=%s stdout_mtime=%y" "${stdout}" 2>/dev/null || true
    grep -E "^\[[0-9]{4}-|^before_|^after_|^stage result_dir=|^skip missing|^unstage |^staged_oversized_check=|^slurm_queue=|^No relevant|^To ssh|^To https|^Project:" "${stdout}" \
      | tail -40 || tail -40 "${stdout}"
  fi
  if [[ -n "${stderr}" && -s "${stderr}" ]]; then
    echo "--- stderr tail ---"
    tail -40 "${stderr}"
  else
    echo "stderr=empty-or-missing"
  fi
  echo

  echo "[active related processes]"
  ps -u "$USER" -o pid,etime,pcpu,pmem,stat,cmd \
    | grep -E "cluster_finalize_relevant_results|update_experiment_coverage|ResultsAndArtifacts.comparison|git (add|commit|pull|rebase|push|pack-objects)" \
    | grep -v grep || true

  if [[ "${COUNT_RUNS}" == "1" ]]; then
    echo "[run count]"
    find results -name run_summary.json | wc -l
  fi

  if ! squeue -j "${JOB_ID}" -h >/dev/null 2>&1; then
    echo "Job is no longer in squeue. Final log tail above."
    break
  fi
  echo "Sleeping ${INTERVAL_SECONDS}s. Ctrl+C stops monitor only."
  sleep "${INTERVAL_SECONDS}"
done
