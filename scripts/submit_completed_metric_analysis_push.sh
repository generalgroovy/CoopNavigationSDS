#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv-linux/bin/python}"
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT}/results}"
WRAPPER="${WRAPPER:-${ROOT}/slurm/analyze_push_completed_metrics.sbatch}"
PARTITION="${PARTITION:-standard}"
CPUS="${CPUS:-2}"
MEMORY="${MEMORY:-8G}"
TIME_LIMIT="${TIME_LIMIT:-01:30:00}"

cd "${ROOT}"
mkdir -p slurm/logs

test -f "${WRAPPER}" || {
  echo "ERROR: missing wrapper: ${WRAPPER}" >&2
  exit 1
}

job_id="$(
  sbatch \
    --parsable \
    --partition="${PARTITION}" \
    --cpus-per-task="${CPUS}" \
    --mem="${MEMORY}" \
    --time="${TIME_LIMIT}" \
    --export=ALL,PROJECT_ROOT="${ROOT}",PYTHON_BIN="${PYTHON_BIN}",RESULTS_ROOT="${RESULTS_ROOT}" \
    "${WRAPPER}"
)"

echo "submitted_job_id=${job_id}"
echo
echo "Monitor:"
echo "  squeue -j ${job_id} -o \"%.18i %.12P %.32j %.8T %.10M %.10l %.6C %.8m %.24R\""
echo "  tail -f slurm/logs/analyze-push-completed-${job_id}.out"
echo "  tail -f slurm/logs/analyze-push-completed-${job_id}.err"
