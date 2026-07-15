#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv-linux/bin/python}"
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT}/results}"
WRAPPER="${WRAPPER:-${ROOT}/slurm/finalize_relevant_results.sbatch}"
PARTITION="${PARTITION:-standard}"
CPUS="${CPUS:-2}"
MEMORY="${MEMORY:-8G}"
TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
STABILITY_SECONDS="${STABILITY_SECONDS:-120}"

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
    --export=ALL,PROJECT_ROOT="${ROOT}",PYTHON_BIN="${PYTHON_BIN}",RESULTS_ROOT="${RESULTS_ROOT}",STABILITY_SECONDS="${STABILITY_SECONDS}" \
    "${WRAPPER}"
)"

echo "submitted_job_id=${job_id}"
echo
echo "Monitor with low overhead:"
echo "  JOB_ID=${job_id} COUNT_RUNS=0 INTERVAL_SECONDS=120 bash scripts/monitor_finalize_relevant_results.sh"
