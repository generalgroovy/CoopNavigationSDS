#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

submit_job() {
  local tier="$1"
  local job_file="$2"
  local wrapper="slurm/transformers_agent_b_${tier}_cpu_array.sbatch"
  echo "SUBMIT | ${tier} | ${job_file}"
  JOB_FILE="${job_file}" sbatch "${wrapper}"
}

if [[ "${1:-}" == "--small-medium" ]]; then
  tiers=(small medium)
else
  tiers=(small medium large)
fi

for tier in "${tiers[@]}"; do
  while IFS= read -r job_file; do
    submit_job "${tier}" "${job_file}"
  done < <(find "jobs/agent_b_llm/transformers_speech_grid/${tier}" -maxdepth 1 -type f -name '*.job' | sort)
done
