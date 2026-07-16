#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/beegfs/home/users/g/generalgroovy/experiments/CoopNavigationSDS}"
REMOTE="${REMOTE:-origin}"
BRANCH="${BRANCH:-main}"
GIT_KEY_PATH="${GIT_KEY_PATH:-${PROJECT_ROOT}/key2}"
MAX_BYTES="${MAX_BYTES:-95000000}"

cd "${PROJECT_ROOT}"

echo "[1/9] Configure Git SSH"
test -f "${GIT_KEY_PATH}" || { echo "ERROR: missing Git key: ${GIT_KEY_PATH}"; exit 1; }
export GIT_SSH_COMMAND="ssh -i ${GIT_KEY_PATH} -o IdentitiesOnly=yes -o HostName=ssh.github.com -o Port=443"

echo "[2/9] Verify repository and results"
git rev-parse --show-toplevel
test -d results || { echo "ERROR: results directory missing"; exit 1; }

echo "[3/9] Write handoff manifests"
HANDOFF_DIR="results/_handoff"
mkdir -p "${HANDOFF_DIR}"

cat > "${HANDOFF_DIR}/handoff_links.txt" <<EOF
GitHub project: https://github.com/generalgroovy/CoopNavigationSDS
Git SSH remote: ssh://ssh.github.com:443/generalgroovy/CoopNavigationSDS.git
TU cluster login: ssh generalgroovy@gateway.hpc.tu-berlin.de
TU cluster project path: ${PROJECT_ROOT}
Generated: $(date -Iseconds)
EOF

cat > "${HANDOFF_DIR}/excluded_scope.txt" <<'EOF'
Not included in GitHub handoff:
- Local Python environments: .venv*, .runtime, Miniforge installers.
- Model and speech assets: .speech-providers, .model-providers, Hugging Face caches, Ollama caches.
- Python/build caches: __pycache__, .pytest_cache, .pytest_tmp.
- Generated aggregate files larger than 95 MB, because GitHub rejects files over 100 MB.
- Obsolete thesis-out-of-scope model results, especially large2 / Mistral-style configurations, unless already included in active comparison summaries.
Included:
- Source code, scripts, jobs, Slurm wrappers, documentation.
- Thesis-relevant completed run folders for selected Agent B models.
- Per-run evidence: run summaries, conditions, transcripts, metrics, logs, runtime state, and compact analysis files.
- Slurm logs where available and below the size limit.
EOF

THESIS_RESULT_PATHS=(
  "results/01-small-tinyllama-1.1b"
  "results/01-small-qwen2.5-0.5b"
  "results/02-medium-qwen2.5-1.5b"
  "results/02-medium-phi3-mini"
  "results/03-large-qwen2.5-7b"
  "results/agent_b/userlm_transformers/01-small-tinyllama-1.1b"
  "results/agent_b/userlm_transformers/01-small-qwen2.5-0.5b"
  "results/agent_b/userlm_transformers/02-medium-qwen2.5-1.5b"
  "results/agent_b/userlm_transformers/02-medium-phi3-mini"
  "results/agent_b/userlm_transformers/03-large-qwen2.5-7b"
)

: > "${HANDOFF_DIR}/run_summary_counts_by_model.txt"
for p in "${THESIS_RESULT_PATHS[@]}"; do
  if [ -d "$p" ]; then
    c="$(find "$p" -name run_summary.json | wc -l | tr -d ' ')"
    echo "$p: $c run_summary.json files" | tee -a "${HANDOFF_DIR}/run_summary_counts_by_model.txt"
  else
    echo "$p: missing" >> "${HANDOFF_DIR}/run_summary_counts_by_model.txt"
  fi
done

squeue -u "$USER" -o "%.18i %.12P %.40j %.8T %.12M %.12l %.6C %.10m %.30R" > "${HANDOFF_DIR}/slurm_jobs_at_handoff.txt" 2>/dev/null || true
git status --short --branch > "${HANDOFF_DIR}/git_state_before_handoff.txt" || true

echo "[4/9] Fetch remote"
git fetch "${REMOTE}" "${BRANCH}"

echo "[5/9] Stage source, docs, jobs, Slurm files, selected results"
stage_if_exists() {
  for p in "$@"; do
    [ -e "$p" ] && git add -A -- "$p"
  done
}
stage_force_if_exists() {
  for p in "$@"; do
    [ -e "$p" ] && git add -f -- "$p"
  done
}

stage_if_exists \
  coop_navigation_sds tests scripts jobs slurm README.md docs \
  API_REFERENCE.md METRIC_REFERENCE.md pyproject.toml requirements.txt

stage_force_if_exists \
  results/general results/comparison results/_handoff \
  results/agent_model_combination_coverage.csv \
  results/agent_model_combination_coverage.html \
  results/experiment_case_coverage.csv \
  results/experiment_coverage.csv \
  results/experiment_coverage.html \
  results/experiment_coverage_conditions.csv \
  results/experiment_coverage_matrix.csv \
  results/experiment_coverage_runs.csv \
  results/experiment_coverage_summary.json \
  slurm/logs

for p in "${THESIS_RESULT_PATHS[@]}"; do
  [ -e "$p" ] && git add -f -- "$p"
done

echo "[6/9] Exclude staged files over ${MAX_BYTES} bytes"
python3 - <<PY
import os, subprocess, pathlib
max_bytes = int("${MAX_BYTES}")
excluded = pathlib.Path("${HANDOFF_DIR}/excluded_files_over_95mb.txt")
files = subprocess.check_output(["git", "diff", "--cached", "--name-only"], text=True).splitlines()
large = []
for f in files:
    if os.path.isfile(f) and os.path.getsize(f) > max_bytes:
        large.append((f, os.path.getsize(f)))
with excluded.open("w", encoding="utf-8") as out:
    for f, size in large:
        out.write(f"{size}\t{f}\n")
for f, _ in large:
    subprocess.run(["git", "reset", "-q", "HEAD", "--", f], check=False)
print(f"excluded_large_files={len(large)}")
PY

git add -f -- "${HANDOFF_DIR}/excluded_files_over_95mb.txt"
git diff --cached --name-only > "${HANDOFF_DIR}/included_paths.txt"
git add -f -- "${HANDOFF_DIR}/included_paths.txt"

echo "[7/9] Commit staged handoff state"
if git diff --cached --quiet; then
  echo "No staged changes. Repository may already contain current handoff state."
else
  git commit -m "Handoff thesis project state and relevant results"
fi

echo "[8/9] Rebase safely, preserving unstaged oversized local files"
git pull --rebase --autostash "${REMOTE}" "${BRANCH}"

echo "[9/9] Push"
git push --progress "${REMOTE}" "HEAD:${BRANCH}"

echo
echo "Handoff push complete."
cat "${HANDOFF_DIR}/handoff_links.txt"
echo
echo "Excluded oversized files are listed in: ${HANDOFF_DIR}/excluded_files_over_95mb.txt"
