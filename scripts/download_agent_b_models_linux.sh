#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${project_root}/.venv/bin/python"
if [[ ! -x "${python_bin}" ]]; then
  python_bin="python3"
fi

exec "${python_bin}" "${project_root}/scripts/setup_agent_b_models.py" --pull "$@"
