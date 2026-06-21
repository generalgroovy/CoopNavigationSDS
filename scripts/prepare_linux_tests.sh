#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v espeak-ng >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y espeak-ng
  else
    echo "Install eSpeak NG with the platform package manager before continuing." >&2
    exit 2
  fi
fi

python -m pip install -r requirements.txt
python -m pip install -r requirements-speech-optional.txt
python scripts/setup_speech_providers.py
python scripts/prepare_test_environment.py
python -m pytest -q
python scripts/run_speech_backend_matrix.py --live
