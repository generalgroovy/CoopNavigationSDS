$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Test-Path "$env:ProgramFiles\eSpeak NG\espeak-ng.exe")) {
    winget install --id eSpeak-NG.eSpeak-NG --exact --accept-package-agreements --accept-source-agreements --silent
}

python -m pip install -r requirements.txt
python -m pip install -r requirements-speech-optional.txt
python scripts/setup_speech_providers.py
python scripts/prepare_test_environment.py
python -m pytest -q
python scripts/run_speech_backend_matrix.py --live
