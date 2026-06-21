"""whisper.cpp runtime discovery shared by preflight and live ASR."""
from __future__ import annotations

import json
from pathlib import Path
import shutil


EXECUTABLE_NAMES = (
    "whisper-cli.exe",
    "whisper-cli",
    "main.exe",
    "main",
)
MODEL_PATTERNS = (
    "ggml-base.en.bin",
    "ggml-small.en.bin",
    "ggml-base.bin",
    "ggml-small.bin",
    "*.bin",
)


def _provider_root(environment_dir):
    return Path(environment_dir or ".speech-providers").expanduser()


def _existing_file(value, base=None):
    raw = str(value or "").strip()
    if not raw:
        return ""
    path = Path(raw)
    if not path.is_absolute() and base is not None:
        path = Path(base) / path
    return str(path.resolve()) if path.is_file() else ""


def _valid_executable_path(value):
    path = Path(str(value or ""))
    if not path.is_file():
        return ""
    return str(path.resolve()) if path.name.lower() in EXECUTABLE_NAMES else ""


def _manifest_entry(environment_dir):
    manifest_path = _provider_root(environment_dir) / "providers.json"
    if not manifest_path.is_file():
        return {}, manifest_path.parent
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}, manifest_path.parent
    entry = document.get("providers", {}).get("whisper_cpp", {})
    return (entry if isinstance(entry, dict) else {}), manifest_path.parent


def _path_candidates(root, names):
    provider = root / "whisper_cpp"
    directories = (
        provider / "bin",
        provider / "build" / "bin",
        provider / "build",
        provider,
        root,
    )
    for directory in directories:
        for name in names:
            yield directory / name


def _model_candidates(root):
    provider = root / "whisper_cpp"
    directories = (
        provider / "models",
        provider,
        root / "models",
        root,
    )
    for directory in directories:
        for pattern in MODEL_PATTERNS:
            yield from sorted(directory.glob(pattern))


def resolve_whisper_cpp_paths(
    executable="",
    model="",
    vad_model="",
    environment_dir=".speech-providers",
):
    """Resolve whisper.cpp executable and model paths from config, manifest, or provider folder."""
    root = _provider_root(environment_dir)
    entry, manifest_base = _manifest_entry(environment_dir)
    configured_executable = _existing_file(executable)
    configured_model = _existing_file(model)
    configured_vad = _existing_file(vad_model)

    manifest_executable = _existing_file(entry.get("executable"), manifest_base)
    manifest_model = _existing_file(entry.get("model"), manifest_base)
    manifest_vad = _existing_file(entry.get("vad_model"), manifest_base)

    path_executable = ""
    for name in EXECUTABLE_NAMES:
        found = shutil.which(name)
        valid = _valid_executable_path(found)
        if valid:
            path_executable = valid
            break
    provider_executable = next(
        (str(path.resolve()) for path in _path_candidates(root, EXECUTABLE_NAMES) if path.is_file()),
        "",
    )
    provider_model = next(
        (str(path.resolve()) for path in _model_candidates(root) if path.is_file()),
        "",
    )

    return {
        "executable": configured_executable or manifest_executable or path_executable or provider_executable,
        "model": configured_model or manifest_model or provider_model,
        "vad_model": configured_vad or manifest_vad,
        "source": {
            "environment_dir": str(root),
            "manifest": str((_provider_root(environment_dir) / "providers.json").resolve()),
        },
    }


def whisper_cpp_ready(
    executable="",
    model="",
    vad_model="",
    environment_dir=".speech-providers",
):
    """Return readiness and human-readable diagnostics for whisper.cpp."""
    resolved = resolve_whisper_cpp_paths(
        executable=executable,
        model=model,
        vad_model=vad_model,
        environment_dir=environment_dir,
    )
    executable_ok = bool(resolved["executable"])
    model_ok = bool(resolved["model"])
    ready = executable_ok and model_ok
    message = (
        "whisper.cpp executable and GGML model resolved"
        if ready
        else "whisper.cpp needs whisper-cli/main and a GGML .bin model"
    )
    return ready, message, resolved
