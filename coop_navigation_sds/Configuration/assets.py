"""Canonical local asset resolution shared by preflight and runtime."""
from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_asset_path(value):
    """Resolve project-managed relative asset paths independently of the shell cwd."""
    text = str(value or "").strip()
    if not text:
        return text
    path = Path(text).expanduser()
    if path.is_absolute():
        return str(path.resolve())
    if path.parts and path.parts[0] == ".speech-providers":
        return str((PROJECT_ROOT / path).resolve())
    return text


def resolve_faster_whisper_model(value):
    """Return the exact local CTranslate2 model directory when discoverable.

    Faster-Whisper cache roots contain Hugging Face snapshot directories.
    CTranslate2 must receive the snapshot containing ``model.bin``, not the
    parent cache directory.
    """
    resolved = resolve_project_asset_path(value)
    if not resolved:
        return resolved
    path = Path(resolved)
    if path.is_file() and path.name.casefold() == "model.bin":
        return str(path.parent.resolve())
    if not path.is_dir():
        return resolved
    if (path / "model.bin").is_file():
        return str(path.resolve())
    candidates = sorted(
        {
            model.parent.resolve()
            for model in path.rglob("model.bin")
            if model.is_file() and (model.parent / "config.json").is_file()
        },
        key=lambda candidate: str(candidate).casefold(),
    )
    return str(candidates[-1]) if candidates else str(path.resolve())


def faster_whisper_model_ready(value):
    """Return readiness and the canonical Faster-Whisper model location."""
    resolved = resolve_faster_whisper_model(value)
    path = Path(resolved) if resolved else None
    ready = bool(
        path
        and path.is_dir()
        and (path / "model.bin").is_file()
        and (path / "config.json").is_file()
    )
    return ready, resolved
