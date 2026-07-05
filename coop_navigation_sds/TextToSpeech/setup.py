"""Provider-environment catalog and setup utilities."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderProfile:
    key: str
    engines: tuple[str, ...]
    packages: tuple[str, ...]
    description: str


PROVIDER_PROFILES = {
    "project_python": ProviderProfile(
        "project_python",
        ("chattts", "piper", "faster_whisper", "vosk", "qwen3_asr"),
        (
            "ChatTTS==0.2.5",
            "faster-whisper==1.2.1",
            "piper-tts==1.4.2",
            "qwen-asr==0.0.6",
            "vosk==0.3.45",
        ),
        "Speech libraries installed in the supported project Python runtime.",
    ),
}

PROJECT_PROVIDER_PROFILE = "project_python"
MINIMUM_PROJECT_PYTHON = (3, 11)
MAXIMUM_PROJECT_PYTHON_EXCLUSIVE = (3, 15)


def project_python_supported(version):
    """Return whether a Python major/minor can host project speech providers."""
    normalized = tuple(int(value) for value in tuple(version)[:2])
    return MINIMUM_PROJECT_PYTHON <= normalized < MAXIMUM_PROJECT_PYTHON_EXCLUSIVE
