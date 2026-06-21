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
    "python314": ProviderProfile(
        "python314",
        ("chattts", "piper", "faster_whisper", "vosk", "qwen3_asr"),
        (
            "ChatTTS==0.2.5",
            "faster-whisper==1.2.1",
            "piper-tts==1.4.2",
            "qwen-asr==0.0.6",
            "vosk==0.3.45",
        ),
        "Speech libraries verified in the single Python 3.14 project runtime.",
    ),
}
