"""Speech transport abstractions and simulated ASR/TTS transformations for dialog experiments.
"""
from dataclasses import dataclass
from difflib import SequenceMatcher
from contextlib import redirect_stdout
import hashlib
from io import StringIO
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import shutil
import struct
import subprocess
import sys
import time
from typing import Protocol
import wave

from coop_navigation_sds.Configuration.assets import (
    faster_whisper_model_ready,
    resolve_faster_whisper_model,
)
from coop_navigation_sds.Configuration.speech import (
    DEFAULT_SPEECH_PATTERN,
    SPEECH_AUDIO_DIR,
    SPEECH_ASR_ENGINE,
    SPEECH_PLAYBACK_ENABLED,
    SPEECH_REALTIME_ENABLED,
    SPEECH_TTS_ENGINE,
    speech_pattern_settings,
)
from coop_navigation_sds.DialogManagement.whisper_cpp_runtime import resolve_whisper_cpp_paths
from coop_navigation_sds.TextToSpeech.personas import (
    DEFAULT_AGENT_A_AUDIO_PERSONA,
    DEFAULT_AGENT_B_AUDIO_PERSONA,
    synthesis_values,
)
from coop_navigation_sds.NaturalLanguageUnderstanding.transcript_normalization import (
    normalize_transit_transcript,
    transcript_token_changes,
)

SYNTHESIS_CONTROL_KEYS = (
    "voice",
    "speech_rate",
    "words_per_minute",
    "volume",
    "pitch_semitones",
    "pause_ms",
    "emphasis",
)

PUNCTUATION_PAUSE_PATTERNS = (
    (r",", 0.55),
    (r";", 0.75),
    (r"\.+", 1.0),
    (r"!+", 1.05),
    (r"\?+", 1.15),
)


def synthesis_controls(prosody):
    """Return only arguments accepted by a synthesis backend."""
    return {
        key: prosody[key]
        for key in SYNTHESIS_CONTROL_KEYS
        if key in prosody
    }


def punctuation_pause_duration_sec(text, pause_ms):
    """Return punctuation-weighted pause time for natural cadence."""
    base_seconds = max(0, int(pause_ms)) / 1000.0
    return sum(
        len(re.findall(pattern, str(text or ""))) * factor * base_seconds
        for pattern, factor in PUNCTUATION_PAUSE_PATTERNS
    )


_SMALL_NUMBER_WORDS = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
)
_TENS_WORDS = ("", "", "twenty", "thirty", "forty", "fifty")


def _number_under_sixty(value):
    value = int(value)
    if value < 20:
        return _SMALL_NUMBER_WORDS[value]
    tens, units = divmod(value, 10)
    return _TENS_WORDS[tens] if not units else f"{_TENS_WORDS[tens]} {_SMALL_NUMBER_WORDS[units]}"


def normalize_text_for_speech(text):
    """Convert compact clock notation into recognizer-friendly spoken English."""
    def replace_time(match):
        hour = int(match.group(1))
        minute_text = match.group(2)
        minute = int(minute_text)
        if minute == 0:
            spoken_minute = "o'clock"
        elif minute < 10:
            spoken_minute = f"oh {_SMALL_NUMBER_WORDS[minute]}"
        else:
            spoken_minute = _number_under_sixty(minute)
        return f"{_number_under_sixty(hour)} {spoken_minute}"

    spoken = re.sub(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", replace_time, str(text))
    spoken = re.sub(r"\s+([,;.!?])", r"\1", spoken)
    spoken = re.sub(r"[ \t]+", " ", spoken).strip()
    if spoken and spoken[-1] not in ".!?":
        spoken += "."
    return spoken


class SpeechPipelineError(RuntimeError):
    """Raised when a strict speech pipeline stage fails."""

    def __init__(self, message, diagnostics=None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


@dataclass(frozen=True)
class SpeechEngineSpec:
    """Selectable speech backend metadata used by runtime and configuration UI."""

    key: str
    kind: str
    label: str
    description: str
    optional_dependency: bool = False
    windows_only: bool = False


TTS_ENGINE_SPECS = {
    "sapi": SpeechEngineSpec(
        "sapi", "tts", "Windows SAPI", "Native Windows synthesis with voice and prosody controls.",
        windows_only=True,
    ),
    "chattts": SpeechEngineSpec(
        "chattts", "tts", "ChatTTS", "Neural conversational synthesis with reproducible speaker sampling.",
        optional_dependency=True,
    ),
    "piper": SpeechEngineSpec(
        "piper", "tts", "Piper", "Fast local synthesis from an explicitly configured Piper ONNX voice.",
        optional_dependency=True,
    ),
    "espeak_ng": SpeechEngineSpec(
        "espeak_ng", "tts", "eSpeak NG", "Small cross-platform command-line synthesizer with explicit voice, rate, pitch, and volume controls.",
        optional_dependency=True,
    ),
    "coqui": SpeechEngineSpec(
        "coqui", "tts", "Coqui TTS", "Neural synthesis through the Coqui TTS Python API or an isolated provider environment.",
        optional_dependency=True,
    ),
    "file": SpeechEngineSpec(
        "file", "tts", "Deterministic WAV", "Dependency-free reproducible WAV carrier for tests and experiments.",
    ),
}

ASR_ENGINE_SPECS = {
    "sapi": SpeechEngineSpec(
        "sapi", "asr", "Windows SAPI", "Native Windows recognition with domain phrase hints.",
        windows_only=True,
    ),
    "faster_whisper": SpeechEngineSpec(
        "faster_whisper", "asr", "Faster-Whisper", "Local neural transcription with configurable model, device, and beam size.",
        optional_dependency=True,
    ),
    "vosk": SpeechEngineSpec(
        "vosk", "asr", "Vosk", "Low-latency offline recognition suitable for CPU and constrained systems.",
        optional_dependency=True,
    ),
    "whisper_cpp": SpeechEngineSpec(
        "whisper_cpp", "asr", "whisper.cpp", "Portable quantized Whisper recognition through whisper-cli.",
        optional_dependency=True,
    ),
    "qwen3_asr": SpeechEngineSpec(
        "qwen3_asr", "asr", "Qwen3-ASR", "Multilingual neural recognition with robust offline transcription.",
        optional_dependency=True,
    ),
    "sherpa_onnx": SpeechEngineSpec(
        "sherpa_onnx", "asr", "sherpa-onnx", "Portable offline ONNX recognition with transducer, Whisper, or Paraformer model directories.",
        optional_dependency=True,
    ),
    "file": SpeechEngineSpec(
        "file", "asr", "Deterministic sidecar", "Reads the generated transcript sidecar for reproducible tests.",
    ),
}


def available_tts_engine_keys():
    """Return selectable text-to-speech implementations."""
    return tuple(TTS_ENGINE_SPECS)


def available_asr_engine_keys():
    """Return selectable automatic speech recognition implementations."""
    return tuple(ASR_ENGINE_SPECS)


def speech_engine_description(kind, key):
    """Return concise help for one configured speech backend."""
    registry = TTS_ENGINE_SPECS if kind == "tts" else ASR_ENGINE_SPECS
    spec = registry.get(str(key or "").strip().lower())
    return spec.description if spec else "Custom or unsupported speech backend."


def resolve_espeak_executable(configured=""):
    if str(configured or "").strip():
        return str(configured).strip()
    discovered = shutil.which("espeak-ng") or shutil.which("espeak")
    if discovered:
        return discovered
    if platform.system() == "Windows":
        installed = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "eSpeak NG" / "espeak-ng.exe"
        if installed.is_file():
            return str(installed)
    return ""


def platform_default_tts_engine():
    """Select an installed backend, with a dependency-free final fallback."""
    if platform.system() == "Windows":
        return "sapi"
    if resolve_espeak_executable():
        return "espeak_ng"
    return "file"


def platform_default_asr_engine():
    """Select an installed backend, with a dependency-free final fallback."""
    if platform.system() == "Windows":
        return "sapi"
    if importlib.util.find_spec("faster_whisper") is not None:
        return "faster_whisper"
    if importlib.util.find_spec("sherpa_onnx") is not None:
        return "sherpa_onnx"
    return "file"


def _prepare_optional_audio_runtime():
    """Expose WinGet SoX installs and silence an irrelevant Qwen fallback banner."""
    if platform.system() == "Windows" and shutil.which("sox") is None:
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            packages = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
            matches = sorted(packages.glob("ChrisBagwell.SoX_*/sox-*/sox.exe"))
            if matches:
                os.environ["PATH"] = f"{matches[-1].parent}{os.pathsep}{os.environ.get('PATH', '')}"


def _import_qwen_tts_model():
    """Import Qwen TTS without printing its optional flash-attn fallback banner."""
    _prepare_optional_audio_runtime()
    try:
        with redirect_stdout(StringIO()):
            from qwen_tts import Qwen3TTSModel
    except TypeError as exc:
        if "check_model_inputs" not in str(exc):
            raise
        raise SpeechPipelineError(
            "Qwen3-TTS is incompatible with the installed `transformers` version.",
            {
                "troubleshooting": (
                    "Use a dedicated Qwen3-TTS environment with transformers==4.57.3. "
                    "The qwen-asr package requires transformers==4.57.6, so the two "
                    "official packages cannot share one environment."
                )
            },
        ) from exc
    return Qwen3TTSModel


@dataclass
class SpeechSignal:
    """Speech payload model passed between simulated TTS and ASR components.
    """
    speaker: str
    text: str
    audio: object = None
    diagnostics: dict | None = None


@dataclass(frozen=True)
class SpeechPipelineConfig:
    """Runtime configuration for the mandatory speech stages."""
    pattern_key: str = DEFAULT_SPEECH_PATTERN
    tts_engine: str = SPEECH_TTS_ENGINE or platform_default_tts_engine()
    asr_engine: str = SPEECH_ASR_ENGINE or platform_default_asr_engine()
    audio_dir: str = SPEECH_AUDIO_DIR
    agent_a_audio_persona: str = DEFAULT_AGENT_A_AUDIO_PERSONA
    agent_b_audio_persona: str = DEFAULT_AGENT_B_AUDIO_PERSONA
    agent_a_custom_audio: bool = True
    agent_b_custom_audio: bool = True
    agent_a_words_per_minute: int = 140
    agent_b_words_per_minute: int = 145
    agent_a_voice: str = ""
    agent_b_voice: str = ""
    agent_a_speech_rate: int = -3
    agent_b_speech_rate: int = -3
    agent_a_volume: int = 100
    agent_b_volume: int = 100
    agent_a_pitch_semitones: int = 0
    agent_b_pitch_semitones: int = 0
    agent_a_pause_ms: int = 260
    agent_b_pause_ms: int = 280
    agent_a_emphasis: str = "none"
    agent_b_emphasis: str = "none"
    agent_a_language: str = "EN"
    agent_b_language: str = "EN"
    agent_a_speed: float = 1.0
    agent_b_speed: float = 1.0
    agent_a_temperature: float = 0.15
    agent_b_temperature: float = 0.12
    agent_a_top_p: float = 0.55
    agent_b_top_p: float = 0.5
    agent_a_top_k: int = 12
    agent_b_top_k: int = 10
    agent_a_seed: int = 11
    agent_b_seed: int = 29
    agent_a_oral_level: int = 2
    agent_b_oral_level: int = 2
    agent_a_laugh_level: int = 0
    agent_b_laugh_level: int = 0
    agent_a_break_level: int = 4
    agent_b_break_level: int = 4
    agent_a_reference_audio: str = ""
    agent_b_reference_audio: str = ""
    agent_a_reference_text: str = ""
    agent_b_reference_text: str = ""
    tts_device: str = "auto"
    tts_model: str = ""
    tts_executable: str = ""
    tts_python_executable: str = ""
    tts_timeout_sec: float = 60.0
    asr_language: str = "en-US"
    asr_model: str = "small.en"
    asr_device: str = "auto"
    asr_compute_type: str = "default"
    asr_executable: str = ""
    asr_python_executable: str = ""
    asr_vad_model: str = ""
    asr_timeout_sec: float = 60.0
    asr_beam_size: int = 8
    asr_initial_silence_sec: float = 4.0
    asr_babble_timeout_sec: float = 6.0
    asr_end_silence_ms: int = 2500
    asr_ambiguous_end_silence_ms: int = 4500
    asr_domain_normalization_enabled: bool = True
    asr_domain_similarity_threshold: float = 0.86
    channel_noise_snr_db: float | None = None
    channel_gain_db: float = 0.0
    channel_clip_threshold: float = 1.0
    channel_dropout_rate: float = 0.0
    min_utterance_sec: float = 0.25
    max_utterance_sec: float = 20.0
    provider_environment_dir: str = ".speech-providers"
    playback_enabled: bool = SPEECH_PLAYBACK_ENABLED
    realtime_enabled: bool = SPEECH_REALTIME_ENABLED

    def applies_to(self, speaker: str) -> bool:
        return speaker.lower().replace(" ", "_") in {"agent_a", "agent_b"}

    def prosody_for(self, speaker: str) -> dict:
        """Return normalized synthesis controls for one agent."""
        prefix = "agent_b" if speaker.lower().replace(" ", "_") == "agent_b" else "agent_a"
        fallback = DEFAULT_AGENT_B_AUDIO_PERSONA if prefix == "agent_b" else DEFAULT_AGENT_A_AUDIO_PERSONA
        persona_key = str(getattr(self, f"{prefix}_audio_persona", fallback) or fallback)
        values = synthesis_values(persona_key, fallback)
        if bool(getattr(self, f"{prefix}_custom_audio", False)):
            values.update({
                "voice": getattr(self, f"{prefix}_voice", ""),
                "speech_rate": getattr(self, f"{prefix}_speech_rate", -3),
                "words_per_minute": getattr(self, f"{prefix}_words_per_minute", 140 if prefix == "agent_a" else 145),
                "volume": getattr(self, f"{prefix}_volume", 100),
                "pitch_semitones": getattr(self, f"{prefix}_pitch_semitones", 0),
                "pause_ms": getattr(self, f"{prefix}_pause_ms", 260 if prefix == "agent_a" else 280),
                "emphasis": getattr(self, f"{prefix}_emphasis", "none"),
                "language": getattr(self, f"{prefix}_language", "EN"),
                "speed": getattr(self, f"{prefix}_speed", 1.0),
                "temperature": getattr(self, f"{prefix}_temperature", 0.3),
                "top_p": getattr(self, f"{prefix}_top_p", 0.7),
                "top_k": getattr(self, f"{prefix}_top_k", 20),
                "seed": getattr(self, f"{prefix}_seed", 11),
                "oral_level": getattr(self, f"{prefix}_oral_level", 2),
                "laugh_level": getattr(self, f"{prefix}_laugh_level", 0),
                "break_level": getattr(self, f"{prefix}_break_level", 4),
                "reference_audio": getattr(self, f"{prefix}_reference_audio", ""),
                "reference_text": getattr(self, f"{prefix}_reference_text", ""),
            })
        emphasis = str(values["emphasis"]).lower()
        if emphasis not in {"none", "reduced", "moderate", "strong"}:
            emphasis = "none"
        return {
            "audio_persona": persona_key,
            "custom_audio": bool(getattr(self, f"{prefix}_custom_audio", False)),
            "voice": str(values["voice"] or "").strip(),
            "speech_rate": max(-10, min(10, int(values["speech_rate"]))),
            "words_per_minute": max(1, int(values["words_per_minute"])),
            "volume": max(0, min(100, int(values["volume"]))),
            "pitch_semitones": max(-12, min(12, int(values["pitch_semitones"]))),
            "pause_ms": max(0, min(2000, int(values["pause_ms"]))),
            "emphasis": emphasis,
            "language": str(values["language"] or "EN"),
            "speed": max(0.25, min(4.0, float(values["speed"]))),
            "temperature": max(0.01, min(2.0, float(values["temperature"]))),
            "top_p": max(0.01, min(1.0, float(values["top_p"]))),
            "top_k": max(1, min(1000, int(values["top_k"]))),
            "seed": int(values["seed"]),
            "oral_level": max(0, min(9, int(values["oral_level"]))),
            "laugh_level": max(0, min(2, int(values["laugh_level"]))),
            "break_level": max(0, min(7, int(values["break_level"]))),
            "reference_audio": str(values["reference_audio"] or "").strip(),
            "reference_text": str(values["reference_text"] or "").strip(),
            "clarity_level": str(values["clarity_level"]),
            "hesitation_probability": max(0.0, min(1.0, float(values["hesitation_probability"]))),
            "filler_probability": max(0.0, min(1.0, float(values["filler_probability"]))),
            "stutter_probability": max(0.0, min(1.0, float(values["stutter_probability"]))),
            "clipping_probability": max(0.0, min(1.0, float(values["clipping_probability"]))),
            "station_substitution_probability": max(0.0, min(1.0, float(values["station_substitution_probability"]))),
            "noise_error_probability": max(0.0, min(1.0, float(values["noise_error_probability"]))),
        }

    @property
    def label(self) -> str:
        playback = ":playback" if self.playback_enabled else ""
        realtime = ":realtime" if self.realtime_enabled else ""
        return f"speech:tts={self.tts_engine}:asr={self.asr_engine}:pattern={self.pattern_key}{playback}{realtime}"


@dataclass(frozen=True)
class SpeechPipelineTrace:
    """Trace of one generated utterance through optional speech stages."""
    speaker: str
    generated_text: str
    outgoing_text: str
    incoming_transcript: str
    outgoing_enabled: bool
    incoming_enabled: bool
    tts_engine: str
    asr_engine: str
    pattern_key: str
    simulated_duration_sec: float
    audio: object = None
    mode: str = "speech"
    pipeline_ok: bool = True
    failure_reason: str | None = None
    diagnostics: dict | None = None
    tts_latency_sec: float | None = None
    asr_latency_sec: float | None = None
    pipeline_latency_sec: float | None = None

    @property
    def signal(self):
        return SpeechSignal(speaker=self.speaker, text=self.outgoing_text, audio=self.audio)


class TextToSpeechEngine(Protocol):
    """Protocol for TTS implementations that convert text into a speech signal.
    """
    name: str

    def synthesize(self, speaker: str, text: str) -> SpeechSignal:
        """Synthesize method for this module's MVC responsibility.
        
        Args:
            speaker: Input value used by `synthesize`; see the function signature and caller context for the expected type.
            text: Input value used by `synthesize`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        ...


class SpeechToTextEngine(Protocol):
    """Protocol for ASR implementations that convert speech signals into transcript text.
    """
    name: str

    def transcribe(self, signal: SpeechSignal) -> str:
        """Transcribe method for this module's MVC responsibility.
        
        Args:
            signal: Input value used by `transcribe`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        ...


class WaveFileTextToSpeech:
    """Dependency-free TTS adapter that writes a simple WAV carrier plus transcript sidecar."""

    name = "wavefile-tts"

    def __init__(
        self,
        audio_dir=SPEECH_AUDIO_DIR,
        sample_rate=8000,
        playback_enabled=False,
        realtime_enabled=False,
        agent_a_words_per_minute=140,
        agent_b_words_per_minute=145,
        max_duration_sec=20.0,
        pattern_key="clean",
        config=None,
    ):
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.playback_enabled = playback_enabled
        self.realtime_enabled = realtime_enabled
        self.agent_a_words_per_minute = agent_a_words_per_minute
        self.agent_b_words_per_minute = agent_b_words_per_minute
        self.max_duration_sec = max_duration_sec
        self.pattern_key = pattern_key
        self.config = config or SpeechPipelineConfig(
            agent_a_words_per_minute=agent_a_words_per_minute,
            agent_b_words_per_minute=agent_b_words_per_minute,
        )
        self._counter = 0

    def synthesize(self, speaker: str, text: str) -> SpeechSignal:
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self._counter += 1
        spoken_text = PatternedSpeechToText(
            self.pattern_key,
            seed=self._counter,
        ).transform_text(normalize_text_for_speech(text), include_recognition_errors=False)
        digest = hashlib.sha1(f"{speaker}:{self._counter}:{spoken_text}".encode("utf-8")).hexdigest()[:10]
        stem = f"{self._counter:04d}-{speaker.lower().replace(' ', '-')}-{digest}"
        wav_path = self.audio_dir / f"{stem}.wav"
        transcript_path = self.audio_dir / f"{stem}.txt"
        duration_sec = self._write_wave(wav_path, speaker, spoken_text)
        prosody = self.config.prosody_for(speaker)
        transcript_path.write_text(spoken_text, encoding="utf-8")
        played = (
            self._play_wave(wav_path, realtime=self.realtime_enabled, fallback_duration=duration_sec)
            if self.playback_enabled
            else False
        )
        waited = bool(self.playback_enabled and self.realtime_enabled)
        return SpeechSignal(
            speaker=speaker,
            text=spoken_text,
            audio={
                "engine": "wavefile",
                "path": str(wav_path),
                "transcript_path": str(transcript_path),
                "sample_rate": self.sample_rate,
                "duration_sec": duration_sec,
                "played": played,
                "realtime": self.realtime_enabled,
                "waited": waited,
                "prosody": prosody,
            },
            diagnostics={"synthesis": "wave carrier with transcript sidecar", "prosody": prosody},
        )

    def _write_wave(self, path: Path, speaker: str, text: str):
        words = re.findall(r"[A-Za-z0-9]+", text)
        prosody = self.config.prosody_for(speaker)
        rate = prosody["words_per_minute"]
        settings = speech_pattern_settings(self.pattern_key)
        duration_multiplier = float(settings.get("duration_multiplier", 1.0) or 1.0)
        punctuation_count = len(re.findall(r"[,;:.!?]", text))
        pause_duration = punctuation_count * prosody["pause_ms"] / 1000.0
        duration = min(
            max(len(words) * 60 / max(rate, 1) * max(duration_multiplier, 0.1) + pause_duration, 0.25),
            self.max_duration_sec,
        )
        samples = int(self.sample_rate * duration)
        base_frequency = 185 if speaker.lower().replace(" ", "_") == "agent_a" else 230
        base_frequency *= 2 ** (prosody["pitch_semitones"] / 12)
        emphasis_scale = {"reduced": 0.82, "none": 1.0, "moderate": 1.12, "strong": 1.25}[prosody["emphasis"]]
        amplitude = min(30000, int(7500 * prosody["volume"] / 100 * emphasis_scale))
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(self.sample_rate)
            frames = bytearray()
            for index in range(samples):
                t = index / self.sample_rate
                word_phase = int((index / max(samples, 1)) * max(len(words), 1))
                frequency = base_frequency + (word_phase % 5) * 12
                envelope = min(1.0, index / max(self.sample_rate * 0.04, 1), (samples - index) / max(self.sample_rate * 0.06, 1))
                value = int(amplitude * envelope * math.sin(2 * math.pi * frequency * t))
                frames.extend(struct.pack("<h", value))
            handle.writeframes(frames)
        return round(duration, 3)

    @staticmethod
    def _play_wave(path: Path, realtime=False, fallback_duration=0.0) -> bool:
        try:
            if platform.system() == "Windows":
                import winsound

                flags = winsound.SND_FILENAME
                if not realtime:
                    flags |= winsound.SND_ASYNC
                winsound.PlaySound(str(path), flags)
                return True
            player = next(
                (
                    command
                    for command in (
                        ("paplay", str(path)),
                        ("aplay", "-q", str(path)),
                        ("ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)),
                    )
                    if shutil.which(command[0])
                ),
                None,
            )
            if player:
                if realtime:
                    subprocess.run(player, check=True, timeout=max(float(fallback_duration) + 10.0, 15.0))
                else:
                    subprocess.Popen(
                        player,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        start_new_session=True,
                    )
                return True
        except Exception:
            pass
        if realtime and fallback_duration:
            time.sleep(fallback_duration)
        return False


class WaveFileSpeechToText:
    """ASR adapter for generated speech artifacts using their transcript sidecar."""

    name = "wavefile-asr"

    def transcribe(self, signal: SpeechSignal) -> str:
        audio = signal.audio if isinstance(signal.audio, dict) else {}
        transcript_path = audio.get("transcript_path")
        if transcript_path and Path(transcript_path).exists():
            return Path(transcript_path).read_text(encoding="utf-8")
        return signal.text


def _write_float_wave(path, samples, sample_rate):
    """Write mono floating-point samples as a 16-bit PCM WAV."""
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise SpeechPipelineError(
            "Text-to-speech produced samples but NumPy is unavailable.",
            {"troubleshooting": "Install numpy in the selected TTS environment."},
        ) from exc
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        raise SpeechPipelineError("Text-to-speech produced an empty audio array.")
    peak = float(np.max(np.abs(audio))) or 1.0
    pcm = np.clip(audio / max(peak, 1.0), -1.0, 1.0)
    pcm = (pcm * 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm.tobytes())


def _wave_duration(path):
    with wave.open(str(path), "rb") as handle:
        return round(handle.getnframes() / float(handle.getframerate()), 3)


class OptionalTextToSpeechBase:
    """Common artifact, playback, and failure handling for optional TTS engines."""

    engine_key = "optional"
    name = "optional-tts"

    def __init__(self, config):
        self.config = config
        self.audio_dir = Path(config.audio_dir)
        self._counter = 0

    def synthesize(self, speaker: str, text: str) -> SpeechSignal:
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self._counter += 1
        spoken_text = PatternedSpeechToText(
            self.config.pattern_key,
            seed=self._counter,
        ).transform_text(normalize_text_for_speech(text), include_recognition_errors=False)
        digest = hashlib.sha1(
            f"{self.engine_key}:{speaker}:{self._counter}:{spoken_text}".encode("utf-8")
        ).hexdigest()[:10]
        stem = f"{self._counter:04d}-{speaker.lower().replace(' ', '-')}-{digest}"
        wav_path = self.audio_dir / f"{stem}.wav"
        transcript_path = self.audio_dir / f"{stem}.txt"
        transcript_path.write_text(spoken_text, encoding="utf-8")
        prosody = self.config.prosody_for(speaker)
        backend_diagnostics = self._synthesize_wave(wav_path, spoken_text, prosody)
        if not wav_path.exists() or wav_path.stat().st_size <= 44:
            raise SpeechPipelineError(
                f"{self.name} failed to produce usable WAV audio.",
                {
                    "engine": self.name,
                    "path": str(wav_path),
                    "backend_diagnostics": dict(backend_diagnostics or {}),
                    "troubleshooting": self.install_hint(),
                },
            )
        duration_sec = _wave_duration(wav_path)
        played = (
            WaveFileTextToSpeech._play_wave(
                wav_path,
                realtime=self.config.realtime_enabled,
                fallback_duration=duration_sec,
            )
            if self.config.playback_enabled
            else False
        )
        if self.config.playback_enabled and not played:
            raise SpeechPipelineError(
                "Text-to-speech audio was generated but playback failed.",
                {"engine": self.name, "path": str(wav_path)},
            )
        audio = {
            "engine": self.engine_key,
            "path": str(wav_path),
            "transcript_path": str(transcript_path),
            "duration_sec": duration_sec,
            "played": played,
            "realtime": self.config.realtime_enabled,
            "waited": bool(self.config.playback_enabled and self.config.realtime_enabled),
            "prosody": prosody,
        }
        return SpeechSignal(
            speaker=speaker,
            text=spoken_text,
            audio=audio,
            diagnostics={
                "synthesis": self.engine_key,
                "prosody": prosody,
                **dict(backend_diagnostics or {}),
            },
        )

    def _synthesize_wave(self, wav_path, text, prosody):
        raise NotImplementedError

    def install_hint(self):
        return "Install and configure the selected optional TTS backend."


class ChatTTSTextToSpeech(OptionalTextToSpeechBase):
    """Lazy ChatTTS adapter using the official Python API."""

    engine_key = "chattts"
    name = "chattts-tts"

    def __init__(self, config):
        super().__init__(config)
        self._chat = None
        self._speaker_embeddings = {}

    def _model(self):
        if self._chat is not None:
            return self._chat
        ChatTTS = self._import_chattts()
        chat = ChatTTS.Chat()
        model_path = Path(str(self.config.tts_model or "")).expanduser()
        loaded = False
        if self.config.tts_model and model_path.exists():
            candidates = [model_path]
            snapshots = model_path / "models--2Noise--ChatTTS" / "snapshots"
            if snapshots.is_dir():
                candidates = sorted(
                    (path for path in snapshots.iterdir() if path.is_dir()),
                    key=lambda path: path.stat().st_mtime,
                    reverse=True,
                ) + candidates
            for candidate in candidates:
                loaded = chat.load(source="custom", custom_path=str(candidate), compile=False)
                if loaded:
                    break
        if not loaded and not self.config.tts_model:
            raise SpeechPipelineError("ChatTTS requires a configured asset directory.")
        if loaded is False:
            raise SpeechPipelineError(
                "ChatTTS model loading failed.",
                {"model_path": str(model_path), "troubleshooting": self.install_hint()},
            )
        self._chat = chat
        return chat

    def _synthesize_wave(self, wav_path, text, prosody):
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise SpeechPipelineError(
                "PyTorch is required by ChatTTS but is not installed.",
                {
                    "missing_module": exc.name,
                    "python_version": platform.python_version(),
                    "troubleshooting": self.install_hint(),
                },
            ) from exc
        ChatTTS = self._import_chattts()
        chat = self._model()
        speaker, speaker_strategy = self._speaker_embedding(chat, torch, prosody["seed"])
        infer = ChatTTS.Chat.InferCodeParams(
            spk_emb=speaker,
            temperature=prosody["temperature"],
            top_P=prosody["top_p"],
            top_K=prosody["top_k"],
        )
        refine = ChatTTS.Chat.RefineTextParams(
            prompt=(
                f"[oral_{prosody['oral_level']}]"
                f"[laugh_{prosody['laugh_level']}]"
                f"[break_{prosody['break_level']}]"
            )
        )
        wavs = chat.infer([text], params_refine_text=refine, params_infer_code=infer)
        _write_float_wave(wav_path, wavs[0], 24000)
        return {
            "sample_rate": 24000,
            "speaker_seed": prosody["seed"],
            "speaker_strategy": speaker_strategy,
            "speaker_cached": True,
        }

    def _speaker_embedding(self, chat, torch, seed):
        """Return a cached speaker tensor without ChatTTS' memory-heavy LZMA encoding."""
        seed = int(seed)
        if seed in self._speaker_embeddings:
            return self._speaker_embeddings[seed], "cached_tensor"

        torch.manual_seed(seed)
        speaker_model = getattr(chat, "speaker", None)
        sample_tensor = getattr(speaker_model, "_sample_random", None)
        if callable(sample_tensor):
            speaker = sample_tensor()
            if hasattr(speaker, "detach"):
                speaker = speaker.detach().to(device="cpu")
            self._speaker_embeddings[seed] = speaker
            return speaker, "uncompressed_tensor"

        try:
            speaker = chat.sample_random_speaker()
        except MemoryError as exc:
            raise SpeechPipelineError(
                "ChatTTS could not allocate memory while creating a speaker embedding.",
                {
                    "speaker_seed": seed,
                    "troubleshooting": (
                        "Use a ChatTTS release exposing speaker._sample_random, reduce other "
                        "loaded models, or select a lower-memory TTS engine."
                    ),
                },
            ) from exc
        self._speaker_embeddings[seed] = speaker
        return speaker, "encoded_fallback"

    def install_hint(self):
        return (
            "Install ChatTTS in a supported project Python environment and place its model "
            "assets in the configured directory, or enable the ChatTTS-only missing-asset download."
        )

    def _import_chattts(self):
        try:
            self._ensure_base16384()
            import ChatTTS
        except ModuleNotFoundError as exc:
            if exc.name == "ChatTTS":
                message = "ChatTTS is selected but not installed."
            else:
                message = f"ChatTTS cannot import required module '{exc.name}'."
            raise SpeechPipelineError(
                message,
                {
                    "missing_module": exc.name,
                    "python_executable": sys.executable,
                    "python_version": platform.python_version(),
                    "troubleshooting": self.install_hint(),
                },
            ) from exc
        return ChatTTS

    @staticmethod
    def _ensure_base16384():
        """Prefer the native codec and supply the ChatTTS subset if its wheel is incomplete."""
        try:
            import pybase16384  # noqa: F401
        except ModuleNotFoundError as exc:
            if not str(exc.name or "").startswith("pybase16384"):
                raise
            from coop_navigation_sds.TextToSpeech import base16384_compat

            sys.modules.pop("pybase16384", None)
            sys.modules["pybase16384"] = base16384_compat


class MeloTextToSpeech(OptionalTextToSpeechBase):
    """MeloTTS adapter using its documented command-line interface."""

    engine_key = "melotts"
    name = "melotts-tts"

    def _synthesize_wave(self, wav_path, text, prosody):
        executable = shutil.which("melo") or shutil.which("melotts")
        if not executable:
            raise SpeechPipelineError(
                "MeloTTS is selected but its `melo` command is unavailable.",
                {"troubleshooting": self.install_hint()},
            )
        command = [
            executable,
            text,
            str(wav_path),
            "--language",
            prosody["language"],
            "--speed",
            str(prosody["speed"]),
        ]
        if prosody["voice"]:
            command.extend(["--speaker", prosody["voice"]])
        completed = subprocess.run(command, text=True, capture_output=True, timeout=self.config.tts_timeout_sec)
        if completed.returncode != 0:
            raise SpeechPipelineError(
                "MeloTTS synthesis failed.",
                {
                    "return_code": completed.returncode,
                    "stderr": completed.stderr.strip(),
                    "troubleshooting": self.install_hint(),
                },
            )
        return {"command": command, "stdout": completed.stdout.strip()}

    def install_hint(self):
        return "Install MeloTTS in a compatible environment and ensure the `melo` command is on PATH."


class PiperTextToSpeech(OptionalTextToSpeechBase):
    """Piper adapter using the official Python API and an explicit ONNX voice."""

    engine_key = "piper"
    name = "piper-tts"

    def __init__(self, config):
        super().__init__(config)
        self._voice = None

    def _model(self):
        if self._voice is not None:
            return self._voice
        model_path = Path(str(self.config.tts_model or "").strip())
        if not model_path.is_file():
            raise SpeechPipelineError(
                "Piper requires a readable ONNX voice in `tts_model`.",
                {"tts_model": str(model_path), "troubleshooting": self.install_hint()},
            )
        try:
            from piper import PiperVoice
        except ModuleNotFoundError as exc:
            raise SpeechPipelineError(
                "Piper is selected but `piper-tts` is not installed.",
                {"troubleshooting": self.install_hint()},
            ) from exc
        use_cuda = str(self.config.tts_device).lower().startswith(("cuda", "gpu"))
        self._voice = PiperVoice.load(str(model_path), use_cuda=use_cuda)
        return self._voice

    def _synthesize_wave(self, wav_path, text, prosody):
        voice = self._model()
        try:
            from piper import SynthesisConfig
        except ModuleNotFoundError as exc:
            raise SpeechPipelineError(
                "Piper synthesis dependencies are unavailable.",
                {"troubleshooting": self.install_hint()},
            ) from exc
        synthesis_config = SynthesisConfig(
            volume=prosody["volume"] / 100.0,
            length_scale=1.0 / max(prosody["speed"], 0.1),
            noise_scale=max(0.0, min(1.0, prosody["temperature"])),
            noise_w_scale=max(0.0, min(1.0, prosody["top_p"])),
            normalize_audio=True,
        )
        with wave.open(str(wav_path), "wb") as wav_file:
            voice.synthesize_wav(text, wav_file, syn_config=synthesis_config)
        return {"model": self.config.tts_model, "device": self.config.tts_device}

    def install_hint(self):
        return (
            "Install `piper-tts`, download a Piper voice with "
            "`python -m piper.download_voices`, and set `tts_model` to its ONNX file."
        )


class EspeakNgTextToSpeech(OptionalTextToSpeechBase):
    """Cross-platform eSpeak NG command-line synthesis adapter."""

    engine_key = "espeak_ng"
    name = "espeak-ng-tts"

    def _synthesize_wave(self, wav_path, text, prosody):
        executable = (
            resolve_espeak_executable(self.config.tts_executable)
        )
        if not executable:
            raise SpeechPipelineError(
                "eSpeak NG is selected but no executable was found.",
                {"troubleshooting": self.install_hint()},
            )
        command = [
            executable,
            "-w", str(wav_path),
            "-s", str(prosody["words_per_minute"]),
            "-a", str(max(0, min(200, prosody["volume"] * 2))),
            "-p", str(max(0, min(99, 50 + prosody["pitch_semitones"] * 3))),
        ]
        voice = prosody["voice"] or str(prosody["language"] or "en").lower()
        if voice:
            command.extend(["-v", voice])
        command.append(text)
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=self.config.tts_timeout_sec,
        )
        if completed.returncode != 0:
            raise SpeechPipelineError(
                "eSpeak NG synthesis failed.",
                {"return_code": completed.returncode, "stderr": completed.stderr.strip(), "command": command},
            )
        return {"command": command, "executable": executable, "voice": voice}

    def install_hint(self):
        return "Install `espeak-ng` with the operating-system package manager or configure `tts_executable`."


class CoquiTextToSpeech(OptionalTextToSpeechBase):
    """Coqui TTS adapter with lazy model loading."""

    engine_key = "coqui"
    name = "coqui-tts"

    def __init__(self, config):
        super().__init__(config)
        self._tts = None

    def _model(self):
        if self._tts is not None:
            return self._tts
        try:
            from TTS.api import TTS
        except ModuleNotFoundError as exc:
            raise SpeechPipelineError(
                "Coqui TTS is selected but the `TTS` package is unavailable.",
                {"troubleshooting": self.install_hint()},
            ) from exc
        model = self.config.tts_model or "tts_models/en/ljspeech/tacotron2-DDC"
        model_root = Path(model)
        if not model_root.exists():
            raise SpeechPipelineError("Coqui model assets are not available locally.")
        os.environ.setdefault(
            "TTS_HOME",
            str(Path(self.config.provider_environment_dir) / "models" / "coqui"),
        )
        use_gpu = str(self.config.tts_device).lower().startswith(("cuda", "gpu"))
        config_path = next(model_root.rglob("config.json"), None)
        checkpoint_path = next(
            (
                path
                for pattern in ("model_file.pth", "best_model.pth", "*.pth")
                for path in model_root.rglob(pattern)
            ),
            None,
        )
        if config_path is None or checkpoint_path is None:
            raise SpeechPipelineError(
                "Prepared Coqui assets require config.json and a model checkpoint."
            )
        self._tts = TTS(
            model_path=str(checkpoint_path),
            config_path=str(config_path),
            progress_bar=False,
            gpu=use_gpu,
        )
        return self._tts

    def _synthesize_wave(self, wav_path, text, prosody):
        model = self._model()
        kwargs = {"text": text, "file_path": str(wav_path)}
        speakers = getattr(model, "speakers", None) or []
        if speakers:
            kwargs["speaker"] = prosody["voice"] if prosody["voice"] in speakers else speakers[0]
        model.tts_to_file(**kwargs)
        return {"model": self.config.tts_model or "tts_models/en/ljspeech/tacotron2-DDC", "speaker": kwargs.get("speaker")}

    def install_hint(self):
        return "Install Coqui `TTS` in a compatible isolated provider environment and configure `tts_model`."


class KokoroTextToSpeech(OptionalTextToSpeechBase):
    """Kokoro adapter using its reusable Python pipeline."""

    engine_key = "kokoro"
    name = "kokoro-tts"

    def __init__(self, config):
        super().__init__(config)
        self._pipelines = {}

    @staticmethod
    def _language_code(language):
        normalized = str(language or "EN").lower().replace("_", "-")
        if normalized.startswith("en-gb"):
            return "b"
        return {
            "en": "a",
            "en-us": "a",
            "ja": "j",
            "zh": "z",
            "es": "e",
            "fr": "f",
            "hi": "h",
            "it": "i",
            "pt": "p",
        }.get(normalized.split("-")[0], "a")

    def _pipeline(self, language):
        language_code = self._language_code(language)
        if language_code in self._pipelines:
            return self._pipelines[language_code]
        try:
            from kokoro import KPipeline
        except ModuleNotFoundError as exc:
            raise SpeechPipelineError(
                "Kokoro is selected but its Python package is not installed.",
                {"troubleshooting": self.install_hint()},
            ) from exc
        self._pipelines[language_code] = KPipeline(lang_code=language_code)
        return self._pipelines[language_code]

    def _synthesize_wave(self, wav_path, text, prosody):
        try:
            import numpy as np
        except ModuleNotFoundError as exc:
            raise SpeechPipelineError(
                "Kokoro requires NumPy to combine generated audio chunks.",
                {"troubleshooting": self.install_hint()},
            ) from exc
        default_voice = (
            "af_heart"
            if "caller" in prosody["audio_persona"]
            else "am_michael"
        )
        voice = prosody["voice"] or default_voice
        chunks = [
            audio
            for _graphemes, _phonemes, audio in self._pipeline(prosody["language"])(
                text,
                voice=voice,
                speed=prosody["speed"],
            )
            if audio is not None
        ]
        if not chunks:
            raise SpeechPipelineError("Kokoro produced no audio chunks.")
        _write_float_wave(wav_path, np.concatenate(chunks), 24000)
        return {"voice": voice, "language": prosody["language"], "sample_rate": 24000}

    def install_hint(self):
        return "Install `kokoro`, `soundfile`, and the language's required phonemizer such as `espeak-ng`."


class F5TextToSpeech(OptionalTextToSpeechBase):
    """F5-TTS adapter using the official inference command."""

    engine_key = "f5_tts"
    name = "f5-tts"

    def _synthesize_wave(self, wav_path, text, prosody):
        if not prosody["reference_audio"]:
            raise SpeechPipelineError(
                "F5-TTS requires reference audio for each speaking agent.",
                {"reference_audio": prosody["reference_audio"], "troubleshooting": self.install_hint()},
            )
        executable = (
            str(self.config.tts_executable or "").strip()
            or shutil.which("f5-tts_infer-cli")
        )
        if not executable:
            raise SpeechPipelineError(
                "F5-TTS is selected but `f5-tts_infer-cli` is unavailable.",
                {"troubleshooting": self.install_hint()},
            )
        command = [
            executable,
            "--model",
            self.config.tts_model or "F5TTS_v1_Base",
            "--ref_audio",
            prosody["reference_audio"],
            "--ref_text",
            prosody["reference_text"],
            "--gen_text",
            text,
            "--output_dir",
            str(wav_path.parent),
            "--output_file",
            wav_path.name,
            "--speed",
            str(prosody["speed"]),
        ]
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=self.config.tts_timeout_sec,
        )
        if completed.returncode != 0:
            raise SpeechPipelineError(
                "F5-TTS synthesis failed.",
                {
                    "return_code": completed.returncode,
                    "stderr": completed.stderr.strip(),
                    "troubleshooting": self.install_hint(),
                },
            )
        return {"command": command, "stdout": completed.stdout.strip()}

    def install_hint(self):
        return "Install F5-TTS, expose `f5-tts_infer-cli`, and configure reference audio and transcript."


class Qwen3TextToSpeech(OptionalTextToSpeechBase):
    """Qwen3-TTS adapter for custom speakers or reference-voice cloning."""

    engine_key = "qwen3_tts"
    name = "qwen3-tts"

    def __init__(self, config):
        super().__init__(config)
        self._model_instances = {}
        self._worker = None

    def _isolated_runtime(self):
        if os.environ.get("MINILLAMA_SPEECH_PROVIDER_WORKER") == "1":
            return None, None
        if self.config.tts_python_executable:
            python = Path(self.config.tts_python_executable)
            worker = Path(__file__).resolve().parents[2] / "scripts" / "qwen_tts_worker.py"
            return (python, worker) if python.is_file() and worker.is_file() else (None, None)
        root = Path(__file__).resolve().parents[2]
        python = root / ".venv-qwen-tts" / "Scripts" / "python.exe"
        worker = root / "scripts" / "qwen_tts_worker.py"
        return (python, worker) if python.exists() and worker.exists() else (None, None)

    def _worker_request(self, payload):
        python, worker = self._isolated_runtime()
        if python is None:
            return None
        if self._worker is None or self._worker.poll() is not None:
            environment = dict(os.environ)
            _prepare_optional_audio_runtime()
            environment["PATH"] = os.environ.get("PATH", "")
            self._worker = subprocess.Popen(
                [str(python), str(worker)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                bufsize=1,
                env=environment,
            )
        self._worker.stdin.write(json.dumps(payload) + "\n")
        self._worker.stdin.flush()
        response_line = self._worker.stdout.readline()
        if not response_line:
            raise SpeechPipelineError(
                "The isolated Qwen3-TTS worker stopped unexpectedly.",
                {"troubleshooting": self.install_hint()},
            )
        response = json.loads(response_line)
        if not response.get("ok"):
            raise SpeechPipelineError(
                f"Qwen3-TTS worker failed: {response.get('error', 'unknown error')}",
                {
                    "worker_traceback": response.get("traceback", ""),
                    "troubleshooting": self.install_hint(),
                },
            )
        return response

    def _model(self, model_name):
        if model_name in self._model_instances:
            return self._model_instances[model_name]
        try:
            import torch
            Qwen3TTSModel = _import_qwen_tts_model()
        except ModuleNotFoundError as exc:
            raise SpeechPipelineError(
                "Qwen3-TTS is selected but `qwen-tts` is not installed.",
                {"troubleshooting": self.install_hint()},
            ) from exc
        device = str(self.config.tts_device or "auto").lower()
        use_cuda = device.startswith("cuda") or (device == "auto" and torch.cuda.is_available())
        self._model_instances[model_name] = Qwen3TTSModel.from_pretrained(
            model_name,
            device_map=device if device != "auto" else ("cuda:0" if use_cuda else "cpu"),
            dtype=torch.bfloat16 if use_cuda else torch.float32,
            attn_implementation="eager",
        )
        return self._model_instances[model_name]

    @staticmethod
    def _language(language):
        normalized = str(language or "EN").lower().split("-")[0]
        return {
            "en": "English",
            "de": "German",
            "fr": "French",
            "es": "Spanish",
            "it": "Italian",
            "pt": "Portuguese",
            "ru": "Russian",
            "ja": "Japanese",
            "ko": "Korean",
            "zh": "Chinese",
        }.get(normalized, "Auto")

    def _synthesize_wave(self, wav_path, text, prosody):
        language = self._language(prosody["language"])
        if prosody["reference_audio"]:
            model_name = self.config.tts_model or "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
            mode = "voice_clone"
            speaker = None
        else:
            model_name = (
                self.config.tts_model
                or "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
            )
            default_speaker = "Serena" if "caller" in prosody["audio_persona"] else "Ryan"
            speaker = prosody["voice"] or default_speaker
            emphasis = "" if prosody["emphasis"] == "none" else f"Use {prosody['emphasis']} emphasis."
            mode = "custom_voice"
        response = self._worker_request({
            "command": "generate",
            "model": model_name,
            "device": str(self.config.tts_device or "auto").lower(),
            "mode": mode,
            "output_path": str(wav_path.resolve()),
            "text": text,
            "language": language,
            "speaker": speaker,
            "instruct": emphasis if mode == "custom_voice" else "",
            "temperature": prosody["temperature"],
            "top_p": prosody["top_p"],
            "top_k": prosody["top_k"],
            "reference_audio": prosody["reference_audio"],
            "reference_text": prosody["reference_text"],
        })
        if response is not None:
            sample_rate = response["sample_rate"]
        else:
            model = self._model(model_name)
            if mode == "voice_clone":
                wavs, sample_rate = model.generate_voice_clone(
                    text=text,
                    language=language,
                    ref_audio=prosody["reference_audio"],
                    ref_text=prosody["reference_text"] or None,
                )
            else:
                wavs, sample_rate = model.generate_custom_voice(
                    text=text,
                    language=language,
                    speaker=speaker,
                    instruct=emphasis,
                    temperature=prosody["temperature"],
                    top_p=prosody["top_p"],
                    top_k=prosody["top_k"],
                )
            _write_float_wave(wav_path, wavs[0], sample_rate)
        return {
            "model": model_name,
            "mode": mode,
            "speaker": speaker,
            "language": language,
            "sample_rate": sample_rate,
        }

    def install_hint(self):
        return (
            "Create `.venv-qwen-tts` with Python 3.11 and install `qwen-tts==0.1.1`. "
            "Use a CustomVoice checkpoint or provide reference audio for a Base checkpoint."
        )


class WindowsSapiTextToSpeech:
    """Windows SAPI TTS stage that writes actual spoken WAV audio."""

    name = "windows-sapi-tts"

    def __init__(
        self,
        audio_dir=SPEECH_AUDIO_DIR,
        playback_enabled=False,
        realtime_enabled=False,
        voice_rate=-3,
        pattern_key="clean",
        config=None,
    ):
        self.audio_dir = Path(audio_dir)
        self.playback_enabled = playback_enabled
        self.realtime_enabled = realtime_enabled
        self.voice_rate = max(-10, min(10, int(voice_rate)))
        self.pattern_key = pattern_key
        self.config = config or SpeechPipelineConfig(
            agent_a_speech_rate=voice_rate,
            agent_b_speech_rate=voice_rate,
        )
        self._counter = 0

    def synthesize(self, speaker: str, text: str) -> SpeechSignal:
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self._counter += 1
        spoken_text = PatternedSpeechToText(
            self.pattern_key,
            seed=self._counter,
        ).transform_text(normalize_text_for_speech(text), include_recognition_errors=False)
        digest = hashlib.sha1(f"sapi:{speaker}:{self._counter}:{spoken_text}".encode("utf-8")).hexdigest()[:10]
        stem = f"{self._counter:04d}-{speaker.lower().replace(' ', '-')}-{digest}"
        wav_path = self.audio_dir / f"{stem}.wav"
        transcript_path = self.audio_dir / f"{stem}.txt"
        transcript_path.write_text(spoken_text, encoding="utf-8")
        prosody = self.config.prosody_for(speaker)
        command = self._powershell_command(wav_path, **synthesis_controls(prosody))
        completed = subprocess.run(
            command,
            input=spoken_text,
            text=True,
            capture_output=True,
            timeout=30,
        )
        if completed.returncode != 0 or not wav_path.exists() or wav_path.stat().st_size <= 44:
            raise SpeechPipelineError(
                "Text-to-speech failed; no usable spoken audio was generated.",
                {
                    "engine": self.name,
                    "return_code": completed.returncode,
                    "stderr": completed.stderr.strip(),
                    "path": str(wav_path),
                    "troubleshooting": "Install or enable Windows speech synthesis voices.",
                },
            )
        duration_sec = self._wave_duration(wav_path)
        played = (
            WaveFileTextToSpeech._play_wave(wav_path, realtime=self.realtime_enabled, fallback_duration=duration_sec)
            if self.playback_enabled
            else False
        )
        if self.playback_enabled and not played:
            raise SpeechPipelineError(
                "Text-to-speech audio was generated but playback failed.",
                {
                    "engine": self.name,
                    "path": str(wav_path),
                    "troubleshooting": "Check the output device, Windows audio service, and application audio permissions.",
                },
            )
        return SpeechSignal(
            speaker=speaker,
            text=spoken_text,
            audio={
                "engine": "windows_sapi",
                "path": str(wav_path),
                "transcript_path": str(transcript_path),
                "duration_sec": duration_sec,
                "played": played,
                "realtime": self.realtime_enabled,
                "waited": bool(self.playback_enabled and self.realtime_enabled),
                "prosody": prosody,
            },
            diagnostics={"synthesis": "windows_sapi", "prosody": prosody},
        )

    @staticmethod
    def _powershell_command(
        wav_path: Path,
        voice="",
        speech_rate=-3,
        words_per_minute=140,
        volume=100,
        pitch_semitones=0,
        pause_ms=260,
        emphasis="none",
    ):
        powershell = shutil.which("powershell") or shutil.which("powershell.exe") or "powershell"
        escaped_path = str(wav_path).replace("'", "''")
        escaped_voice = str(voice or "").replace("'", "''")
        safe_rate = max(-10, min(10, int(speech_rate)))
        safe_volume = max(0, min(100, int(volume)))
        safe_pitch = max(-12, min(12, int(pitch_semitones)))
        safe_pause = max(0, min(2000, int(pause_ms)))
        safe_emphasis = emphasis if emphasis in {"reduced", "moderate", "strong"} else ""
        emphasis_open = f"<emphasis level='{safe_emphasis}'>" if safe_emphasis else ""
        emphasis_close = "</emphasis>" if safe_emphasis else ""
        pitch = f"{safe_pitch:+d}st"
        script_parts = [
            "Add-Type -AssemblyName System.Speech; ",
            "$text = [Console]::In.ReadToEnd(); ",
            "$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer; ",
            f"$speaker.Rate = {safe_rate}; ",
            f"$speaker.Volume = {safe_volume}; ",
        ]
        if escaped_voice:
            script_parts.append(f"$speaker.SelectVoice('{escaped_voice}'); ")
        script_parts.append("$escaped = [System.Security.SecurityElement]::Escape($text); ")
        if safe_pause:
            for pattern, factor in PUNCTUATION_PAUSE_PATTERNS:
                duration_ms = max(1, round(safe_pause * factor))
                script_parts.append(
                    f"$escaped = [regex]::Replace($escaped, '{pattern}', "
                    f"'$0<break time=\"{duration_ms}ms\"/>'); "
                )
        script_parts.extend([
            "$ssml = \"<speak version='1.0' xml:lang='en-US'>"
            f"<prosody pitch='{pitch}'>{emphasis_open}\" + $escaped + "
            f"\"{emphasis_close}</prosody></speak>\"; ",
            f"$speaker.SetOutputToWaveFile('{escaped_path}'); ",
            "$speaker.SpeakSsml($ssml); ",
            "$speaker.Dispose();",
        ])
        script = "".join(script_parts)
        return [powershell, "-NoProfile", "-NonInteractive", "-Command", script]

    @staticmethod
    def _wave_duration(path: Path) -> float:
        with wave.open(str(path), "rb") as handle:
            return round(handle.getnframes() / float(handle.getframerate()), 3)


class WindowsSapiSpeechToText:
    """Windows SAPI ASR stage that transcribes a WAV file."""

    name = "windows-sapi-asr"

    def __init__(
        self,
        language="en-US",
        phrase_hints=None,
        initial_silence_sec=4.0,
        babble_timeout_sec=6.0,
        end_silence_ms=2500,
        ambiguous_end_silence_ms=4500,
    ):
        self.language = language or "en-US"
        self.phrase_hints = tuple(phrase_hints or self._default_phrase_hints())
        self.initial_silence_sec = max(0.1, float(initial_silence_sec))
        self.babble_timeout_sec = max(0.1, float(babble_timeout_sec))
        self.end_silence_ms = max(100, int(end_silence_ms))
        self.ambiguous_end_silence_ms = max(
            self.end_silence_ms,
            int(ambiguous_end_silence_ms),
        )

    def transcribe(self, signal: SpeechSignal) -> str:
        audio = signal.audio if isinstance(signal.audio, dict) else {}
        wav_path = audio.get("path")
        if not wav_path or not Path(wav_path).exists():
            raise SpeechPipelineError(
                "Automatic speech recognition failed; no audio file reached the recognizer.",
                {"engine": self.name, "audio": audio},
            )
        command = self._powershell_command(
            Path(wav_path),
            self.language,
            initial_silence_sec=self.initial_silence_sec,
            babble_timeout_sec=self.babble_timeout_sec,
            end_silence_ms=self.end_silence_ms,
            ambiguous_end_silence_ms=self.ambiguous_end_silence_ms,
        )
        completed = subprocess.run(command, text=True, capture_output=True, timeout=30)
        raw_output = completed.stdout.strip()
        transcript, confidence, alternatives = self._parse_result(raw_output)
        primary_transcript = transcript
        selected_transcript = self._best_domain_alternative(transcript, alternatives)
        transcript = selected_transcript
        if completed.returncode != 0:
            raise SpeechPipelineError(
                "Automatic speech recognition failed; no transcript was produced from audio.",
                {
                    "engine": self.name,
                    "return_code": completed.returncode,
                    "stderr": completed.stderr.strip(),
                    "path": str(wav_path),
                    "troubleshooting": "Install a Windows speech recognizer for the language and verify Windows speech services.",
                },
            )
        diagnostics = signal.diagnostics if isinstance(signal.diagnostics, dict) else {}
        signal.diagnostics = diagnostics
        diagnostics["raw_asr_transcript"] = primary_transcript
        diagnostics["asr_selected_transcript"] = transcript
        diagnostics["asr_repair_used"] = transcript.casefold() != str(primary_transcript or "").strip().casefold()
        diagnostics["asr_engine"] = self.name
        diagnostics["asr_confidence"] = confidence
        diagnostics["asr_alternatives"] = alternatives
        diagnostics["asr_language"] = self.language
        if not transcript:
            raise SpeechPipelineError(
                "Automatic speech recognition failed; no transcript was produced from audio.",
                {
                    "engine": self.name,
                    "return_code": completed.returncode,
                    "stderr": completed.stderr.strip(),
                    "path": str(wav_path),
                    "troubleshooting": "Install a Windows speech recognizer for the language and verify Windows speech services.",
                },
            )
        return transcript

    @staticmethod
    def _powershell_command(
        wav_path: Path,
        language="en-US",
        initial_silence_sec=4.0,
        babble_timeout_sec=6.0,
        end_silence_ms=2500,
        ambiguous_end_silence_ms=4500,
    ):
        powershell = shutil.which("powershell") or shutil.which("powershell.exe") or "powershell"
        escaped_path = str(wav_path).replace("'", "''")
        escaped_language = str(language or "en-US").replace("'", "''")
        script = (
            "Add-Type -AssemblyName System.Speech; "
            f"$culture = New-Object System.Globalization.CultureInfo('{escaped_language}'); "
            "$recognizer = New-Object System.Speech.Recognition.SpeechRecognitionEngine($culture); "
            "$recognizer.LoadGrammar((New-Object System.Speech.Recognition.DictationGrammar)); "
            f"$recognizer.InitialSilenceTimeout = [TimeSpan]::FromSeconds({max(0.1, float(initial_silence_sec))}); "
            f"$recognizer.BabbleTimeout = [TimeSpan]::FromSeconds({max(0.1, float(babble_timeout_sec))}); "
            f"$recognizer.EndSilenceTimeout = [TimeSpan]::FromMilliseconds({max(100, int(end_silence_ms))}); "
            f"$recognizer.EndSilenceTimeoutAmbiguous = [TimeSpan]::FromMilliseconds({max(int(end_silence_ms), int(ambiguous_end_silence_ms))}); "
            f"$recognizer.SetInputToWaveFile('{escaped_path}'); "
            "$result = $recognizer.Recognize(); "
            "if ($result -ne $null) { "
            "$alternatives = @($result.Alternates | Select-Object -First 8 | ForEach-Object { "
            "@{ text = $_.Text; confidence = $_.Confidence } }); "
            "@{ text = $result.Text; confidence = $result.Confidence; alternatives = $alternatives } "
            "| ConvertTo-Json -Compress -Depth 4 | Write-Output }; "
            "$recognizer.Dispose();"
        )
        return [powershell, "-NoProfile", "-NonInteractive", "-Command", script]

    @staticmethod
    def _parse_result(raw_output):
        try:
            payload = json.loads(raw_output)
        except (json.JSONDecodeError, TypeError):
            return raw_output, None, []
        alternatives = payload.get("alternatives") or []
        if isinstance(alternatives, dict):
            alternatives = [alternatives]
        return (
            str(payload.get("text") or "").strip(),
            payload.get("confidence"),
            [
                {
                    "text": str(item.get("text") or "").strip(),
                    "confidence": item.get("confidence"),
                }
                for item in alternatives
                if isinstance(item, dict)
            ],
        )

    def _best_domain_alternative(self, transcript, alternatives):
        candidates = [{"text": transcript, "confidence": 0.0}, *alternatives]
        hints = tuple(hint.casefold() for hint in self.phrase_hints)

        def score(candidate):
            text = self._normalize_domain_terms(candidate.get("text", ""))
            folded = text.casefold()
            domain_matches = sum(bool(re.search(rf"\b{re.escape(hint)}\b", folded)) for hint in hints)
            confidence = float(candidate.get("confidence") or 0.0)
            return domain_matches * 2.0 + confidence + min(len(text.split()), 20) * 0.01

        selected = max(candidates, key=score).get("text", "").strip()
        return self._normalize_domain_terms(selected)

    def _normalize_domain_terms(self, transcript):
        """Repair high-confidence spelling variants using only the public domain lexicon."""
        return normalize_transit_transcript(
            transcript,
            threshold=0.86,
            vocabulary=self.phrase_hints,
        )

    @staticmethod
    def _default_phrase_hints():
        try:
            from coop_navigation_sds.TransportNetwork.network import LINES, STATIONS

            return (
                *STATIONS,
                *LINES.keys(),
                "route",
                "routes",
                "direction",
                "directions",
                "transfer",
                "change",
                "minutes",
            )
        except Exception:
            return ("route", "routes", "transfer", "change", "minutes")


class FasterWhisperSpeechToText:
    """Optional local Faster-Whisper ASR adapter."""

    name = "faster-whisper-asr"

    def __init__(
        self,
        model_name="small.en",
        device="auto",
        compute_type="default",
        language="en",
        beam_size=5,
        min_silence_duration_ms=1200,
        model_cache_dir=".speech-providers/models/faster-whisper",
    ):
        self.model_name = resolve_faster_whisper_model(model_name or "small.en")
        self.device = device or "auto"
        self.compute_type = compute_type or "default"
        self.language = (language or "en").split("-")[0].lower()
        self.beam_size = max(1, int(beam_size))
        self.min_silence_duration_ms = max(100, int(min_silence_duration_ms))
        self.model_cache_dir = str(model_cache_dir)
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel
        except ModuleNotFoundError as exc:
            raise SpeechPipelineError(
                "Faster-Whisper is selected but not installed.",
                {"troubleshooting": "Install `faster-whisper` and ensure its model can be loaded."},
            ) from exc
        local_path = Path(str(self.model_name))
        if local_path.exists():
            ready, resolved = faster_whisper_model_ready(local_path)
            if not ready:
                raise SpeechPipelineError(
                    "Faster-Whisper model assets are incomplete.",
                    {
                        "configured_model": str(local_path),
                        "resolved_model": resolved,
                        "required_file": "model.bin",
                        "troubleshooting": (
                            "Configure asr_model to the snapshot directory containing "
                            "model.bin and config.json, or to its prepared cache parent."
                        ),
                    },
                )
            self.model_name = resolved
        self._model = WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type,
            download_root=self.model_cache_dir,
            local_files_only=True,
        )
        return self._model

    def transcribe(self, signal: SpeechSignal) -> str:
        audio = signal.audio if isinstance(signal.audio, dict) else {}
        wav_path = audio.get("path")
        if not wav_path or not Path(wav_path).exists():
            raise SpeechPipelineError(
                "Faster-Whisper received no readable audio artifact.",
                {"engine": self.name, "audio": audio},
            )
        try:
            from coop_navigation_sds.TransportNetwork.network import LINES, STATIONS

            prompt = "Transit hotline. Stations: " + ", ".join(STATIONS) + ". Lines: " + ", ".join(LINES)
        except Exception:
            prompt = "Public transit route with station names, line changes, transfers, and travel minutes."
        segments, info = self._load_model().transcribe(
            str(wav_path),
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": self.min_silence_duration_ms},
            initial_prompt=prompt,
            condition_on_previous_text=False,
        )
        transcript = " ".join(segment.text.strip() for segment in segments).strip()
        diagnostics = signal.diagnostics if isinstance(signal.diagnostics, dict) else {}
        signal.diagnostics = diagnostics
        diagnostics.update({
            "raw_asr_transcript": transcript,
            "asr_engine": self.name,
            "asr_language": getattr(info, "language", self.language),
            "asr_language_probability": getattr(info, "language_probability", None),
            "asr_model": self.model_name,
            "asr_search_width": self.beam_size,
            "asr_search_width_applied": True,
        })
        if not transcript:
            raise SpeechPipelineError(
                "Faster-Whisper produced an empty transcript.",
                {"engine": self.name, "path": str(wav_path)},
            )
        return transcript


def _speech_audio_path(signal, engine_name):
    """Return the readable WAV path carried by a speech signal."""
    audio = signal.audio if isinstance(signal.audio, dict) else {}
    wav_path = Path(str(audio.get("path") or ""))
    if not wav_path.is_file():
        raise SpeechPipelineError(
            f"{engine_name} received no readable audio artifact.",
            {"engine": engine_name, "audio": audio},
        )
    return wav_path


def _record_asr_diagnostics(signal, engine_name, transcript, **details):
    diagnostics = signal.diagnostics if isinstance(signal.diagnostics, dict) else {}
    signal.diagnostics = diagnostics
    diagnostics.update({
        "raw_asr_transcript": transcript,
        "asr_engine": engine_name,
        **details,
    })
    if not str(transcript).strip():
        raise SpeechPipelineError(
            f"{engine_name} produced an empty transcript.",
            {"engine": engine_name},
        )
    return str(transcript).strip()


class VoskSpeechToText:
    """Offline Vosk recognizer for mono PCM WAV artifacts."""

    name = "vosk-asr"

    def __init__(
        self,
        model_name="",
        language="en-US",
        search_width=1,
        model_cache_dir=".speech-providers/models/vosk",
    ):
        self.model_name = str(model_name or "").strip()
        self.language = str(language or "en-US").lower()
        self.search_width = max(1, int(search_width))
        self.model_cache_dir = Path(model_cache_dir)
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            import vosk
            from vosk import Model
        except ModuleNotFoundError as exc:
            raise SpeechPipelineError(
                "Vosk is selected but its Python package is not installed.",
                {"troubleshooting": "Install `vosk` and configure a local model path if required."},
            ) from exc
        if self.model_name and Path(self.model_name).exists():
            model_path = Path(self.model_name)
            self._model = Model(model_path=str(model_path))
        else:
            raise SpeechPipelineError("Vosk model assets are not available locally.")
        return self._model

    def transcribe(self, signal: SpeechSignal) -> str:
        wav_path = _speech_audio_path(signal, self.name)
        try:
            from vosk import KaldiRecognizer
        except ModuleNotFoundError as exc:
            raise SpeechPipelineError(
                "Vosk recognition dependencies are unavailable.",
                {"troubleshooting": "Install `vosk` and a compatible language model."},
            ) from exc
        chunks = []
        with wave.open(str(wav_path), "rb") as wav_file:
            if (
                wav_file.getnchannels() != 1
                or wav_file.getsampwidth() != 2
                or wav_file.getcomptype() != "NONE"
            ):
                raise SpeechPipelineError(
                    "Vosk requires mono 16-bit PCM WAV audio.",
                    {"path": str(wav_path)},
                )
            recognizer = KaldiRecognizer(self._load_model(), wav_file.getframerate())
            recognizer.SetWords(True)
            recognizer.SetMaxAlternatives(self.search_width)
            while data := wav_file.readframes(4000):
                if recognizer.AcceptWaveform(data):
                    chunks.append(self._result_text(json.loads(recognizer.Result())))
            chunks.append(self._result_text(json.loads(recognizer.FinalResult())))
        transcript = " ".join(chunk.strip() for chunk in chunks if chunk.strip())
        return _record_asr_diagnostics(
            signal,
            self.name,
            transcript,
            asr_model=self.model_name or self.language,
            asr_language=self.language,
            asr_search_width=self.search_width,
            asr_search_width_applied=True,
        )

    @staticmethod
    def _result_text(payload):
        """Read Vosk's ordinary or N-best result structure."""
        if payload.get("text"):
            return payload["text"]
        alternatives = payload.get("alternatives") or ()
        return alternatives[0].get("text", "") if alternatives else ""


class SherpaOnnxSpeechToText:
    """Offline sherpa-onnx adapter with model-directory auto-detection."""

    name = "sherpa-onnx-asr"

    def __init__(self, model_name="", language="en-US", search_width=1):
        self.model_name = str(model_name or "").strip()
        self.language = str(language or "en-US")
        self.search_width = max(1, int(search_width))
        self._recognizer = None

    @staticmethod
    def _first(root, patterns):
        for pattern in patterns:
            matches = sorted(root.glob(pattern))
            if matches:
                return matches[0]
        return None

    def _load_model(self):
        if self._recognizer is not None:
            return self._recognizer
        try:
            import sherpa_onnx
        except ModuleNotFoundError as exc:
            raise SpeechPipelineError(
                "sherpa-onnx is selected but its Python package is unavailable.",
                {"troubleshooting": self.install_hint()},
            ) from exc
        root = Path(self.model_name)
        if not root.is_dir():
            raise SpeechPipelineError(
                "sherpa-onnx requires `asr_model` to be a local model directory.",
                {"asr_model": self.model_name, "troubleshooting": self.install_hint()},
            )
        tokens = self._first(root, ("tokens.txt", "*tokens*.txt"))
        encoder = self._first(root, ("*encoder*.onnx",))
        decoder = self._first(root, ("*decoder*.onnx",))
        joiner = self._first(root, ("*joiner*.onnx",))
        paraformer = self._first(root, ("*paraformer*.onnx", "model.onnx"))
        if not tokens:
            raise SpeechPipelineError("sherpa-onnx model directory has no tokens file.")
        common = {"tokens": str(tokens), "num_threads": 2, "debug": False}
        if encoder and decoder and joiner:
            decoding_method = (
                "greedy_search" if self.search_width == 1 else "modified_beam_search"
            )
            self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=str(encoder), decoder=str(decoder), joiner=str(joiner),
                decoding_method=decoding_method,
                max_active_paths=self.search_width,
                **common,
            )
            model_type = "transducer"
            search_width_applied = True
        elif encoder and decoder:
            self._recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
                encoder=str(encoder), decoder=str(decoder), language=self.language.split("-")[0], **common
            )
            model_type = "whisper"
            search_width_applied = False
        elif paraformer:
            self._recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
                paraformer=str(paraformer), **common
            )
            model_type = "paraformer"
            search_width_applied = False
        else:
            raise SpeechPipelineError(
                "Unsupported sherpa-onnx model directory layout.",
                {"asr_model": str(root), "troubleshooting": self.install_hint()},
            )
        self._model_type = model_type
        self._search_width_applied = search_width_applied
        return self._recognizer

    def transcribe(self, signal: SpeechSignal) -> str:
        from array import array

        wav_path = _speech_audio_path(signal, self.name)
        with wave.open(str(wav_path), "rb") as handle:
            if handle.getsampwidth() != 2 or handle.getcomptype() != "NONE":
                raise SpeechPipelineError("sherpa-onnx requires uncompressed 16-bit PCM WAV audio.")
            channels = handle.getnchannels()
            sample_rate = handle.getframerate()
            samples = array("h", handle.readframes(handle.getnframes()))
        if channels > 1:
            samples = array("h", samples[::channels])
        normalized = [sample / 32768.0 for sample in samples]
        recognizer = self._load_model()
        stream = recognizer.create_stream()
        stream.accept_waveform(sample_rate, normalized)
        recognizer.decode_stream(stream)
        transcript = getattr(stream.result, "text", "")
        return _record_asr_diagnostics(
            signal,
            self.name,
            transcript,
            asr_model=self.model_name,
            asr_language=self.language,
            sherpa_model_type=getattr(self, "_model_type", "unknown"),
            asr_search_width=self.search_width,
            asr_search_width_applied=getattr(self, "_search_width_applied", False),
        )

    @staticmethod
    def install_hint():
        return "Install `sherpa-onnx`, download an offline model directory, and set `asr_model` to that directory."


class WhisperCppSpeechToText:
    """Portable whisper.cpp adapter using the official whisper-cli executable."""

    name = "whisper-cpp-asr"
    maximum_beam_size = 8

    def __init__(
        self,
        model_name="",
        executable="",
        language="en",
        vad_model="",
        search_width=1,
        timeout_sec=60.0,
        provider_environment_dir=".speech-providers",
    ):
        self.model_name = str(model_name or "").strip()
        self.executable = str(executable or "").strip()
        self.language = str(language or "en").split("-")[0].lower()
        self.vad_model = str(vad_model or "").strip()
        self.search_width = max(1, int(search_width))
        self.timeout_sec = max(1.0, float(timeout_sec))
        self.provider_environment_dir = str(provider_environment_dir or ".speech-providers")

    def transcribe(self, signal: SpeechSignal) -> str:
        wav_path = _speech_audio_path(signal, self.name)
        resolved = resolve_whisper_cpp_paths(
            executable=self.executable,
            model=self.model_name,
            vad_model=self.vad_model,
            environment_dir=self.provider_environment_dir,
        )
        executable = resolved["executable"]
        model_path = Path(resolved["model"] or self.model_name)
        if not executable or not model_path.is_file():
            raise SpeechPipelineError(
                "whisper.cpp requires `whisper-cli` and a readable GGML model path.",
                {
                    "executable": executable,
                    "asr_model": resolved["model"] or self.model_name,
                    "provider_environment_dir": self.provider_environment_dir,
                    "troubleshooting": (
                        "Set `asr_executable` and `asr_model`, or register them in "
                        ".speech-providers/providers.json under providers.whisper_cpp."
                    ),
                },
            )
        output_base = wav_path.with_name(f"{wav_path.stem}-whispercpp")
        output_file = Path(f"{output_base}.txt")
        effective_search_width = min(self.search_width, self.maximum_beam_size)
        command = [
            executable,
            "-m",
            str(model_path),
            "-f",
            str(wav_path),
            "-l",
            self.language,
            "--output-txt",
            "--output-file",
            str(output_base),
            "--no-timestamps",
            "--no-prints",
            "--beam-size",
            str(effective_search_width),
        ]
        if resolved["vad_model"]:
            command.extend(["--vad", "--vad-model", resolved["vad_model"]])
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=self.timeout_sec,
        )
        if completed.returncode != 0 or not output_file.is_file():
            raise SpeechPipelineError(
                "whisper.cpp transcription failed.",
                {
                    "return_code": completed.returncode,
                    "stderr": completed.stderr.strip(),
                    "output": str(output_file),
                },
            )
        transcript = output_file.read_text(encoding="utf-8").strip()
        output_file.unlink(missing_ok=True)
        return _record_asr_diagnostics(
            signal,
            self.name,
            transcript,
            asr_model=self.model_name,
            asr_language=self.language,
            resolved_executable=executable,
            resolved_model=str(model_path),
            command=command,
            asr_search_width=effective_search_width,
            asr_search_width_requested=self.search_width,
            asr_search_width_saturated=self.search_width != effective_search_width,
            asr_search_width_maximum=self.maximum_beam_size,
            asr_search_width_applied=True,
        )


class ParakeetSpeechToText:
    """NVIDIA NeMo adapter for Parakeet automatic speech recognition."""

    name = "parakeet-asr"

    def __init__(self, model_name="", device="auto"):
        self.model_name = (
            model_name
            if model_name and model_name != "small.en"
            else "nvidia/parakeet-tdt-0.6b-v2"
        )
        self.device = str(device or "auto")
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            import nemo.collections.asr as nemo_asr
        except ModuleNotFoundError as exc:
            raise SpeechPipelineError(
                "Parakeet is selected but NVIDIA NeMo ASR is not installed.",
                {"troubleshooting": "Install `nemo_toolkit[asr]` in a compatible environment."},
            ) from exc
        model_path = Path(self.model_name)
        self._model = (
            nemo_asr.models.ASRModel.restore_from(str(model_path))
            if model_path.is_file()
            else nemo_asr.models.ASRModel.from_pretrained(self.model_name)
        )
        if self.device != "auto" and hasattr(self._model, "to"):
            self._model.to(self.device)
        return self._model

    def transcribe(self, signal: SpeechSignal) -> str:
        wav_path = _speech_audio_path(signal, self.name)
        result = self._load_model().transcribe([str(wav_path)])[0]
        transcript = getattr(result, "text", result)
        return _record_asr_diagnostics(
            signal,
            self.name,
            transcript,
            asr_model=self.model_name,
            asr_language="en",
        )


class Qwen3SpeechToText:
    """Qwen3-ASR adapter using the official Python package."""

    name = "qwen3-asr"

    def __init__(
        self,
        model_name="",
        device="auto",
        language="en-US",
        model_cache_dir=".speech-providers/models/qwen3-asr",
    ):
        self.model_name = (
            model_name
            if model_name and model_name != "small.en"
            else "Qwen/Qwen3-ASR-1.7B"
        )
        self.device = str(device or "auto").lower()
        self.language = str(language or "").split("-")[0].lower()
        self.model_cache_dir = str(model_cache_dir)
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            import torch
            _prepare_optional_audio_runtime()
            from qwen_asr import Qwen3ASRModel
        except ModuleNotFoundError as exc:
            raise SpeechPipelineError(
                "Qwen3-ASR is selected but `qwen-asr` is not installed.",
                {"troubleshooting": "Install `qwen-asr` and use a compatible PyTorch environment."},
            ) from exc
        use_cuda = self.device.startswith("cuda") or (
            self.device == "auto" and torch.cuda.is_available()
        )
        self._model = Qwen3ASRModel.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16 if use_cuda else torch.float32,
            device_map=self.device if self.device != "auto" else ("cuda:0" if use_cuda else "cpu"),
            max_inference_batch_size=1,
            max_new_tokens=256,
            cache_dir=self.model_cache_dir,
            local_files_only=True,
        )
        return self._model

    def transcribe(self, signal: SpeechSignal) -> str:
        wav_path = _speech_audio_path(signal, self.name)
        language = {
            "en": "English",
            "de": "German",
            "fr": "French",
            "es": "Spanish",
            "it": "Italian",
            "pt": "Portuguese",
            "ru": "Russian",
            "ja": "Japanese",
            "ko": "Korean",
            "zh": "Chinese",
        }.get(self.language)
        result = self._load_model().transcribe(
            audio=str(wav_path),
            language=language,
        )[0]
        transcript = getattr(result, "text", result)
        return _record_asr_diagnostics(
            signal,
            self.name,
            transcript,
            asr_model=self.model_name,
            asr_language=getattr(result, "language", language),
        )

def _persona_random(speaker, text, prosody, stage):
    material = f"{speaker}|{prosody.get('audio_persona')}|{prosody.get('seed')}|{stage}|{text}"
    seed = int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:16], 16)
    return random.Random(seed)


def _apply_persona_delivery(text, speaker, prosody):
    """Apply reproducible lexical manifestations of the selected delivery style."""
    rng = _persona_random(speaker, text, prosody, "delivery")
    words = text.split()
    if words and rng.random() < prosody.get("hesitation_probability", 0.0):
        words.insert(min(2, len(words)), "um,")
    if words and rng.random() < prosody.get("filler_probability", 0.0):
        words.insert(0, "Okay,")
    if words and rng.random() < prosody.get("stutter_probability", 0.0):
        index = rng.randrange(len(words))
        clean = re.sub(r"[^A-Za-z]", "", words[index])
        if clean:
            words[index] = f"{clean[0]}-{words[index]}"
    return " ".join(words)


def _apply_persona_recognition_errors(text, speaker, prosody):
    """Apply logged deterministic transcript degradation for controlled personas."""
    rng = _persona_random(speaker, text, prosody, "recognition")
    transformed = text
    substitutions = {"Bravo": "Brava", "Harbor": "Habor", "Sierra": "Sarah", "Golf": "Gulf"}
    if rng.random() < prosody.get("station_substitution_probability", 0.0):
        for source, target in substitutions.items():
            if re.search(rf"\b{source}\b", transformed, re.IGNORECASE):
                transformed = re.sub(rf"\b{source}\b", target, transformed, count=1, flags=re.IGNORECASE)
                break
    words = transformed.split()
    probability = max(
        prosody.get("clipping_probability", 0.0),
        prosody.get("noise_error_probability", 0.0),
    )
    if len(words) > 4 and rng.random() < probability:
        removable = [i for i, word in enumerate(words) if not re.search(r"\d|Alpha|Bravo|Harbor", word, re.IGNORECASE)]
        if removable:
            del words[rng.choice(removable)]
    return " ".join(words)


def _apply_audio_channel_impairment(signal, config, speaker):
    """Apply a deterministic PCM channel treatment between TTS and ASR.

    The clean synthesis path is retained for TTS evaluation. ASR receives the
    derived channel waveform, which makes acoustic degradation observable and
    replayable instead of simulating it only by editing the transcript.
    """
    audio = signal.audio if isinstance(signal.audio, dict) else {}
    source = Path(str(audio.get("path") or ""))
    settings = {
        "noise_snr_db": config.channel_noise_snr_db,
        "gain_db": max(-40.0, min(20.0, float(config.channel_gain_db))),
        "clip_threshold": max(0.05, min(1.0, float(config.channel_clip_threshold))),
        "dropout_rate": max(0.0, min(0.95, float(config.channel_dropout_rate))),
    }
    requested = (
        settings["noise_snr_db"] is not None
        or settings["gain_db"] != 0.0
        or settings["clip_threshold"] < 1.0
        or settings["dropout_rate"] > 0.0
    )
    if config.tts_engine == "file" and config.asr_engine == "file":
        audio["channel_impairment"] = {**settings, "applied": False, "reason": "text_control"}
        return signal
    if not requested:
        audio["channel_impairment"] = {**settings, "applied": False, "reason": "clean_channel"}
        return signal
    if not source.is_file():
        raise SpeechPipelineError(
            "Audio channel treatment requires a readable synthesized WAV file.",
            {"path": str(source), "settings": settings},
        )
    try:
        with wave.open(str(source), "rb") as input_wave:
            parameters = input_wave.getparams()
            frames = input_wave.readframes(parameters.nframes)
    except (OSError, wave.Error) as exc:
        raise SpeechPipelineError(
            "Audio channel treatment could not read the synthesized WAV file.",
            {"path": str(source), "settings": settings, "error": repr(exc)},
        ) from exc
    if parameters.sampwidth != 2 or parameters.comptype != "NONE":
        raise SpeechPipelineError(
            "Audio channel treatment supports uncompressed 16-bit PCM WAV only.",
            {
                "path": str(source),
                "sample_width": parameters.sampwidth,
                "compression": parameters.comptype,
            },
        )
    sample_count = len(frames) // 2
    if not sample_count:
        raise SpeechPipelineError("Audio channel treatment received an empty waveform.")
    samples = list(struct.unpack(f"<{sample_count}h", frames))
    gain = 10.0 ** (settings["gain_db"] / 20.0)
    gained = [sample * gain for sample in samples]
    signal_rms = math.sqrt(sum(sample * sample for sample in gained) / sample_count)
    snr = settings["noise_snr_db"]
    noise_rms = 0.0 if snr is None else signal_rms / (10.0 ** (float(snr) / 20.0))
    seed_material = (
        f"{speaker}|{signal.text}|{config.pattern_key}|{settings}|{source.name}"
    )
    rng = random.Random(int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:16], 16))
    block_samples = max(1, int(parameters.framerate * parameters.nchannels * 0.02))
    clip_limit = int(32767 * settings["clip_threshold"])
    output = []
    clipped = 0
    dropped = 0
    drop_block = False
    for index, sample in enumerate(gained):
        if index % block_samples == 0:
            drop_block = rng.random() < settings["dropout_rate"]
        if drop_block:
            value = 0.0
            dropped += 1
        else:
            value = sample + (rng.gauss(0.0, noise_rms) if noise_rms else 0.0)
        if value > clip_limit:
            value = clip_limit
            clipped += 1
        elif value < -clip_limit:
            value = -clip_limit
            clipped += 1
        output.append(int(round(value)))
    destination = source.with_name(f"{source.stem}-channel.wav")
    with wave.open(str(destination), "wb") as output_wave:
        output_wave.setparams(parameters)
        output_wave.writeframes(struct.pack(f"<{sample_count}h", *output))
    audio["clean_path"] = str(source)
    audio["path"] = str(destination)
    audio["channel_impairment"] = {
        **settings,
        "applied": True,
        "sample_count": sample_count,
        "clipped_sample_rate": round(clipped / sample_count, 6),
        "dropped_sample_rate": round(dropped / sample_count, 6),
        "clean_path": str(source),
        "channel_path": str(destination),
    }
    return signal


class SpeechTransport:
    """Controller utility that composes TTS and ASR engines into one transmit operation.
    """
    def __init__(self, tts_engine=None, asr_engine=None, config=None):
        """  init   method for this module's MVC responsibility.
        
        Args:
            tts_engine: Input value used by `__init__`; see the function signature and caller context for the expected type.
            asr_engine: Input value used by `__init__`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        self.config = config or SpeechPipelineConfig()
        self.tts_engine = tts_engine or self._default_tts_engine()
        self.asr_engine = asr_engine or self._default_asr_engine()
        self.validate_configuration()

    def _default_tts_engine(self):
        engine = self._stage_engine(self.config.tts_engine)
        aliases = {
            "windows": "sapi",
            "windows_sapi": "sapi",
            "speech": "sapi",
            "chat_tts": "chattts",
            "wav": "file",
            "wave": "file",
        }
        engine = aliases.get(engine, engine)
        if engine not in TTS_ENGINE_SPECS:
            raise SpeechPipelineError(
                f"Unsupported text-to-speech engine '{engine}'.",
                {"allowed": available_tts_engine_keys()},
            )
        isolated = self._isolated_stage("tts", engine)
        if isolated is not None:
            return isolated
        if engine in {"sapi", "windows", "windows_sapi", "speech"}:
            if platform.system() != "Windows":
                raise SpeechPipelineError(
                    "Windows SAPI text-to-speech is only available on Windows.",
                    {"troubleshooting": "Select ChatTTS or Piper on Linux."},
                )
            return WindowsSapiTextToSpeech(
                self.config.audio_dir,
                playback_enabled=self.config.playback_enabled,
                realtime_enabled=self.config.realtime_enabled,
                pattern_key=self.config.pattern_key,
                config=self.config,
            )
        if engine in {"chattts", "chat_tts"}:
            return ChatTTSTextToSpeech(self.config)
        if engine in {"melotts", "melo", "melo_tts"}:
            return MeloTextToSpeech(self.config)
        if engine == "piper":
            return PiperTextToSpeech(self.config)
        if engine in {"espeak_ng", "espeak-ng", "espeak"}:
            return EspeakNgTextToSpeech(self.config)
        if engine in {"coqui", "coqui_tts"}:
            return CoquiTextToSpeech(self.config)
        if engine == "kokoro":
            return KokoroTextToSpeech(self.config)
        if engine in {"f5_tts", "f5-tts", "f5"}:
            return F5TextToSpeech(self.config)
        if engine in {"qwen3_tts", "qwen3-tts", "qwen_tts"}:
            return Qwen3TextToSpeech(self.config)
        if engine in {"file", "wav", "wave"}:
            return WaveFileTextToSpeech(
                self.config.audio_dir,
                playback_enabled=self.config.playback_enabled,
                realtime_enabled=self.config.realtime_enabled,
                agent_a_words_per_minute=self.config.agent_a_words_per_minute,
                agent_b_words_per_minute=self.config.agent_b_words_per_minute,
                max_duration_sec=self.config.max_utterance_sec,
                pattern_key=self.config.pattern_key,
                config=self.config,
            )
        raise SpeechPipelineError(
            f"Unsupported text-to-speech engine '{engine}'.",
            {"allowed": available_tts_engine_keys()},
        )

    def _default_asr_engine(self):
        engine = self._stage_engine(self.config.asr_engine)
        aliases = {
            "windows": "sapi",
            "windows_sapi": "sapi",
            "speech": "sapi",
            "whisper": "faster_whisper",
            "faster-whisper": "faster_whisper",
            "whisper.cpp": "whisper_cpp",
            "whisper-cpp": "whisper_cpp",
            "qwen3-asr": "qwen3_asr",
            "qwen_asr": "qwen3_asr",
            "wav": "file",
            "wave": "file",
        }
        engine = aliases.get(engine, engine)
        if engine not in ASR_ENGINE_SPECS:
            raise SpeechPipelineError(
                f"Unsupported automatic speech recognition engine '{engine}'.",
                {"allowed": available_asr_engine_keys()},
            )
        isolated = self._isolated_stage("asr", engine)
        if isolated is not None:
            return isolated
        if engine in {"sapi", "windows", "windows_sapi", "speech"}:
            if platform.system() != "Windows":
                raise SpeechPipelineError(
                    "Windows SAPI automatic speech recognition is only available on Windows.",
                    {"troubleshooting": "Select `faster_whisper` on Linux."},
                )
            return WindowsSapiSpeechToText(
                language=self.config.asr_language,
                initial_silence_sec=self.config.asr_initial_silence_sec,
                babble_timeout_sec=self.config.asr_babble_timeout_sec,
                end_silence_ms=self.config.asr_end_silence_ms,
                ambiguous_end_silence_ms=self.config.asr_ambiguous_end_silence_ms,
            )
        if engine in {"faster_whisper", "whisper", "faster-whisper"}:
            return FasterWhisperSpeechToText(
                model_name=self.config.asr_model,
                device=self.config.asr_device,
                compute_type=self.config.asr_compute_type,
                language=self.config.asr_language,
                beam_size=self.config.asr_beam_size,
                min_silence_duration_ms=self.config.asr_end_silence_ms,
                model_cache_dir=str(
                    Path(self.config.provider_environment_dir) / "models" / "faster-whisper"
                ),
            )
        if engine == "vosk":
            return VoskSpeechToText(
                model_name=self.config.asr_model,
                language=self.config.asr_language,
                search_width=self.config.asr_beam_size,
                model_cache_dir=str(
                    Path(self.config.provider_environment_dir) / "models" / "vosk"
                ),
            )
        if engine in {"sherpa_onnx", "sherpa-onnx", "sherpa"}:
            return SherpaOnnxSpeechToText(
                model_name=self.config.asr_model,
                language=self.config.asr_language,
                search_width=self.config.asr_beam_size,
            )
        if engine in {"whisper_cpp", "whisper.cpp", "whisper-cpp"}:
            return WhisperCppSpeechToText(
                model_name=self.config.asr_model,
                executable=self.config.asr_executable,
                language=self.config.asr_language,
                vad_model=self.config.asr_vad_model,
                search_width=self.config.asr_beam_size,
                timeout_sec=self.config.asr_timeout_sec,
                provider_environment_dir=self.config.provider_environment_dir,
            )
        if engine in {"parakeet", "nvidia_parakeet"}:
            return ParakeetSpeechToText(
                model_name=self.config.asr_model,
                device=self.config.asr_device,
            )
        if engine in {"qwen3_asr", "qwen3-asr", "qwen_asr"}:
            return Qwen3SpeechToText(
                model_name=self.config.asr_model,
                device=self.config.asr_device,
                language=self.config.asr_language,
                model_cache_dir=str(
                    Path(self.config.provider_environment_dir) / "models" / "qwen3-asr"
                ),
            )
        if engine in {"file", "wav", "wave"}:
            return WaveFileSpeechToText()
        raise SpeechPipelineError(
            f"Unsupported automatic speech recognition engine '{engine}'.",
            {"allowed": available_asr_engine_keys()},
        )

    def _stage_engine(self, stage_engine):
        return (stage_engine or "sapi").strip().lower()

    def _isolated_stage(self, stage, engine):
        """Use another provider interpreter only when explicitly configured."""
        if engine in {"sapi", "windows", "windows_sapi", "speech", "file", "wav", "wave"}:
            return None
        if stage == "asr" and engine in {"whisper_cpp", "whisper.cpp", "whisper-cpp"}:
            return None
        from coop_navigation_sds.DialogManagement.provider_runtime import (
            IsolatedSpeechToText,
            IsolatedTextToSpeech,
            resolve_provider_python,
        )

        explicit = (
            self.config.tts_python_executable
            if stage == "tts"
            else self.config.asr_python_executable
        )
        try:
            python = resolve_provider_python(
                engine,
                explicit=explicit,
                environment_dir=self.config.provider_environment_dir,
            )
        except FileNotFoundError as exc:
            raise SpeechPipelineError(
                str(exc),
                {
                    "stage": stage,
                    "engine": engine,
                    "troubleshooting": (
                        "Correct the configured provider Python path or run "
                        "`python scripts/setup_speech_providers.py`."
                    ),
                },
            ) from exc
        if python is None:
            return None
        if stage == "tts":
            return IsolatedTextToSpeech(engine, python, self.config)
        return IsolatedSpeechToText(engine, python, self.config)

    @property
    def description(self):
        """Description method for this module's MVC responsibility.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        return f"{self.config.label} ({self.tts_engine.name} to {self.asr_engine.name})"

    def validate_configuration(self):
        """Fail early unless both stages are actual configured components."""
        if not callable(getattr(self.tts_engine, "synthesize", None)):
            raise SpeechPipelineError("Text-to-speech implementation has no synthesize method.")
        if not callable(getattr(self.asr_engine, "transcribe", None)):
            raise SpeechPipelineError("Automatic speech recognition implementation has no transcribe method.")

    def close(self):
        """Release provider workers and their operating-system resources."""
        for engine in (self.tts_engine, self.asr_engine):
            client = getattr(engine, "client", None)
            if client is not None and callable(getattr(client, "close", None)):
                client.close()
            worker = getattr(engine, "_worker", None)
            if worker is not None and worker.poll() is None:
                worker.terminate()
                try:
                    worker.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    worker.kill()
            if worker is not None:
                for stream in (worker.stdin, worker.stdout, worker.stderr):
                    if stream is not None and not stream.closed:
                        stream.close()

    def health_check(self):
        """Run a short end-to-end check through both agents before a speech dialog."""
        checks = []
        probes = (
            (
                "Agent A",
                "I am at Bravo at 08:07, going to Harbor.",
                (("bravo",), ("harbor", "harbour")),
            ),
            (
                "Agent B",
                "Take metro line M1 from Bravo to Harbor with no changes.",
                (("m1",), ("bravo",), ("harbor", "harbour")),
            ),
        )
        for speaker, probe_text, expected_groups in probes:
            trace = self.transmit_trace(speaker, probe_text)
            audio = trace.audio if isinstance(trace.audio, dict) else {}
            audio_path = audio.get("path")
            transcript_ok = bool(trace.incoming_transcript.strip())
            audio_ok = bool(audio_path and Path(audio_path).exists())
            folded = trace.incoming_transcript.casefold()
            semantic_matches = [
                any(variant in folded for variant in variants)
                for variants in expected_groups
            ]
            missing_entity_groups = [
                list(variants)
                for variants, matched in zip(expected_groups, semantic_matches)
                if not matched
            ]
            semantic_coverage = sum(semantic_matches) / len(semantic_matches)
            semantic_ok = self.config.pattern_key != "clean" or semantic_coverage == 1.0
            operational_ok = trace.pipeline_ok and audio_ok and transcript_ok
            checks.append({
                "speaker": speaker,
                "pipeline_ok": operational_ok,
                "quality_ok": semantic_ok,
                "tts_engine": trace.tts_engine,
                "asr_engine": trace.asr_engine,
                "audio": trace.audio,
                "audio_path": audio_path,
                "audio_ok": audio_ok,
                "probe_text": probe_text,
                "incoming_transcript": trace.incoming_transcript,
                "transcript_ok": transcript_ok,
                "semantic_coverage": semantic_coverage,
                "semantic_ok": semantic_ok,
                "expected_entity_groups": [list(group) for group in expected_groups],
                "missing_entity_groups": missing_entity_groups,
                "diagnostics": trace.diagnostics or {},
            })
        audio_root_value = getattr(self.config, "audio_dir", None)
        if audio_root_value:
            audio_root = Path(audio_root_value).resolve()
            for check in checks:
                removed = []
                audio = check.get("audio") if isinstance(check.get("audio"), dict) else {}
                for value in (audio.get("path"), audio.get("transcript_path")):
                    if not value:
                        continue
                    path = Path(value)
                    try:
                        path.resolve().relative_to(audio_root)
                    except (OSError, ValueError):
                        continue
                    try:
                        path.unlink(missing_ok=True)
                        removed.append(str(path))
                    except OSError:
                        continue
                check["removed_probe_artifacts"] = removed
        operational_ok = all(check["pipeline_ok"] for check in checks)
        quality_ok = all(check["quality_ok"] for check in checks)
        return {
            "mode": "speech",
            # Readiness and recognition quality are separate signals. A usable
            # audio/transcription path may produce imperfect entity recognition;
            # that degradation belongs in retrospective ASR metrics, not preflight.
            "ok": operational_ok,
            "operational_ok": operational_ok,
            "quality_ok": quality_ok,
            "checks": checks,
        }

    def transmit(self, speaker: str, text: str):
        """Transmit method for this module's MVC responsibility.
        
        Args:
            speaker: Input value used by `transmit`; see the function signature and caller context for the expected type.
            text: Input value used by `transmit`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        trace = self.transmit_trace(speaker, text)
        return trace.signal, trace.incoming_transcript

    def transmit_trace(self, speaker: str, text: str) -> SpeechPipelineTrace:
        """Run mandatory TTS and ASR stages and return a full trace."""
        if not self.config.applies_to(speaker):
            raise SpeechPipelineError("Speech transport only accepts Agent A and Agent B turns.")
        outgoing_enabled = True
        incoming_enabled = True

        pipeline_started_at = time.perf_counter()
        try:
            prosody = self.config.prosody_for(speaker)
            delivered_text = _apply_persona_delivery(text, speaker, prosody)
            tts_started_at = time.perf_counter()
            signal = self.tts_engine.synthesize(speaker, delivered_text)
            tts_latency_sec = time.perf_counter() - tts_started_at
            channel_started_at = time.perf_counter()
            signal = _apply_audio_channel_impairment(
                signal,
                self.config,
                speaker,
            )
            channel_latency_sec = time.perf_counter() - channel_started_at
            signal_diagnostics = signal.diagnostics if isinstance(signal.diagnostics, dict) else {}
            signal.diagnostics = signal_diagnostics
            signal_diagnostics["audio_channel_processing_sec"] = channel_latency_sec
            simulated_duration_sec = self.estimate_duration_sec(speaker, signal.text)
            audio = signal.audio if isinstance(signal.audio, dict) else {}
            if not audio.get("path"):
                raise SpeechPipelineError(
                    "Text-to-speech did not produce an audio artifact.",
                    {"speaker": speaker, "tts_engine": self.tts_engine.name},
                )
            if self.config.realtime_enabled and outgoing_enabled and not audio.get("waited"):
                time.sleep(simulated_duration_sec)
                if isinstance(signal.audio, dict):
                    signal.audio["waited"] = True
                    signal.audio["software_wait_sec"] = simulated_duration_sec
            asr_started_at = time.perf_counter()
            raw_transcript = self.asr_engine.transcribe(signal)
            raw_transcript = _apply_persona_recognition_errors(raw_transcript, speaker, prosody)
            transcript = raw_transcript
            if self.config.asr_domain_normalization_enabled:
                transcript = normalize_transit_transcript(
                    raw_transcript,
                    threshold=self.config.asr_domain_similarity_threshold,
                )
            misinterpretations = transcript_token_changes(signal.text, raw_transcript)
            transcript_corrections = transcript_token_changes(raw_transcript, transcript)
            diagnostics = signal.diagnostics if isinstance(signal.diagnostics, dict) else {}
            signal.diagnostics = diagnostics
            diagnostics.setdefault("raw_asr_transcript", raw_transcript)
            diagnostics["misinterpreted_tokens"] = misinterpretations
            diagnostics["transcript_corrections"] = transcript_corrections
            diagnostics["agent_input_transcript"] = transcript
            diagnostics["audio_persona"] = prosody["audio_persona"]
            diagnostics["audio_persona_clarity"] = prosody["clarity_level"]
            diagnostics["persona_delivery_changed"] = delivered_text != text
            diagnostics["persona_recognition_changed"] = raw_transcript.casefold() != signal.text.casefold()
            diagnostics["pipeline_asr_transcript"] = transcript
            diagnostics["asr_domain_normalization_enabled"] = bool(
                self.config.asr_domain_normalization_enabled
            )
            diagnostics["asr_domain_normalization_used"] = (
                transcript.casefold() != str(raw_transcript).strip().casefold()
            )
            asr_latency_sec = time.perf_counter() - asr_started_at
            if not transcript.strip():
                raise SpeechPipelineError(
                    "Automatic speech recognition produced an empty transcript.",
                    {"speaker": speaker, "asr_engine": self.asr_engine.name, "audio": signal.audio},
                )
            return SpeechPipelineTrace(
                speaker=speaker,
                generated_text=text,
                outgoing_text=signal.text,
                incoming_transcript=transcript,
                outgoing_enabled=outgoing_enabled,
                incoming_enabled=incoming_enabled,
                tts_engine=self.tts_engine.name if outgoing_enabled else "disabled",
                asr_engine=self.asr_engine.name if incoming_enabled else "disabled",
                pattern_key=self.config.pattern_key,
                simulated_duration_sec=simulated_duration_sec,
                audio=signal.audio,
                mode="speech",
                pipeline_ok=True,
                diagnostics=signal.diagnostics or {},
                tts_latency_sec=round(tts_latency_sec, 6),
                asr_latency_sec=round(asr_latency_sec, 6),
                pipeline_latency_sec=round(time.perf_counter() - pipeline_started_at, 6),
            )
        except SpeechPipelineError:
            raise
        except Exception as exc:
            raise SpeechPipelineError(
                "Speech pipeline failed unexpectedly.",
                {
                    "speaker": speaker,
                    "stage": "transmit",
                    "error": repr(exc),
                    "troubleshooting": "Check selected speech engines, audio output, generated audio files, and recognition availability.",
                },
            ) from exc

    def estimate_duration_sec(self, speaker: str, text: str) -> float:
        """Estimate natural speech duration from utterance length and speaker rate."""
        words = re.findall(r"[A-Za-z0-9]+", text)
        prosody = self.config.prosody_for(speaker)
        rate = prosody["words_per_minute"]
        settings = speech_pattern_settings(self.config.pattern_key)
        duration_multiplier = float(settings.get("duration_multiplier", 1.0) or 1.0)
        seconds = len(words) * 60 / max(rate, 1)
        seconds *= max(duration_multiplier, 0.1)
        seconds += punctuation_pause_duration_sec(text, prosody["pause_ms"])
        return round(min(max(seconds, self.config.min_utterance_sec), self.config.max_utterance_sec), 3)


class PatternedSpeechToText:
    """Configurable ASR simulator that injects configured transcript patterns."""
    def __init__(self, pattern_key="clean", seed=0):
        """  init   method for this module's MVC responsibility.
        
        Args:
            pattern_key: Input value used by `__init__`; see the function signature and caller context for the expected type.
            seed: Input value used by `__init__`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        self.pattern_key = pattern_key
        self.name = f"patterned-asr:{pattern_key}"
        self.rng = random.Random(seed)
        self.settings = speech_pattern_settings(pattern_key)

    def transcribe(self, signal: SpeechSignal) -> str:
        """Transcribe method for this module's MVC responsibility.
        
        Args:
            signal: Input value used by `transcribe`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        return self.transform_text(signal.text, include_recognition_errors=True)

    def transform_text(self, text: str, include_recognition_errors=True) -> str:
        """Apply the configured speech-pattern transformations."""
        if self.settings.get("compression_enabled"):
            text = self._compress(text)
        text = self._insert_tokens(
            text,
            probability=float(self.settings.get("hesitation_probability", 0.0) or 0.0),
            tokens=self.settings.get("hesitation_tokens", ()),
            sentence_start=True,
        )
        text = self._insert_tokens(
            text,
            probability=float(self.settings.get("filler_probability", 0.0) or 0.0),
            tokens=self.settings.get("filler_tokens", ()),
            sentence_start=False,
        )
        text = self._insert_tokens(
            text,
            probability=float(self.settings.get("pause_probability", 0.0) or 0.0),
            tokens=self.settings.get("pause_tokens", ()),
            sentence_start=False,
        )
        text = self._add_stutters(
            text,
            probability=float(self.settings.get("stutter_probability", 0.0) or 0.0),
            max_words=int(self.settings.get("stutter_max_words", 1) or 1),
        )
        if include_recognition_errors:
            text = self._substitute_words(
                text,
                probability=float(self.settings.get("substitution_probability", 0.0) or 0.0),
                substitutions=self.settings.get("substitutions", {}),
            )
            text = self._drop_some_words(
                text,
                drop_probability=float(self.settings.get("drop_probability", 0.0) or 0.0),
                protected_terms=self.settings.get("protected_terms", ()),
            )
        return re.sub(r"\s+", " ", text).strip()

    def _insert_tokens(self, text: str, probability: float, tokens, sentence_start=False) -> str:
        """Insert configured filler, pause, or hesitation tokens."""
        tokens = list(tokens or [])
        if probability <= 0.0 or not tokens:
            return text
        sentences = re.split(r"(?<=[.!?])\s+", text)
        out = []
        for sentence in sentences:
            if not sentence or self.rng.random() >= probability:
                out.append(sentence)
                continue
            token = self.rng.choice(tokens)
            if sentence_start:
                out.append(f"{token}, {sentence}")
            else:
                words = sentence.split()
                if not words:
                    out.append(sentence)
                    continue
                index = self.rng.randint(0, len(words))
                words.insert(index, token)
                out.append(" ".join(words))
        return " ".join(out)

    def _add_stutters(self, text: str, probability: float, max_words=1) -> str:
        if probability <= 0.0:
            return text
        words = text.split()
        if not words:
            return text
        changed = 0
        out = []
        for word in words:
            if changed < max_words and re.match(r"^[A-Za-z]{3,}", word) and self.rng.random() < probability:
                out.append(f"{word[0]}-{word}")
                changed += 1
            else:
                out.append(word)
        return " ".join(out)

    def _compress(self, text: str) -> str:
        """ compress method for this module's MVC responsibility.
        
        Args:
            text: Input value used by `_compress`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        text = re.sub(r"\bplease\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\bI would\b", "I'd", text)
        return re.sub(r"\s+", " ", text).strip()

    def _substitute_words(self, text: str, probability: float, substitutions) -> str:
        if probability <= 0.0 or not isinstance(substitutions, dict):
            return text
        words = text.split()
        out = []
        for word in words:
            clean = re.sub(r"[^A-Za-z0-9]", "", word)
            replacement = substitutions.get(clean)
            if replacement and self.rng.random() < probability:
                out.append(word.replace(clean, str(replacement)))
            else:
                out.append(word)
        return " ".join(out)

    def _drop_some_words(self, text: str, drop_probability: float, protected_terms=()) -> str:
        """ drop some words method for this module's MVC responsibility.
        
        Args:
            text: Input value used by `_drop_some_words`; see the function signature and caller context for the expected type.
            drop_probability: Input value used by `_drop_some_words`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        if drop_probability <= 0.0:
            return text
        protected = {str(term).lower() for term in protected_terms or ()}
        words = text.split()
        kept = [
            word
            for word in words
            if re.sub(r"[^A-Za-z0-9]", "", word).lower() in protected
            or self.rng.random() >= drop_probability
        ]
        return " ".join(kept) if kept else text
