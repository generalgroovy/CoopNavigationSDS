"""Speech transport abstractions and simulated ASR/TTS transformations for dialog experiments.
"""
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import random
import re
import shutil
import struct
import subprocess
import time
from typing import Protocol
import wave

from minillama.agent_b.config import (
    DEFAULT_SPEECH_PATTERN,
    RUN_MODE,
    SPEECH_AUDIO_DIR,
    SPEECH_ASR_ENGINE,
    SPEECH_ENGINE,
    SPEECH_INCOMING_ENABLED,
    SPEECH_OUTGOING_ENABLED,
    SPEECH_PATTERNS,
    SPEECH_PLAYBACK_ENABLED,
    SPEECH_REALTIME_ENABLED,
    SPEECH_SCOPE,
    SPEECH_TTS_ENGINE,
)


class SpeechPipelineError(RuntimeError):
    """Raised when a strict speech pipeline stage fails."""

    def __init__(self, message, diagnostics=None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


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
    """Runtime configuration for optional speech stages."""
    mode: str = RUN_MODE
    incoming_enabled: bool = SPEECH_INCOMING_ENABLED
    outgoing_enabled: bool = SPEECH_OUTGOING_ENABLED
    scope: str = SPEECH_SCOPE
    pattern_key: str = DEFAULT_SPEECH_PATTERN
    engine: str = SPEECH_ENGINE
    tts_engine: str = SPEECH_TTS_ENGINE
    asr_engine: str = SPEECH_ASR_ENGINE
    audio_dir: str = SPEECH_AUDIO_DIR
    agent_a_words_per_minute: int = 165
    agent_b_words_per_minute: int = 175
    min_utterance_sec: float = 0.6
    max_utterance_sec: float = 3.5
    playback_enabled: bool = SPEECH_PLAYBACK_ENABLED
    realtime_enabled: bool = SPEECH_REALTIME_ENABLED

    @property
    def normalized_mode(self) -> str:
        mode = (self.mode or "pure_text").strip().lower().replace("-", "_")
        if mode in {"text", "puretext", "pure_text", "off"}:
            return "pure_text"
        if mode in {"speech", "audio", "spoken"}:
            return "speech"
        raise SpeechPipelineError(
            f"Unsupported run mode '{self.mode}'. Use pure_text or speech.",
            {"mode": self.mode},
        )

    @property
    def strict_speech(self) -> bool:
        return self.normalized_mode == "speech"

    def applies_to(self, speaker: str) -> bool:
        if self.normalized_mode == "pure_text":
            return False
        if self.scope in {"both", "all", "*"}:
            return True
        if self.scope in {"none", "off", "text"}:
            return False
        normalized = speaker.lower().replace(" ", "_")
        return self.scope == normalized or self.scope == normalized.replace("_", "")

    @property
    def label(self) -> str:
        if self.normalized_mode == "pure_text":
            return "pure_text"
        directions = []
        if self.outgoing_enabled:
            directions.append("outgoing")
        if self.incoming_enabled:
            directions.append("incoming")
        if not directions:
            directions.append("text-only")
        playback = ":playback" if self.playback_enabled else ""
        realtime = ":realtime" if self.realtime_enabled else ""
        tts = self.tts_engine or self.engine
        asr = self.asr_engine or self.engine
        return f"speech:{'+'.join(directions)}:{self.pattern_key}:{self.scope}:tts={tts}:asr={asr}{playback}{realtime}"


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
    mode: str = "pure_text"
    pipeline_ok: bool = True
    failure_reason: str | None = None
    diagnostics: dict | None = None

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


class LoopbackTextToSpeech:
    """TTS test double that returns text unchanged as a speech signal.
    """
    name = "loopback-tts"

    def synthesize(self, speaker: str, text: str) -> SpeechSignal:
        """Synthesize method for this module's MVC responsibility.
        
        Args:
            speaker: Input value used by `synthesize`; see the function signature and caller context for the expected type.
            text: Input value used by `synthesize`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        return SpeechSignal(speaker=speaker, text=text, audio=None)


class PatternedTextToSpeech:
    """Optional outgoing speech style simulator."""
    def __init__(self, pattern_key="clean", seed=0):
        self.pattern_key = pattern_key
        self.name = f"patterned-tts:{pattern_key}"
        self.rng = random.Random(seed)

    def synthesize(self, speaker: str, text: str) -> SpeechSignal:
        if self.pattern_key == "hesitant":
            text = PatternedSpeechToText(self.pattern_key, seed=self.rng.randint(0, 10**6))._add_hesitations(text)
        elif self.pattern_key == "compressed":
            text = PatternedSpeechToText(self.pattern_key)._compress(text)
        return SpeechSignal(speaker=speaker, text=text, audio=None)


class WaveFileTextToSpeech:
    """Dependency-free TTS adapter that writes a simple WAV carrier plus transcript sidecar."""

    name = "wavefile-tts"

    def __init__(
        self,
        audio_dir="speech_artifacts",
        sample_rate=8000,
        playback_enabled=False,
        realtime_enabled=False,
        agent_a_words_per_minute=165,
        agent_b_words_per_minute=175,
        max_duration_sec=3.5,
    ):
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.playback_enabled = playback_enabled
        self.realtime_enabled = realtime_enabled
        self.agent_a_words_per_minute = agent_a_words_per_minute
        self.agent_b_words_per_minute = agent_b_words_per_minute
        self.max_duration_sec = max_duration_sec
        self._counter = 0

    def synthesize(self, speaker: str, text: str) -> SpeechSignal:
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self._counter += 1
        digest = hashlib.sha1(f"{speaker}:{self._counter}:{text}".encode("utf-8")).hexdigest()[:10]
        stem = f"{self._counter:04d}-{speaker.lower().replace(' ', '-')}-{digest}"
        wav_path = self.audio_dir / f"{stem}.wav"
        transcript_path = self.audio_dir / f"{stem}.txt"
        duration_sec = self._write_wave(wav_path, speaker, text)
        transcript_path.write_text(text, encoding="utf-8")
        played = (
            self._play_wave(wav_path, realtime=self.realtime_enabled, fallback_duration=duration_sec)
            if self.playback_enabled
            else False
        )
        waited = bool(self.playback_enabled and self.realtime_enabled)
        return SpeechSignal(
            speaker=speaker,
            text=text,
            audio={
                "engine": "wavefile",
                "path": str(wav_path),
                "transcript_path": str(transcript_path),
                "sample_rate": self.sample_rate,
                "duration_sec": duration_sec,
                "played": played,
                "realtime": self.realtime_enabled,
                "waited": waited,
            },
            diagnostics={"synthesis": "wave carrier with transcript sidecar"},
        )

    def _write_wave(self, path: Path, speaker: str, text: str):
        words = re.findall(r"[A-Za-z0-9]+", text)
        rate = self.agent_b_words_per_minute if speaker.lower().replace(" ", "_") == "agent_b" else self.agent_a_words_per_minute
        duration = min(max(len(words) * 60 / max(rate, 1), 0.45), self.max_duration_sec)
        samples = int(self.sample_rate * duration)
        base_frequency = 185 if speaker.lower().replace(" ", "_") == "agent_a" else 230
        amplitude = 7500
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
            import winsound

            flags = winsound.SND_FILENAME
            if not realtime:
                flags |= winsound.SND_ASYNC
            winsound.PlaySound(str(path), flags)
            return True
        except Exception:
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


class WindowsSapiTextToSpeech:
    """Windows SAPI TTS stage that writes actual spoken WAV audio."""

    name = "windows-sapi-tts"

    def __init__(self, audio_dir="speech_artifacts", playback_enabled=False, realtime_enabled=False):
        self.audio_dir = Path(audio_dir)
        self.playback_enabled = playback_enabled
        self.realtime_enabled = realtime_enabled
        self._counter = 0

    def synthesize(self, speaker: str, text: str) -> SpeechSignal:
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self._counter += 1
        digest = hashlib.sha1(f"sapi:{speaker}:{self._counter}:{text}".encode("utf-8")).hexdigest()[:10]
        stem = f"{self._counter:04d}-{speaker.lower().replace(' ', '-')}-{digest}"
        wav_path = self.audio_dir / f"{stem}.wav"
        transcript_path = self.audio_dir / f"{stem}.txt"
        transcript_path.write_text(text, encoding="utf-8")
        command = self._powershell_command(wav_path)
        completed = subprocess.run(
            command,
            input=text,
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
                    "troubleshooting": "Install or enable Windows speech synthesis voices, or use pure_text mode.",
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
            text=text,
            audio={
                "engine": "windows_sapi",
                "path": str(wav_path),
                "transcript_path": str(transcript_path),
                "duration_sec": duration_sec,
                "played": played,
                "realtime": self.realtime_enabled,
                "waited": bool(self.playback_enabled and self.realtime_enabled),
            },
            diagnostics={"synthesis": "windows_sapi"},
        )

    @staticmethod
    def _powershell_command(wav_path: Path):
        powershell = shutil.which("powershell") or shutil.which("powershell.exe") or "powershell"
        escaped_path = str(wav_path).replace("'", "''")
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$text = [Console]::In.ReadToEnd(); "
            "$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$speaker.SetOutputToWaveFile('{escaped_path}'); "
            "$speaker.Speak($text); "
            "$speaker.Dispose();"
        )
        return [powershell, "-NoProfile", "-NonInteractive", "-Command", script]

    @staticmethod
    def _wave_duration(path: Path) -> float:
        with wave.open(str(path), "rb") as handle:
            return round(handle.getnframes() / float(handle.getframerate()), 3)


class WindowsSapiSpeechToText:
    """Windows SAPI ASR stage that transcribes a WAV file."""

    name = "windows-sapi-asr"

    def transcribe(self, signal: SpeechSignal) -> str:
        audio = signal.audio if isinstance(signal.audio, dict) else {}
        wav_path = audio.get("path")
        if not wav_path or not Path(wav_path).exists():
            raise SpeechPipelineError(
                "Automatic speech recognition failed; no audio file reached the recognizer.",
                {"engine": self.name, "audio": audio},
            )
        command = self._powershell_command(Path(wav_path))
        completed = subprocess.run(command, text=True, capture_output=True, timeout=30)
        transcript = completed.stdout.strip()
        if completed.returncode != 0 or not transcript:
            raise SpeechPipelineError(
                "Automatic speech recognition failed; no transcript was produced from audio.",
                {
                    "engine": self.name,
                    "return_code": completed.returncode,
                    "stderr": completed.stderr.strip(),
                    "path": str(wav_path),
                    "troubleshooting": "Install a Windows speech recognizer for the language, verify microphone/speech services, or use pure_text mode.",
                },
            )
        return transcript

    @staticmethod
    def _powershell_command(wav_path: Path):
        powershell = shutil.which("powershell") or shutil.which("powershell.exe") or "powershell"
        escaped_path = str(wav_path).replace("'", "''")
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$recognizer = New-Object System.Speech.Recognition.SpeechRecognitionEngine; "
            "$recognizer.LoadGrammar((New-Object System.Speech.Recognition.DictationGrammar)); "
            f"$recognizer.SetInputToWaveFile('{escaped_path}'); "
            "$result = $recognizer.Recognize(); "
            "if ($result -ne $null) { [Console]::Out.Write($result.Text) }; "
            "$recognizer.Dispose();"
        )
        return [powershell, "-NoProfile", "-NonInteractive", "-Command", script]


class LoopbackSpeechToText:
    """ASR test double that returns the signal text unchanged.
    """
    name = "loopback-asr"

    def transcribe(self, signal: SpeechSignal) -> str:
        """Transcribe method for this module's MVC responsibility.
        
        Args:
            signal: Input value used by `transcribe`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        return signal.text


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
        if self.config.normalized_mode == "pure_text":
            return LoopbackTextToSpeech()
        engine = self._stage_engine(self.config.tts_engine)
        if engine in {"sapi", "windows", "windows_sapi", "speech"}:
            return WindowsSapiTextToSpeech(
                self.config.audio_dir,
                playback_enabled=self.config.playback_enabled,
                realtime_enabled=self.config.realtime_enabled,
            )
        if engine in {"file", "wav", "wave"}:
            return WaveFileTextToSpeech(
                self.config.audio_dir,
                playback_enabled=self.config.playback_enabled,
                realtime_enabled=self.config.realtime_enabled,
                agent_a_words_per_minute=self.config.agent_a_words_per_minute,
                agent_b_words_per_minute=self.config.agent_b_words_per_minute,
                max_duration_sec=self.config.max_utterance_sec,
            )
        if engine in {"loopback", "text", "off", "none"}:
            return LoopbackTextToSpeech()
        return PatternedTextToSpeech(self.config.pattern_key)

    def _default_asr_engine(self):
        if self.config.normalized_mode == "pure_text":
            return LoopbackSpeechToText()
        engine = self._stage_engine(self.config.asr_engine)
        if engine in {"sapi", "windows", "windows_sapi", "speech"}:
            return WindowsSapiSpeechToText()
        if engine in {"file", "wav", "wave"}:
            return WaveFileSpeechToText()
        if engine in {"loopback", "text", "off", "none"}:
            return LoopbackSpeechToText()
        return PatternedSpeechToText(self.config.pattern_key)

    def _stage_engine(self, stage_engine):
        return (stage_engine or self.config.engine or "patterned").strip().lower()

    @property
    def description(self):
        """Description method for this module's MVC responsibility.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        return f"{self.config.label} ({self.tts_engine.name} to {self.asr_engine.name})"

    def validate_configuration(self):
        """Fail early when speech mode is not a complete spoken pipeline."""
        if self.config.normalized_mode == "pure_text":
            return
        errors = []
        if not self.config.outgoing_enabled:
            errors.append("outgoing text-to-speech is disabled")
        if not self.config.incoming_enabled:
            errors.append("incoming automatic speech recognition is disabled")
        if self.config.scope in {"none", "off", "text"}:
            errors.append("speech scope disables both agents")
        if isinstance(self.tts_engine, LoopbackTextToSpeech):
            errors.append("loopback text-to-speech is not speech synthesis")
        if isinstance(self.asr_engine, LoopbackSpeechToText):
            errors.append("loopback automatic speech recognition is not speech recognition")
        if errors:
            raise SpeechPipelineError(
                "Speech mode requires a complete text-to-speech and automatic speech recognition pipeline.",
                {
                    "errors": errors,
                    "mode": self.config.mode,
                    "tts_engine": self.tts_engine.name,
                    "asr_engine": self.asr_engine.name,
                    "troubleshooting": "Use pure_text mode for text-only runs, or configure real speech engines for speech mode.",
                },
            )

    def health_check(self):
        """Run a short end-to-end check through both agents before a speech dialog."""
        if self.config.normalized_mode == "pure_text":
            return {
                "mode": "pure_text",
                "ok": True,
                "message": "Speech pipeline disabled; generated text is passed directly.",
            }
        checks = []
        for speaker in ("Agent A", "Agent B"):
            trace = self.transmit_trace(speaker, f"Speech pipeline check from {speaker}.")
            checks.append({
                "speaker": speaker,
                "pipeline_ok": trace.pipeline_ok,
                "tts_engine": trace.tts_engine,
                "asr_engine": trace.asr_engine,
                "audio": trace.audio,
            })
        return {"mode": "speech", "ok": all(check["pipeline_ok"] for check in checks), "checks": checks}

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
        """Run optional outgoing and incoming speech stages and return a full trace."""
        active = self.config.applies_to(speaker)
        outgoing_enabled = bool(active and self.config.outgoing_enabled)
        incoming_enabled = bool(active and self.config.incoming_enabled)

        try:
            signal = (
                self.tts_engine.synthesize(speaker, text)
                if outgoing_enabled
                else SpeechSignal(speaker=speaker, text=text, audio=None)
            )
            simulated_duration_sec = self.estimate_duration_sec(speaker, signal.text)
            audio = signal.audio if isinstance(signal.audio, dict) else {}
            if self.config.strict_speech and outgoing_enabled and not audio.get("path"):
                raise SpeechPipelineError(
                    "Text-to-speech did not produce an audio artifact.",
                    {"speaker": speaker, "tts_engine": self.tts_engine.name},
                )
            if self.config.realtime_enabled and outgoing_enabled and not audio.get("waited"):
                time.sleep(simulated_duration_sec)
                if isinstance(signal.audio, dict):
                    signal.audio["waited"] = True
                    signal.audio["software_wait_sec"] = simulated_duration_sec
            transcript = (
                self.asr_engine.transcribe(signal)
                if incoming_enabled
                else signal.text
            )
            if self.config.strict_speech and incoming_enabled and not transcript.strip():
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
                mode=self.config.normalized_mode,
                pipeline_ok=True,
                diagnostics=signal.diagnostics or {},
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
        rate = (
            self.config.agent_b_words_per_minute
            if speaker.lower().replace(" ", "_") == "agent_b"
            else self.config.agent_a_words_per_minute
        )
        seconds = len(words) * 60 / max(rate, 1)
        return round(min(max(seconds, self.config.min_utterance_sec), self.config.max_utterance_sec), 3)


class PatternedSpeechToText:
    """Configurable ASR simulator that injects clean, hesitant, compressed, or noisy transcript patterns.
    """
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

    def transcribe(self, signal: SpeechSignal) -> str:
        """Transcribe method for this module's MVC responsibility.
        
        Args:
            signal: Input value used by `transcribe`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        text = signal.text

        if self.pattern_key == "clean":
            return text

        if self.pattern_key == "hesitant":
            return self._add_hesitations(text)

        if self.pattern_key == "compressed":
            return self._compress(text)

        if self.pattern_key == "noisy_station":
            return self._drop_some_words(text, drop_probability=SPEECH_PATTERNS["noisy_station_drop_probability"])

        return text

    def _add_hesitations(self, text: str) -> str:
        """ add hesitations method for this module's MVC responsibility.
        
        Args:
            text: Input value used by `_add_hesitations`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        sentences = re.split(r"(?<=[.!?])\s+", text)
        out = []
        for sentence in sentences:
            if sentence and self.rng.random() < SPEECH_PATTERNS["hesitation_probability"]:
                out.append(self.rng.choice(SPEECH_PATTERNS["hesitation_tokens"]) + ", " + sentence)
            else:
                out.append(sentence)
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

    def _drop_some_words(self, text: str, drop_probability: float) -> str:
        """ drop some words method for this module's MVC responsibility.
        
        Args:
            text: Input value used by `_drop_some_words`; see the function signature and caller context for the expected type.
            drop_probability: Input value used by `_drop_some_words`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        words = text.split()
        kept = [
            word
            for word in words
            if self.rng.random() >= drop_probability
        ]
        return " ".join(kept) if kept else text
