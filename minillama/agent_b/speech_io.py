"""Speech transport abstractions and simulated ASR/TTS transformations for dialog experiments.
"""
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import random
import re
import struct
from typing import Protocol
import wave

from minillama.agent_b.config import (
    DEFAULT_SPEECH_PATTERN,
    SPEECH_AUDIO_DIR,
    SPEECH_ENGINE,
    SPEECH_INCOMING_ENABLED,
    SPEECH_OUTGOING_ENABLED,
    SPEECH_PATTERNS,
    SPEECH_PLAYBACK_ENABLED,
    SPEECH_SCOPE,
)

@dataclass
class SpeechSignal:
    """Speech payload model passed between simulated TTS and ASR components.
    """
    speaker: str
    text: str
    audio: object = None


@dataclass(frozen=True)
class SpeechPipelineConfig:
    """Runtime configuration for optional speech stages."""
    incoming_enabled: bool = SPEECH_INCOMING_ENABLED
    outgoing_enabled: bool = SPEECH_OUTGOING_ENABLED
    scope: str = SPEECH_SCOPE
    pattern_key: str = DEFAULT_SPEECH_PATTERN
    engine: str = SPEECH_ENGINE
    audio_dir: str = SPEECH_AUDIO_DIR
    agent_a_words_per_minute: int = 165
    agent_b_words_per_minute: int = 175
    min_utterance_sec: float = 0.6
    max_utterance_sec: float = 3.5
    playback_enabled: bool = SPEECH_PLAYBACK_ENABLED

    def applies_to(self, speaker: str) -> bool:
        if self.scope in {"both", "all", "*"}:
            return True
        if self.scope in {"none", "off", "text"}:
            return False
        normalized = speaker.lower().replace(" ", "_")
        return self.scope == normalized or self.scope == normalized.replace("_", "")

    @property
    def label(self) -> str:
        directions = []
        if self.outgoing_enabled:
            directions.append("outgoing")
        if self.incoming_enabled:
            directions.append("incoming")
        if not directions:
            directions.append("text-only")
        playback = ":playback" if self.playback_enabled else ""
        return f"{'+'.join(directions)}:{self.pattern_key}:{self.scope}{playback}"


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
        agent_a_words_per_minute=165,
        agent_b_words_per_minute=175,
        max_duration_sec=3.5,
    ):
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.playback_enabled = playback_enabled
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
        played = self._play_wave(wav_path) if self.playback_enabled else False
        return SpeechSignal(
            speaker=speaker,
            text=text,
            audio={
                "path": str(wav_path),
                "transcript_path": str(transcript_path),
                "sample_rate": self.sample_rate,
                "duration_sec": duration_sec,
                "played": played,
            },
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
    def _play_wave(path: Path) -> bool:
        try:
            import winsound

            winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC)
            return True
        except Exception:
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

    def _default_tts_engine(self):
        if self.config.engine in {"file", "wav", "wave"}:
            return WaveFileTextToSpeech(
                self.config.audio_dir,
                playback_enabled=self.config.playback_enabled,
                agent_a_words_per_minute=self.config.agent_a_words_per_minute,
                agent_b_words_per_minute=self.config.agent_b_words_per_minute,
                max_duration_sec=self.config.max_utterance_sec,
            )
        return PatternedTextToSpeech(self.config.pattern_key)

    def _default_asr_engine(self):
        if self.config.engine in {"file", "wav", "wave"}:
            return WaveFileSpeechToText()
        return PatternedSpeechToText(self.config.pattern_key)

    @property
    def description(self):
        """Description method for this module's MVC responsibility.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        return f"{self.config.label} ({self.tts_engine.name} -> {self.asr_engine.name})"

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

        signal = (
            self.tts_engine.synthesize(speaker, text)
            if outgoing_enabled
            else SpeechSignal(speaker=speaker, text=text, audio=None)
        )
        transcript = (
            self.asr_engine.transcribe(signal)
            if incoming_enabled
            else signal.text
        )
        simulated_duration_sec = self.estimate_duration_sec(speaker, signal.text)
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
        )

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
