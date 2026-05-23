"""Speech transport abstractions and simulated ASR/TTS transformations for dialog experiments.
"""
from dataclasses import dataclass
import random
import re
from typing import Protocol

from minillama.agent_b.config import (
    DEFAULT_SPEECH_PATTERN,
    SPEECH_INCOMING_ENABLED,
    SPEECH_OUTGOING_ENABLED,
    SPEECH_PATTERNS,
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
    agent_a_words_per_minute: int = 165
    agent_b_words_per_minute: int = 175
    min_utterance_sec: float = 0.8
    max_utterance_sec: float = 8.0

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
        return f"{'+'.join(directions)}:{self.pattern_key}:{self.scope}"


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

    @property
    def signal(self):
        return SpeechSignal(speaker=self.speaker, text=self.outgoing_text, audio=None)


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
        self.tts_engine = tts_engine or PatternedTextToSpeech(self.config.pattern_key)
        self.asr_engine = asr_engine or PatternedSpeechToText(self.config.pattern_key)

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
