"""Speech transport abstractions and simulated ASR/TTS transformations for dialog experiments.
"""
from dataclasses import dataclass
import random
import re
from typing import Protocol

from minillama.config import SPEECH_PATTERNS

@dataclass
class SpeechSignal:
    """Speech payload model passed between simulated TTS and ASR components.
    """
    speaker: str
    text: str
    audio: object = None


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
    def __init__(self, tts_engine=None, asr_engine=None):
        """  init   method for this module's MVC responsibility.
        
        Args:
            tts_engine: Input value used by `__init__`; see the function signature and caller context for the expected type.
            asr_engine: Input value used by `__init__`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        self.tts_engine = tts_engine or LoopbackTextToSpeech()
        self.asr_engine = asr_engine or LoopbackSpeechToText()

    @property
    def description(self):
        """Description method for this module's MVC responsibility.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        return f"{self.tts_engine.name} -> {self.asr_engine.name}"

    def transmit(self, speaker: str, text: str):
        """Transmit method for this module's MVC responsibility.
        
        Args:
            speaker: Input value used by `transmit`; see the function signature and caller context for the expected type.
            text: Input value used by `transmit`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        signal = self.tts_engine.synthesize(speaker, text)
        transcript = self.asr_engine.transcribe(signal)
        return signal, transcript


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
