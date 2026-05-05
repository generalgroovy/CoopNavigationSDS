from dataclasses import dataclass
import random
import re
from typing import Protocol


@dataclass
class SpeechSignal:
    speaker: str
    text: str
    audio: object = None


class TextToSpeechEngine(Protocol):
    name: str

    def synthesize(self, speaker: str, text: str) -> SpeechSignal:
        ...


class SpeechToTextEngine(Protocol):
    name: str

    def transcribe(self, signal: SpeechSignal) -> str:
        ...


class LoopbackTextToSpeech:
    name = "loopback-tts"

    def synthesize(self, speaker: str, text: str) -> SpeechSignal:
        return SpeechSignal(speaker=speaker, text=text, audio=None)


class LoopbackSpeechToText:
    name = "loopback-asr"

    def transcribe(self, signal: SpeechSignal) -> str:
        return signal.text


class SpeechTransport:
    def __init__(self, tts_engine=None, asr_engine=None):
        self.tts_engine = tts_engine or LoopbackTextToSpeech()
        self.asr_engine = asr_engine or LoopbackSpeechToText()

    @property
    def description(self):
        return f"{self.tts_engine.name} -> {self.asr_engine.name}"

    def transmit(self, speaker: str, text: str):
        signal = self.tts_engine.synthesize(speaker, text)
        transcript = self.asr_engine.transcribe(signal)
        return signal, transcript


class PatternedSpeechToText:
    def __init__(self, pattern_key="clean", seed=0):
        self.pattern_key = pattern_key
        self.name = f"patterned-asr:{pattern_key}"
        self.rng = random.Random(seed)

    def transcribe(self, signal: SpeechSignal) -> str:
        text = signal.text

        if self.pattern_key == "clean":
            return text

        if self.pattern_key == "hesitant":
            return self._add_hesitations(text)

        if self.pattern_key == "compressed":
            return self._compress(text)

        if self.pattern_key == "noisy_station":
            return self._drop_some_words(text, drop_probability=0.08)

        return text

    def _add_hesitations(self, text: str) -> str:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        out = []
        for sentence in sentences:
            if sentence and self.rng.random() < 0.45:
                out.append(self.rng.choice(("um", "let me see", "okay")) + ", " + sentence)
            else:
                out.append(sentence)
        return " ".join(out)

    def _compress(self, text: str) -> str:
        text = re.sub(r"\bplease\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\bI would\b", "I'd", text)
        return re.sub(r"\s+", " ", text).strip()

    def _drop_some_words(self, text: str, drop_probability: float) -> str:
        words = text.split()
        kept = [
            word
            for word in words
            if self.rng.random() >= drop_probability
        ]
        return " ".join(kept) if kept else text
