"""Agent B configuration and speech simulation constants."""
import os

AGENT_B_PLUGIN = os.environ.get("MINILLAMA_AGENT_B_PLUGIN", "minillama").strip() or "minillama"
DEFAULT_SPEECH_PATTERN = "clean"
SPEECH_ENGINE = os.environ.get("MINILLAMA_SPEECH_ENGINE", "patterned").strip().lower() or "patterned"
SPEECH_TTS_ENGINE = os.environ.get("MINILLAMA_TTS_ENGINE", "").strip().lower()
SPEECH_ASR_ENGINE = os.environ.get("MINILLAMA_ASR_ENGINE", "").strip().lower()
SPEECH_AUDIO_DIR = os.environ.get("MINILLAMA_SPEECH_AUDIO_DIR", "speech_artifacts")
SPEECH_INCOMING_ENABLED = os.environ.get("MINILLAMA_SPEECH_INCOMING", "0").lower() in {"1", "true", "on", "yes"}
SPEECH_OUTGOING_ENABLED = os.environ.get("MINILLAMA_SPEECH_OUTGOING", "0").lower() in {"1", "true", "on", "yes"}
SPEECH_SCOPE = os.environ.get("MINILLAMA_SPEECH_SCOPE", "none").lower()
SPEECH_PLAYBACK_ENABLED = os.environ.get("MINILLAMA_SPEECH_PLAYBACK", "0").lower() in {"1", "true", "on", "yes"}
SPEECH_REALTIME_ENABLED = os.environ.get("MINILLAMA_SPEECH_REALTIME", "0").lower() in {"1", "true", "on", "yes"}
SPEECH_SHOW_RAW_TEXT = os.environ.get("MINILLAMA_SPEECH_SHOW_RAW", "0").lower() in {"1", "true", "on", "yes"}

SPEECH_PATTERNS = {
    "hesitation_probability": 0.45,
    "hesitation_tokens": ("um", "let me see", "okay"),
    "noisy_station_drop_probability": 0.08,
}

REPAIR_SIMILARITY_THRESHOLD = 0.92
MAX_REPAIR_ATTEMPTS = 1
