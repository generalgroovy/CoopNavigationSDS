"""Agent B configuration and speech simulation constants."""
import os

AGENT_B_PLUGIN = "llm"
DEFAULT_SPEECH_PATTERN = "clean"
SPEECH_INCOMING_ENABLED = os.environ.get("MINILLAMA_SPEECH_INCOMING", "1").lower() not in {"0", "false", "off", "no"}
SPEECH_OUTGOING_ENABLED = os.environ.get("MINILLAMA_SPEECH_OUTGOING", "1").lower() not in {"0", "false", "off", "no"}
SPEECH_SCOPE = os.environ.get("MINILLAMA_SPEECH_SCOPE", "both").lower()
SPEECH_SHOW_RAW_TEXT = os.environ.get("MINILLAMA_SPEECH_SHOW_RAW", "0").lower() in {"1", "true", "on", "yes"}

SPEECH_PATTERNS = {
    "hesitation_probability": 0.45,
    "hesitation_tokens": ("um", "let me see", "okay"),
    "noisy_station_drop_probability": 0.08,
}

REPAIR_SIMILARITY_THRESHOLD = 0.92
MAX_REPAIR_ATTEMPTS = 1
