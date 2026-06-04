"""Agent B configuration and speech simulation constants."""
import json
import os

AGENT_B_PLUGIN = os.environ.get("MINILLAMA_AGENT_B_PLUGIN", "minillama").strip() or "minillama"
DEFAULT_SPEECH_PATTERN = os.environ.get("MINILLAMA_DEFAULT_SPEECH_PATTERN", "clean").strip() or "clean"
RUN_MODE = os.environ.get("MINILLAMA_RUN_MODE", "speech").strip().lower() or "speech"
SPEECH_ENGINE = os.environ.get("MINILLAMA_SPEECH_ENGINE", "file").strip().lower() or "file"
SPEECH_TTS_ENGINE = os.environ.get("MINILLAMA_TTS_ENGINE", "").strip().lower()
SPEECH_ASR_ENGINE = os.environ.get("MINILLAMA_ASR_ENGINE", "").strip().lower()
SPEECH_AUDIO_DIR = os.environ.get("MINILLAMA_SPEECH_AUDIO_DIR", "results")
SPEECH_INCOMING_ENABLED = os.environ.get("MINILLAMA_SPEECH_INCOMING", "1").lower() in {"1", "true", "on", "yes"}
SPEECH_OUTGOING_ENABLED = os.environ.get("MINILLAMA_SPEECH_OUTGOING", "1").lower() in {"1", "true", "on", "yes"}
SPEECH_SCOPE = os.environ.get("MINILLAMA_SPEECH_SCOPE", "both").lower()
SPEECH_PLAYBACK_ENABLED = os.environ.get("MINILLAMA_SPEECH_PLAYBACK", "1").lower() in {"1", "true", "on", "yes"}
SPEECH_REALTIME_ENABLED = os.environ.get("MINILLAMA_SPEECH_REALTIME", "1").lower() in {"1", "true", "on", "yes"}
SPEECH_SHOW_RAW_TEXT = os.environ.get("MINILLAMA_SPEECH_SHOW_RAW", "0").lower() in {"1", "true", "on", "yes"}

DEFAULT_SPEECH_PATTERN_PRESETS = {
    "clean": {},
    "mostly_clean": {
        "filler_probability": 0.08,
        "filler_tokens": ["okay"],
    },
    "hesitant": {
        "hesitation_probability": 0.45,
        "hesitation_tokens": ["um", "let me see", "okay"],
    },
    "long_pauses": {
        "pause_probability": 0.35,
        "pause_tokens": ["...", "one moment"],
        "duration_multiplier": 1.45,
    },
    "stutter_light": {
        "stutter_probability": 0.10,
        "stutter_max_words": 2,
    },
    "stutter_heavy": {
        "stutter_probability": 0.24,
        "stutter_max_words": 4,
        "filler_probability": 0.18,
        "filler_tokens": ["uh", "sorry"],
    },
    "filler_words": {
        "filler_probability": 0.28,
        "filler_tokens": ["uh", "okay", "right"],
    },
    "compressed": {
        "compression_enabled": True,
    },
    "noisy_station": {
        "drop_probability": 0.08,
    },
    "clipped_words": {
        "drop_probability": 0.16,
        "protected_terms": ["Alpha", "Bravo", "Harbor", "Ring", "Tram", "Bus", "Metro"],
    },
    "misheard_station": {
        "substitution_probability": 0.12,
        "substitutions": {
            "Bravo": "Brava",
            "Harbor": "Habor",
            "Sierra": "Sarah",
            "Golf": "Gulf",
        },
    },
}


def _load_speech_pattern_presets():
    """Load speech pattern presets from environment, falling back to defaults."""
    presets = dict(DEFAULT_SPEECH_PATTERN_PRESETS)
    raw_json = os.environ.get("MINILLAMA_SPEECH_PATTERN_PRESETS_JSON", "").strip()
    json_path = os.environ.get("MINILLAMA_SPEECH_PATTERN_PRESETS_FILE", "").strip()
    try:
        if json_path:
            with open(json_path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                presets.update(loaded)
        if raw_json:
            loaded = json.loads(raw_json)
            if isinstance(loaded, dict):
                presets.update(loaded)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return presets
    return presets


SPEECH_PATTERNS = _load_speech_pattern_presets()


def speech_pattern_keys():
    """Return available configured speech pattern keys."""
    return sorted(SPEECH_PATTERNS)


def speech_pattern_settings(pattern_key):
    """Return configured settings for a speech pattern key."""
    settings = SPEECH_PATTERNS.get(pattern_key)
    if settings is None:
        settings = SPEECH_PATTERNS.get(DEFAULT_SPEECH_PATTERN, {})
    return dict(settings or {})


REPAIR_SIMILARITY_THRESHOLD = 0.92
MAX_REPAIR_ATTEMPTS = 1
