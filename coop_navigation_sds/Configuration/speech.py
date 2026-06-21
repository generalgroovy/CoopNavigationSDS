"""Agent B configuration and speech simulation constants."""
import json
import os


def _environment(name, legacy_name, default):
    return os.environ.get(name, os.environ.get(legacy_name, default))


AGENT_B_PLUGIN = _environment("COOP_NAVIGATION_SDS_AGENT_B_PLUGIN", "MINILLAMA_AGENT_B_PLUGIN", "llm").strip() or "llm"
DEFAULT_SPEECH_PATTERN = _environment("COOP_NAVIGATION_SDS_DEFAULT_SPEECH_PATTERN", "MINILLAMA_DEFAULT_SPEECH_PATTERN", "clean").strip() or "clean"
SPEECH_TTS_ENGINE = _environment("COOP_NAVIGATION_SDS_TTS_ENGINE", "MINILLAMA_TTS_ENGINE", "").strip().lower()
SPEECH_ASR_ENGINE = _environment("COOP_NAVIGATION_SDS_ASR_ENGINE", "MINILLAMA_ASR_ENGINE", "").strip().lower()
SPEECH_AUDIO_DIR = _environment("COOP_NAVIGATION_SDS_SPEECH_AUDIO_DIR", "MINILLAMA_SPEECH_AUDIO_DIR", "results")
SPEECH_PLAYBACK_ENABLED = _environment("COOP_NAVIGATION_SDS_SPEECH_PLAYBACK", "MINILLAMA_SPEECH_PLAYBACK", "1").lower() in {"1", "true", "on", "yes"}
SPEECH_REALTIME_ENABLED = _environment("COOP_NAVIGATION_SDS_SPEECH_REALTIME", "MINILLAMA_SPEECH_REALTIME", "1").lower() in {"1", "true", "on", "yes"}
SPEECH_SHOW_RAW_TEXT = _environment("COOP_NAVIGATION_SDS_SPEECH_SHOW_RAW", "MINILLAMA_SPEECH_SHOW_RAW", "0").lower() in {"1", "true", "on", "yes"}

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
        "protected_terms": ["Alpha", "Bravo", "Harbor", "M1", "T1", "B1", "Tram", "Bus", "Metro"],
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
    raw_json = _environment("COOP_NAVIGATION_SDS_SPEECH_PATTERN_PRESETS_JSON", "MINILLAMA_SPEECH_PATTERN_PRESETS_JSON", "").strip()
    json_path = _environment("COOP_NAVIGATION_SDS_SPEECH_PATTERN_PRESETS_FILE", "MINILLAMA_SPEECH_PATTERN_PRESETS_FILE", "").strip()
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
MAX_REPAIR_ATTEMPTS = 2
