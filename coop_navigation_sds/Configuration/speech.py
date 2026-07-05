"""Agent B configuration and speech simulation constants."""
import json
import os

from coop_navigation_sds.Configuration.runtime import RESULTS_DIR


def _environment(name, legacy_name, default):
    return os.environ.get(name, os.environ.get(legacy_name, default))


AGENT_B_PLUGIN = _environment("COOP_NAVIGATION_SDS_AGENT_B_PLUGIN", "MINILLAMA_AGENT_B_PLUGIN", "llm").strip() or "llm"
DEFAULT_SPEECH_PATTERN = _environment("COOP_NAVIGATION_SDS_DEFAULT_SPEECH_PATTERN", "MINILLAMA_DEFAULT_SPEECH_PATTERN", "clean").strip() or "clean"
SPEECH_TTS_ENGINE = _environment("COOP_NAVIGATION_SDS_TTS_ENGINE", "MINILLAMA_TTS_ENGINE", "").strip().lower()
SPEECH_ASR_ENGINE = _environment("COOP_NAVIGATION_SDS_ASR_ENGINE", "MINILLAMA_ASR_ENGINE", "").strip().lower()
SPEECH_AUDIO_DIR = RESULTS_DIR
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
    "severe_channel": {
        "hesitation_probability": 0.20,
        "hesitation_tokens": ["um", "sorry"],
        "stutter_probability": 0.18,
        "stutter_max_words": 3,
        "filler_probability": 0.16,
        "filler_tokens": ["uh", "okay"],
    },
}


# Ordered treatments vary only the speech channel while task and model factors
# remain fixed. Values are selected a priori; observed outcome separation is a
# result, never a condition for retaining a run.
SPEECH_PERFORMANCE_PROFILES = {
    "ceiling": {
        "speech_performance_profile_key": "ceiling",
        "speech_performance_band": "ceiling",
        "speech_performance_rank": 3,
        "agent_a_audio_persona": "high_clarity_caller",
        "agent_b_audio_persona": "high_clarity_operator",
        "speech_pattern_key": "clean",
        "agent_a_custom_audio": False,
        "agent_b_custom_audio": False,
        "asr_beam_size": 16,
        "asr_domain_normalization_enabled": True,
        "asr_domain_similarity_threshold": 0.80,
        "channel_noise_snr_db": None,
        "channel_gain_db": 0.0,
        "channel_clip_threshold": 1.0,
        "channel_dropout_rate": 0.0,
        "max_utterance_sec": 30.0,
    },
    "nominal": {
        "speech_performance_profile_key": "nominal",
        "speech_performance_band": "nominal",
        "speech_performance_rank": 2,
        "agent_a_audio_persona": "neutral_caller",
        "agent_b_audio_persona": "clear_operator",
        "speech_pattern_key": "mostly_clean",
        "agent_a_custom_audio": False,
        "agent_b_custom_audio": False,
        "asr_beam_size": 11,
        "asr_domain_normalization_enabled": True,
        "asr_domain_similarity_threshold": 0.86,
        "channel_noise_snr_db": 30.0,
        "channel_gain_db": -2.0,
        "channel_clip_threshold": 0.95,
        "channel_dropout_rate": 0.005,
        "max_utterance_sec": 22.0,
    },
    "challenging": {
        "speech_performance_profile_key": "challenging",
        "speech_performance_band": "challenging",
        "speech_performance_rank": 1,
        "agent_a_audio_persona": "degraded_caller",
        "agent_b_audio_persona": "degraded_operator",
        "speech_pattern_key": "hesitant",
        "agent_a_custom_audio": False,
        "agent_b_custom_audio": False,
        "asr_beam_size": 6,
        "asr_domain_normalization_enabled": True,
        "asr_domain_similarity_threshold": 0.92,
        "channel_noise_snr_db": 15.0,
        "channel_gain_db": -6.0,
        "channel_clip_threshold": 0.70,
        "channel_dropout_rate": 0.04,
        "max_utterance_sec": 16.0,
    },
    "floor": {
        "speech_performance_profile_key": "floor",
        "speech_performance_band": "floor",
        "speech_performance_rank": 0,
        "agent_a_audio_persona": "barely_understandable_caller",
        "agent_b_audio_persona": "barely_understandable_operator",
        "speech_pattern_key": "severe_channel",
        "agent_a_custom_audio": False,
        "agent_b_custom_audio": False,
        "asr_beam_size": 1,
        "asr_domain_normalization_enabled": False,
        "asr_domain_similarity_threshold": 0.98,
        "channel_noise_snr_db": 5.0,
        "channel_gain_db": -12.0,
        "channel_clip_threshold": 0.35,
        "channel_dropout_rate": 0.15,
        "max_utterance_sec": 10.0,
    },
}


def speech_performance_profile(key):
    """Return one validated floor-to-ceiling speech treatment."""
    normalized = str(key or "").strip().lower()
    if normalized not in SPEECH_PERFORMANCE_PROFILES:
        raise ValueError(
            f"Unknown speech performance band '{key}'. Expected one of: "
            f"{', '.join(SPEECH_PERFORMANCE_PROFILES)}."
        )
    return dict(SPEECH_PERFORMANCE_PROFILES[normalized])


def speech_performance_profiles(keys=None):
    """Return ordered named speech treatments for job expansion."""
    selected = tuple(keys or SPEECH_PERFORMANCE_PROFILES)
    return [speech_performance_profile(key) for key in selected]


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
