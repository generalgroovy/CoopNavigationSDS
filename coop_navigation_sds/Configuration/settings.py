"""Load and save script-friendly run settings as JSON."""
import json
import os
from pathlib import Path

from coop_navigation_sds.Configuration.schema import CONFIG_SCHEMA_VERSION
SETTINGS_SCHEMA_VERSION = CONFIG_SCHEMA_VERSION
TRANSIENT_SETTING_KEYS = {"execution_run_dir"}
FUNDAMENTAL_SETTING_KEYS = {
    "test_case_key",
    "persona_key",
    "agent_a_type",
    "agent_b_plugin",
    "agent_a_objective_mode",
    "num_turns",
    "invalid_route_limit",
    "constraint_miss_limit",
    "clarification_max_attempts",
    "dialogue_stagnation_limit",
    "agent_a_transfer_tolerance",
    "agent_a_ticket_modes",
    "agent_a_max_walking_min",
    "agent_a_max_delay_risk",
    "agent_a_max_transfer_risk",
    "network_seed",
    "maximum_progressive_constraints",
    "minimum_compared_routes",
    "require_constraint_retention",
    "acceptable_duration_ratio",
    "minimum_stage_suboptimal_options",
    "require_stage_suboptimal_options",
    "max_turn_elapsed_sec",
    "calculation_max_time_sec",
    "model_provider",
    "model_profile",
    "model_name",
    "model_device",
    "model_base_url",
    "model_timeout_sec",
    "model_max_new_tokens",
    "allow_model_download",
    "model_service_autostart",
    "speech_pattern_key",
    "tts_engine",
    "asr_engine",
    "speech_playback_enabled",
    "speech_realtime_enabled",
    "agent_a_audio_persona",
    "agent_b_audio_persona",
    "agent_a_seed",
    "agent_b_seed",
    "agent_a_temperature",
    "agent_b_temperature",
    "agent_a_top_p",
    "agent_b_top_p",
    "tts_device",
    "tts_model",
    "tts_executable",
    "tts_python_executable",
    "tts_timeout_sec",
    "asr_language",
    "asr_model",
    "asr_device",
    "asr_compute_type",
    "asr_executable",
    "asr_python_executable",
    "asr_vad_model",
    "asr_timeout_sec",
    "asr_beam_size",
    "asr_initial_silence_sec",
    "asr_babble_timeout_sec",
    "asr_end_silence_ms",
    "asr_ambiguous_end_silence_ms",
    "min_utterance_sec",
    "max_utterance_sec",
    "asr_domain_normalization_enabled",
    "asr_domain_similarity_threshold",
    "results_root",
    "gui_font_size",
    "paired_audio_text_runs",
    "console_view",
    "log_profile",
}


def default_settings_path():
    """Return the configured settings file path."""
    return Path(
        os.environ.get(
            "COOP_NAVIGATION_SDS_SETTINGS_FILE",
            os.environ.get("MINILLAMA_SETTINGS_FILE", "run_settings.json"),
        )
    )


def load_run_settings(defaults=None, path=None):
    """Merge saved JSON settings over supplied defaults.

    Both ``{"schema_version": 1, "config": {...}}`` and a plain JSON object
    are accepted so shell scripts can generate settings with minimal ceremony.
    """
    merged = dict(defaults or {})
    settings_path = Path(path) if path is not None else default_settings_path()
    if not settings_path.exists():
        return merged
    try:
        document = json.loads(settings_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return merged
    if not isinstance(document, dict):
        return merged
    saved = document.get("config", document)
    if not isinstance(saved, dict):
        return merged
    saved = dict(saved)
    if "results_root" not in saved and saved.get("protocol_log_dir"):
        saved["results_root"] = saved.pop("protocol_log_dir")
    if int(document.get("schema_version", 0) or 0) < 2 and saved.get("model_provider") == "ollama":
        if saved.get("model_name") == "llama3.2:3b":
            saved["model_name"] = "llama3.2:latest"
        if float(saved.get("model_timeout_sec", 0) or 0) <= 5.0:
            saved["model_timeout_sec"] = 180.0
    if int(document.get("schema_version", 0) or 0) < 3:
        if int(saved.get("asr_end_silence_ms", 0) or 0) <= 1500:
            saved["asr_end_silence_ms"] = 2500
        if int(saved.get("asr_ambiguous_end_silence_ms", 0) or 0) <= 2400:
            saved["asr_ambiguous_end_silence_ms"] = 4500
        if float(saved.get("max_utterance_sec", 0) or 0) <= 12.0:
            saved["max_utterance_sec"] = 20.0
    merged.update(saved)
    return merged


def save_run_settings(config, path=None):
    """Atomically save persistent run settings and return their path."""
    settings_path = Path(path) if path is not None else default_settings_path()
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    persistent = {}
    for key, value in dict(config).items():
        if key not in FUNDAMENTAL_SETTING_KEYS or key in TRANSIENT_SETTING_KEYS:
            continue
        persistent[key] = value
    document = {
        "schema_version": SETTINGS_SCHEMA_VERSION,
        "config": persistent,
    }
    temporary_path = settings_path.with_suffix(f"{settings_path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(settings_path)
    return settings_path
