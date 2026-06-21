"""Run CoopNavigationSDS from explicit values without a startup dialog."""
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coop_navigation_sds.app import (
    ConsoleEventSink,
    conversation_worker,
    default_run_config,
    validate_run_config_for_start,
)
from coop_navigation_sds.Configuration.settings import load_run_settings


SCRIPT_CONFIG = {
    "agent_a_type": "staged",
    "agent_b_plugin": "simple",
    "test_case_key": "morning_peak_cross_city",
    "persona_key": "focused_commuter",
    "num_turns": 5,
    "speech_pattern_key": "clean",
    "tts_engine": "file",
    "asr_engine": "file",
    "asr_initial_silence_sec": 4.0,
    "asr_babble_timeout_sec": 6.0,
    "asr_end_silence_ms": 2500,
    "asr_ambiguous_end_silence_ms": 4500,
    "asr_domain_normalization_enabled": True,
    "asr_domain_similarity_threshold": 0.86,
    "clarification_max_attempts": 2,
    "agent_a_ticket_modes": "metro,tram",
    "agent_a_max_walking_min": 10,
    "agent_a_max_delay_risk": "high",
    "agent_a_max_transfer_risk": "medium",
    "network_seed": 42,
    "speech_playback_enabled": False,
    "speech_realtime_enabled": False,
    "agent_a_audio_persona": "high_clarity_caller",
    "agent_b_audio_persona": "high_clarity_operator",
}


def main():
    run_config = load_run_settings(default_run_config())
    run_config.update(SCRIPT_CONFIG)
    run_config = validate_run_config_for_start(run_config)
    conversation_worker(ConsoleEventSink(), model_adapter=None, run_config=run_config)


if __name__ == "__main__":
    main()
