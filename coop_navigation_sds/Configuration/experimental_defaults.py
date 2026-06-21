"""Research-oriented defaults shared by GUI, scripts, and batch runs."""

# Speech dialogue systems usually need enough endpointing slack to avoid
# clipping short pauses, while still keeping turn latency measurable.
DEFAULT_ASR_INITIAL_SILENCE_SEC = 4.0
DEFAULT_ASR_BABBLE_TIMEOUT_SEC = 6.0
DEFAULT_ASR_END_SILENCE_MS = 2500
DEFAULT_ASR_AMBIGUOUS_END_SILENCE_MS = 4500
DEFAULT_ASR_BEAM_SIZE = 8

# Default persona-level rates are deliberately clear rather than degraded.
DEFAULT_MAX_UTTERANCE_SEC = 20.0
DEFAULT_MIN_UTTERANCE_SEC = 0.25
DEFAULT_TTS_TIMEOUT_SEC = 60.0
DEFAULT_ASR_TIMEOUT_SEC = 60.0

# Slider ranges expose experimentally useful variation without encouraging
# settings that make the task degenerate for ordinary routing dialogues.
NUMERIC_CONTROL_RANGES = {
    "num_turns": (1, 30, 1),
    "invalid_route_limit": (1, 10, 1),
    "constraint_miss_limit": (1, 10, 1),
    "agent_a_transfer_tolerance": (0, 3, 1),
    "maximum_progressive_constraints": (0, 4, 1),
    "minimum_compared_routes": (1, 6, 1),
    "acceptable_duration_ratio": (1.0, 2.0, 0.05),
    "minimum_stage_suboptimal_options": (0, 5, 1),
    "max_turn_elapsed_sec": (1.0, 20.0, 0.5),
    "calculation_max_time_sec": (1.0, 20.0, 0.5),
    "model_timeout_sec": (5.0, 300.0, 5.0),
    "model_max_new_tokens": (32, 384, 8),
    "agent_a_seed": (0, 9999, 1),
    "agent_b_seed": (0, 9999, 1),
    "agent_a_temperature": (0.01, 1.0, 0.05),
    "agent_b_temperature": (0.01, 1.0, 0.05),
    "agent_a_top_p": (0.05, 1.0, 0.05),
    "agent_b_top_p": (0.05, 1.0, 0.05),
    "asr_beam_size": (1, 16, 1),
    "asr_end_silence_ms": (500, 6000, 100),
    "asr_ambiguous_end_silence_ms": (1000, 8000, 100),
}


def numeric_range(key, fallback):
    """Return a sane GUI slider range for a numeric setting."""
    return NUMERIC_CONTROL_RANGES.get(key, fallback)
