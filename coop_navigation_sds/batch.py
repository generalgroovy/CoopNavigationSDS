"""Batch experiment entry point for automatic evaluation of speech-dialog conditions."""
import argparse
from pathlib import Path

from coop_navigation_sds.Configuration.speech import AGENT_B_PLUGIN
from coop_navigation_sds.Configuration.speech import DEFAULT_SPEECH_PATTERN
from coop_navigation_sds.Configuration.speech import SPEECH_ASR_ENGINE, SPEECH_AUDIO_DIR, SPEECH_PLAYBACK_ENABLED, SPEECH_REALTIME_ENABLED, SPEECH_TTS_ENGINE, speech_pattern_keys
from coop_navigation_sds.Configuration.experimental_defaults import (
    DEFAULT_ASR_AMBIGUOUS_END_SILENCE_MS,
    DEFAULT_ASR_BABBLE_TIMEOUT_SEC,
    DEFAULT_ASR_BEAM_SIZE,
    DEFAULT_ASR_END_SILENCE_MS,
    DEFAULT_ASR_INITIAL_SILENCE_SEC,
    DEFAULT_ASR_TIMEOUT_SEC,
    DEFAULT_TTS_TIMEOUT_SEC,
)
from coop_navigation_sds.NaturalLanguageGeneration.assistant.plugin_registry import AgentBPluginConfig
from coop_navigation_sds.NaturalLanguageGeneration.caller.config import PERSONAS
from coop_navigation_sds.NaturalLanguageGeneration.caller.responder import (
    AGENT_A_MINILLAMA,
    AGENT_A_USERLM,
    available_agent_a_types,
    normalize_agent_a_type,
)
from coop_navigation_sds.Configuration.runtime import (
    AGENT_A_TRANSFER_TOLERANCE,
    CONSTRAINT_MISS_LIMIT,
    INVALID_ROUTE_LIMIT,
    MAXIMUM_PROGRESSIVE_CONSTRAINTS,
    MINIMUM_COMPARED_ROUTES,
    NUM_TURNS,
    REQUIRE_CONSTRAINT_RETENTION,
    RESULTS_DIR,
    SESSION_LOG_PROFILE,
)
from coop_navigation_sds.Configuration.travel import (
    ACCEPTABLE_DURATION_RATIO,
    ALLOW_MODEL_DOWNLOAD,
    CHAT_TIMEOUT_SEC,
    DEVICE,
    GENERATION_MAX_TIME_SEC,
    MAX_INPUT_TOKENS,
    MAX_NEW_TOKENS,
    MIN_STAGE_SUBOPTIMAL_OPTIONS,
    MODEL,
    MODEL_PROVIDER,
    REQUIRE_STAGE_SUBOPTIMAL_OPTIONS,
)
from coop_navigation_sds.NaturalLanguageGeneration.models import available_model_provider_keys
from coop_navigation_sds.DialogManagement.manager import DEFAULT_MAX_TURN_ELAPSED_SEC
from coop_navigation_sds.experiments import ExperimentRunner, build_condition_grid, write_metrics_file
from coop_navigation_sds.EvaluationMetrics.metrics import apply_cross_run_metrics, apply_paired_run_metrics
from coop_navigation_sds.ResultsAndArtifacts.artifacts import create_execution_run_dir, run_scoped_path, write_conversation_protocols, write_experiment_manifest, write_metric_phase_logs, write_network_research_artifacts
from coop_navigation_sds.TransportNetwork.constraints import OBJECTIVE_MODES, OBJECTIVE_SHORTEST_WITH_CONSTRAINTS
from coop_navigation_sds.Configuration.scenarios import DEFAULT_TEST_CASE
from coop_navigation_sds.Configuration.settings import default_settings_path, load_run_settings
from coop_navigation_sds.Configuration.jobs import job_grid_value, job_parameter_grid, load_experiment_job
from coop_navigation_sds.Configuration.pipeline import component_status, serializable_metric_dependency_report
from coop_navigation_sds.Configuration.component_catalog import apply_speech_engine_profiles
from coop_navigation_sds.TransportNetwork.test_cases import TEST_CASES, get_test_case
from coop_navigation_sds.DialogManagement.speech_pipeline import (
    available_asr_engine_keys,
    available_tts_engine_keys,
    platform_default_asr_engine,
    platform_default_tts_engine,
)
from coop_navigation_sds.TextToSpeech.personas import (
    DEFAULT_AGENT_A_AUDIO_PERSONA,
    DEFAULT_AGENT_B_AUDIO_PERSONA,
    audio_persona_keys,
)


def parse_csv_arg(value, all_values=None):
    """Parse a comma-separated CLI argument into a list."""
    if not value:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    if any(item.lower() == "all" for item in items):
        return list(all_values or [])
    return items


def parse_bool_flag(value):
    """Parse CLI booleans."""
    normalized = value.lower()
    if normalized in {"1", "true", "on", "yes"}:
        return True
    if normalized in {"0", "false", "off", "no"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def main():
    """Run the configured experiment grid and write metrics output."""
    settings_parser = argparse.ArgumentParser(add_help=False)
    settings_parser.add_argument("--settings-file", default=str(default_settings_path()))
    settings_parser.add_argument("--job-file")
    settings_parser.add_argument("--preset")
    settings_args, _remaining = settings_parser.parse_known_args()
    if settings_args.preset and not settings_args.job_file:
        preset_name = settings_args.preset
        preset_path = Path(__file__).parent / "Configuration" / "presets" / f"{preset_name}.job"
        if not preset_path.is_file():
            raise SystemExit(f"Unknown preset '{preset_name}'. Expected {preset_path}")
        settings_args.job_file = str(preset_path)
    saved = load_run_settings({}, settings_args.settings_file)
    job = load_experiment_job(settings_args.job_file)
    configured = {**saved, **job.get("config", {})}

    parser = argparse.ArgumentParser()
    parser.add_argument("--settings-file", default=settings_args.settings_file, help="JSON settings file used as batch defaults. Explicit command-line options override it.")
    parser.add_argument("--job-file", default=settings_args.job_file, help="JSON .job file defining batch defaults and experiment grids.")
    parser.add_argument("--preset", default=settings_args.preset, help="Bundled preset name, for example linux_userlm_tinyllama_chattts_faster_whisper.")
    parser.add_argument("--agent-b-plugin", default=configured.get("agent_b_plugin", AGENT_B_PLUGIN), help="Agent B policy: llm, simple, pareto, robust, diverse, or package.module:factory.")
    parser.add_argument("--model-provider", default=configured.get("model_provider", MODEL_PROVIDER), choices=available_model_provider_keys())
    parser.add_argument("--model-name", default=configured.get("model_name", MODEL))
    parser.add_argument("--model-api-key", default=configured.get("model_api_key", ""))
    parser.add_argument("--model-base-url", default=configured.get("model_base_url", ""))
    parser.add_argument("--model-device", default=configured.get("model_device", DEVICE))
    parser.add_argument("--model-timeout-sec", type=float, default=float(configured.get("model_timeout_sec", CHAT_TIMEOUT_SEC)))
    parser.add_argument("--model-max-new-tokens", type=int, default=int(configured.get("model_max_new_tokens", MAX_NEW_TOKENS)))
    parser.add_argument("--model-max-input-tokens", type=int, default=int(configured.get("model_max_input_tokens", MAX_INPUT_TOKENS)))
    parser.add_argument("--test-cases", default=job_grid_value(job, "test_cases", configured.get("test_case_key", "morning_peak_cross_city,midday_transfer,evening_outbound,late_event")))
    parser.add_argument("--personas", default=job_grid_value(job, "personas", configured.get("persona_key", "focused_commuter")))
    parser.add_argument("--agent-a-audio-personas", default=job_grid_value(job, "agent_a_audio_personas", configured.get("agent_a_audio_persona", DEFAULT_AGENT_A_AUDIO_PERSONA)))
    parser.add_argument("--agent-b-audio-personas", default=job_grid_value(job, "agent_b_audio_personas", configured.get("agent_b_audio_persona", DEFAULT_AGENT_B_AUDIO_PERSONA)))
    parser.add_argument("--speech-patterns", default=job_grid_value(job, "speech_patterns", configured.get("speech_pattern_key", DEFAULT_SPEECH_PATTERN)))
    parser.add_argument("--tts-engine", default=configured.get("tts_engine", SPEECH_TTS_ENGINE or platform_default_tts_engine()), choices=available_tts_engine_keys(), help="Text-to-speech implementation.")
    parser.add_argument("--asr-engine", default=configured.get("asr_engine", SPEECH_ASR_ENGINE or platform_default_asr_engine()), choices=available_asr_engine_keys(), help="Automatic speech recognition implementation.")
    parser.add_argument("--tts-engines", default=job_grid_value(job, "tts_engines", configured.get("tts_engine", SPEECH_TTS_ENGINE or platform_default_tts_engine())), help="Comma-separated TTS engine grid.")
    parser.add_argument("--asr-engines", default=job_grid_value(job, "asr_engines", configured.get("asr_engine", SPEECH_ASR_ENGINE or platform_default_asr_engine())), help="Comma-separated ASR engine grid.")
    parser.add_argument("--agent-b-models", default=job_grid_value(job, "agent_b_models", configured.get("model_name", MODEL)), help="Comma-separated Agent B model grid.")
    parser.add_argument("--paired-audio-text", type=parse_bool_flag, default=bool(configured.get("paired_audio_text_runs", True)), help="Generate a deterministic text-only control for every audio condition.")
    parser.add_argument("--speech-audio-dir", default=configured.get("speech_audio_dir", SPEECH_AUDIO_DIR or None), help="Directory for generated speech wave files and transcript artifacts.")
    parser.add_argument("--speech-playback", type=parse_bool_flag, default=configured.get("speech_playback_enabled", SPEECH_PLAYBACK_ENABLED), help="Play generated wave files during file-engine runs.")
    parser.add_argument("--no-speech-playback", dest="speech_playback", action="store_false", default=argparse.SUPPRESS, help="Disable generated-audio playback.")
    parser.add_argument("--speech-real-time", type=parse_bool_flag, default=configured.get("speech_realtime_enabled", SPEECH_REALTIME_ENABLED), help="Wait for each spoken turn before automatic speech recognition transcript delivery.")
    parser.add_argument("--no-speech-real-time", dest="speech_real_time", action="store_false", default=argparse.SUPPRESS, help="Do not wait for real-time playback in batch runs.")
    parser.add_argument("--tts-device", default=configured.get("tts_device", "auto"))
    parser.add_argument("--tts-model", default=configured.get("tts_model", ""), help="Text-to-speech model identifier or local voice path.")
    parser.add_argument("--tts-executable", default=configured.get("tts_executable", ""), help="Optional text-to-speech command path.")
    parser.add_argument("--tts-python-executable", default=configured.get("tts_python_executable", ""), help="Optional isolated text-to-speech Python interpreter.")
    parser.add_argument("--tts-timeout-sec", type=float, default=float(configured.get("tts_timeout_sec", DEFAULT_TTS_TIMEOUT_SEC)))
    parser.add_argument("--asr-language", default=configured.get("asr_language", "en-US"))
    parser.add_argument("--asr-model", default=configured.get("asr_model", "small.en"))
    parser.add_argument("--asr-device", default=configured.get("asr_device", "auto"))
    parser.add_argument("--asr-compute-type", default=configured.get("asr_compute_type", "default"))
    parser.add_argument("--asr-executable", default=configured.get("asr_executable", ""), help="Optional whisper.cpp executable path.")
    parser.add_argument("--asr-python-executable", default=configured.get("asr_python_executable", ""), help="Optional isolated recognition Python interpreter.")
    parser.add_argument("--asr-vad-model", default=configured.get("asr_vad_model", ""), help="Optional whisper.cpp voice-activity model path.")
    parser.add_argument("--asr-timeout-sec", type=float, default=float(configured.get("asr_timeout_sec", DEFAULT_ASR_TIMEOUT_SEC)))
    parser.add_argument("--asr-beam-size", type=int, default=int(configured.get("asr_beam_size", DEFAULT_ASR_BEAM_SIZE)))
    parser.add_argument("--asr-initial-silence-sec", type=float, default=float(configured.get("asr_initial_silence_sec", DEFAULT_ASR_INITIAL_SILENCE_SEC)))
    parser.add_argument("--asr-babble-timeout-sec", type=float, default=float(configured.get("asr_babble_timeout_sec", DEFAULT_ASR_BABBLE_TIMEOUT_SEC)))
    parser.add_argument("--asr-end-silence-ms", type=int, default=int(configured.get("asr_end_silence_ms", DEFAULT_ASR_END_SILENCE_MS)))
    parser.add_argument("--asr-ambiguous-end-silence-ms", type=int, default=int(configured.get("asr_ambiguous_end_silence_ms", DEFAULT_ASR_AMBIGUOUS_END_SILENCE_MS)))
    parser.add_argument("--asr-domain-normalization", type=parse_bool_flag, default=bool(configured.get("asr_domain_normalization_enabled", True)), help="Repair close transit-domain terms while preserving raw recognition output.")
    parser.add_argument("--asr-domain-similarity-threshold", type=float, default=float(configured.get("asr_domain_similarity_threshold", 0.86)), help="Similarity threshold from 0.70 to 1.00 for domain-term repair.")
    parser.add_argument("--provider-environment-dir", default=configured.get("provider_environment_dir", ".speech-providers"), help="Directory containing the speech-provider manifest and isolated environments.")
    for agent, label in (("agent_a", "Agent A"), ("agent_b", "Agent B")):
        option = agent.replace("_", "-")
        parser.add_argument(
            f"--{option}-temperature",
            type=float,
            default=float(configured.get(f"{agent}_temperature", 0.3)),
            help=f"{label} ChatTTS variation; ignored by other synthesizers.",
        )
        parser.add_argument(
            f"--{option}-top-p",
            type=float,
            default=float(configured.get(f"{agent}_top_p", 0.7)),
            help=f"{label} ChatTTS sampling range; ignored by other synthesizers.",
        )
        parser.add_argument(
            f"--{option}-seed",
            type=int,
            default=int(configured.get(f"{agent}_seed", 11 if agent == "agent_a" else 29)),
            help=f"{label} ChatTTS reproducibility seed; ignored by other synthesizers.",
        )
    parser.add_argument("--model-params", default=job_grid_value(job, "model_params", "greedy"))
    parser.add_argument("--objective-modes", default=job_grid_value(job, "objective_modes", configured.get("agent_a_objective_mode", OBJECTIVE_SHORTEST_WITH_CONSTRAINTS)), help="Comma-separated Agent A objective modes or all.")
    configured_agent_a_type = normalize_agent_a_type(
        configured.get("agent_a_type"),
        configured.get("llm_agent_a", False),
    )
    parser.add_argument("--agent-a-type", default=configured_agent_a_type, choices=available_agent_a_types(), help="Agent A implementation: staged or userlm.")
    parser.add_argument("--llm-agent-a", action=argparse.BooleanOptionalAction, default=None, help="Deprecated compatibility alias; --llm-agent-a selects UserLM.")
    parser.add_argument("--iterations", type=int, default=int(job.get("iterations", 1)))
    parser.add_argument("--num-turns", type=int, default=int(configured.get("num_turns", NUM_TURNS)))
    parser.add_argument("--invalid-route-limit", type=int, default=int(configured.get("invalid_route_limit", INVALID_ROUTE_LIMIT)))
    parser.add_argument("--constraint-miss-limit", type=int, default=int(configured.get("constraint_miss_limit", CONSTRAINT_MISS_LIMIT)))
    parser.add_argument("--clarification-max-attempts", type=int, default=int(configured.get("clarification_max_attempts", 2)), help="Targeted repair turns before a structured trip-detail reset.")
    parser.add_argument("--agent-a-transfer-tolerance", type=int, default=int(configured.get("agent_a_transfer_tolerance", AGENT_A_TRANSFER_TOLERANCE)), choices=(0, 1, 2), help="Extra line changes Agent A accepts over the constraint-aware startup route.")
    parser.add_argument("--agent-a-ticket-modes", default=configured.get("agent_a_ticket_modes", "metro,tram"), choices=("metro,tram", "metro,bus", "tram,bus"), help="Exactly two public transport tickets available to Agent A.")
    parser.add_argument("--agent-a-max-walking-min", type=int, default=int(configured.get("agent_a_max_walking_min", 10)), help="Maximum cumulative walking minutes accepted by Agent A.")
    parser.add_argument("--agent-a-max-delay-risk", default=configured.get("agent_a_max_delay_risk", "high"), choices=("low", "medium", "high"), help="Highest acceptable whole-route delay class.")
    parser.add_argument("--agent-a-max-transfer-risk", default=configured.get("agent_a_max_transfer_risk", "medium"), choices=("low", "medium", "high"), help="Highest acceptable missed-connection risk class.")
    parser.add_argument("--network-seed", type=int, default=int(configured.get("network_seed", 42)), help="Reproducible multimodal network seed; may also be varied with a job parameter range.")
    parser.add_argument("--maximum-progressive-constraints", type=int, default=int(configured.get("maximum_progressive_constraints", MAXIMUM_PROGRESSIVE_CONSTRAINTS)))
    parser.add_argument("--minimum-compared-routes", type=int, default=int(configured.get("minimum_compared_routes", MINIMUM_COMPARED_ROUTES)))
    parser.add_argument("--require-constraint-retention", type=parse_bool_flag, default=bool(configured.get("require_constraint_retention", REQUIRE_CONSTRAINT_RETENTION)))
    parser.add_argument("--acceptable-duration-ratio", type=float, default=float(configured.get("acceptable_duration_ratio", ACCEPTABLE_DURATION_RATIO)))
    parser.add_argument("--minimum-stage-suboptimal-options", type=int, default=int(configured.get("minimum_stage_suboptimal_options", MIN_STAGE_SUBOPTIMAL_OPTIONS)))
    parser.add_argument("--require-stage-suboptimal-options", type=parse_bool_flag, default=bool(configured.get("require_stage_suboptimal_options", REQUIRE_STAGE_SUBOPTIMAL_OPTIONS)))
    parser.add_argument("--max-turn-elapsed-sec", type=float, default=float(configured.get("max_turn_elapsed_sec", DEFAULT_MAX_TURN_ELAPSED_SEC)))
    parser.add_argument("--calculation-max-time-sec", type=float, default=float(configured.get("calculation_max_time_sec", GENERATION_MAX_TIME_SEC)))
    parser.add_argument("--output", default="automatic_eval_metrics.xlsx")
    parser.add_argument("--metrics-log-dir", default=None, help="Directory for per-phase metric JSONL files.")
    parser.add_argument("--results-dir", "--research-log-dir", dest="results_dir", default=configured.get("protocol_log_dir", RESULTS_DIR), help="Root results directory. Each execution creates one run folder inside it.")
    parser.add_argument("--protocol-log-dir", default=None, help="Directory for detailed conversation protocol artifacts. Empty writes directly inside the execution run folder.")
    parser.add_argument("--network-picture-dir", default=None, help="Directory for generated network graph SVG. Empty writes directly inside the execution run folder.")
    parser.add_argument("--log-profile", default=SESSION_LOG_PROFILE, choices=("off", "startup", "runtime", "full"), help="Structured batch logging level.")
    parser.add_argument("--log-dir", default=None, help="Directory for optional batch JSONL/session logs. Empty writes directly inside the execution run folder.")
    parser.add_argument("--progress", action="store_true", help="Print each completed condition id.")
    args = parser.parse_args()
    if args.llm_agent_a is not None:
        args.agent_a_type = AGENT_A_USERLM if args.llm_agent_a else AGENT_A_MINILLAMA

    test_case_keys = parse_csv_arg(args.test_cases, TEST_CASES)
    conditions = list(build_condition_grid(
        test_case_keys=test_case_keys,
        persona_keys=parse_csv_arg(args.personas, PERSONAS),
        speech_pattern_keys=parse_csv_arg(args.speech_patterns, speech_pattern_keys()),
        model_param_keys=parse_csv_arg(args.model_params),
        objective_modes=parse_csv_arg(args.objective_modes, OBJECTIVE_MODES),
        agent_a_audio_persona_keys=parse_csv_arg(args.agent_a_audio_personas, audio_persona_keys("caller")),
        agent_b_audio_persona_keys=parse_csv_arg(args.agent_b_audio_personas, audio_persona_keys("assistant")),
        tts_engine_keys=parse_csv_arg(args.tts_engines, available_tts_engine_keys()),
        asr_engine_keys=parse_csv_arg(args.asr_engines, available_asr_engine_keys()),
        agent_b_model_keys=parse_csv_arg(args.agent_b_models),
        iterations=args.iterations,
        parameter_grid=job_parameter_grid(job),
        pair_audio_with_text=args.paired_audio_text,
    ))
    preflight_config = {
        "tts_model": args.tts_model,
        "tts_executable": args.tts_executable,
        "asr_model": args.asr_model,
        "asr_executable": args.asr_executable,
    }
    preflight_failures = []
    for kind, engines in (
        ("tts", sorted({condition.tts_engine or args.tts_engine for condition in conditions if condition.run_type == "audio_variant"})),
        ("asr", sorted({condition.asr_engine or args.asr_engine for condition in conditions if condition.run_type == "audio_variant"})),
    ):
        for engine in engines:
            engine_config = apply_speech_engine_profiles(
                {**preflight_config, f"{kind}_engine": engine},
                replace=True,
            )
            status = component_status(kind, engine, engine_config)
            if not status.available:
                preflight_failures.append(f"{kind.upper()} {engine}: {status.reason}")
    if preflight_failures:
        raise SystemExit("Preflight failed:\n  " + "\n  ".join(preflight_failures))
    run_dir = create_execution_run_dir(
        args.results_dir,
        label=f"batch_{len(conditions)}_conditions",
    )
    if args.job_file:
        (run_dir / "experiment_job.json").write_text(
            Path(args.job_file).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    metrics_output = run_scoped_path(run_dir, args.output, "automatic_eval_metrics.xlsx")
    protocol_log_dir = run_scoped_path(run_dir, args.protocol_log_dir, "")
    phase_log_dir = run_scoped_path(run_dir, args.metrics_log_dir, "")
    network_picture_dir = run_scoped_path(run_dir, args.network_picture_dir, "")
    configured_speech_audio_dir = None if args.speech_audio_dir == SPEECH_AUDIO_DIR else args.speech_audio_dir
    speech_audio_dir = run_scoped_path(run_dir, configured_speech_audio_dir, "")
    session_log_dir = run_scoped_path(run_dir, args.log_dir, "")

    agent_b_config = AgentBPluginConfig(args.agent_b_plugin)
    model_adapter = None
    model_adapter_factory = None
    if agent_b_config.needs_model or args.agent_a_type == AGENT_A_USERLM:
        if args.model_provider == "openai_compatible" and not args.model_api_key:
            raise SystemExit(
                "ChatGPT/OpenAI-compatible batch runs require an API key. "
                "Pass --model-api-key or set OPENAI_API_KEY."
            )
        try:
            from coop_navigation_sds.NaturalLanguageGeneration.model_runtime import create_model_adapter

            model_adapter = create_model_adapter(
                args.model_provider,
                model_name=args.model_name,
                api_key=args.model_api_key,
                base_url=args.model_base_url,
                timeout_sec=args.model_timeout_sec,
                device=args.model_device,
                max_new_tokens=args.model_max_new_tokens,
                max_input_tokens=args.model_max_input_tokens,
                allow_model_download=False,
            )
            def model_adapter_factory(model_name):
                return create_model_adapter(
                    args.model_provider,
                    model_name=model_name,
                    api_key=args.model_api_key,
                    base_url=args.model_base_url,
                    timeout_sec=args.model_timeout_sec,
                    device=args.model_device,
                    max_new_tokens=args.model_max_new_tokens,
                    max_input_tokens=args.model_max_input_tokens,
                    allow_model_download=False,
                )
        except Exception as exc:
            raise SystemExit(f"Preflight failed: Agent model assets are unavailable: {exc}") from exc
    runner = ExperimentRunner(
        model_adapter,
        args.num_turns,
        agent_b_plugin_key=args.agent_b_plugin,
        tts_engine=args.tts_engine,
        asr_engine=args.asr_engine,
        speech_audio_dir=str(speech_audio_dir),
        speech_playback_enabled=args.speech_playback,
        speech_realtime_enabled=args.speech_real_time,
        speech_synthesis_config={
            "tts_device": args.tts_device,
            "tts_model": args.tts_model,
            "tts_executable": args.tts_executable,
            "tts_python_executable": args.tts_python_executable,
            "tts_timeout_sec": args.tts_timeout_sec,
            "asr_language": args.asr_language,
            "asr_model": args.asr_model,
            "asr_device": args.asr_device,
            "asr_compute_type": args.asr_compute_type,
            "asr_executable": args.asr_executable,
            "asr_python_executable": args.asr_python_executable,
            "asr_vad_model": args.asr_vad_model,
            "asr_timeout_sec": args.asr_timeout_sec,
            "asr_beam_size": args.asr_beam_size,
            "asr_initial_silence_sec": args.asr_initial_silence_sec,
            "asr_babble_timeout_sec": args.asr_babble_timeout_sec,
            "asr_end_silence_ms": args.asr_end_silence_ms,
            "asr_ambiguous_end_silence_ms": args.asr_ambiguous_end_silence_ms,
            "asr_domain_normalization_enabled": args.asr_domain_normalization,
            "asr_domain_similarity_threshold": args.asr_domain_similarity_threshold,
            "provider_environment_dir": args.provider_environment_dir,
            **{
                key: getattr(args, key)
                for agent in ("agent_a", "agent_b")
                for key in (
                    f"{agent}_temperature",
                    f"{agent}_top_p",
                    f"{agent}_seed",
                )
            },
        },
        transfer_tolerance=args.agent_a_transfer_tolerance,
        invalid_route_limit=args.invalid_route_limit,
        constraint_miss_limit=args.constraint_miss_limit,
        max_turn_elapsed_sec=args.max_turn_elapsed_sec,
        calculation_max_time_sec=args.calculation_max_time_sec,
        agent_a_type=args.agent_a_type,
        log_profile=args.log_profile,
        log_dir=str(session_log_dir),
        metric_config=configured.get("metric_config"),
        metric_tiers=configured.get("metric_tiers"),
        scenario_overrides={
            "network_seed": args.network_seed,
            "clarification_max_attempts": args.clarification_max_attempts,
            "ticket_modes": tuple(args.agent_a_ticket_modes.split(",")),
            "max_walking_min": max(0, args.agent_a_max_walking_min),
            "max_delay_probability": {"low": 0.24, "medium": 0.44, "high": 1.0}[args.agent_a_max_delay_risk],
            "max_transfer_miss_probability": {"low": 0.24, "medium": 0.44, "high": 1.0}[args.agent_a_max_transfer_risk],
            "maximum_progressive_constraints": args.maximum_progressive_constraints,
            "minimum_compared_routes": args.minimum_compared_routes,
            "require_constraint_retention": args.require_constraint_retention,
            "acceptable_duration_ratio": args.acceptable_duration_ratio,
            "min_stage_suboptimal_options": args.minimum_stage_suboptimal_options,
            "require_stage_suboptimal_options": args.require_stage_suboptimal_options,
            "agent_a_ticket_modes": args.agent_a_ticket_modes,
            "agent_a_max_walking_min": args.agent_a_max_walking_min,
            "agent_a_max_delay_risk": args.agent_a_max_delay_risk,
            "agent_a_max_transfer_risk": args.agent_a_max_transfer_risk,
            "network_seed": args.network_seed,
            "parameter_grid": job_parameter_grid(job),
        },
        model_adapter_factory=model_adapter_factory,
    )
    results = []
    metrics = []
    for condition in conditions:
        runner.speech_audio_dir = str(speech_audio_dir)
        result, metric = runner.run_condition(condition)
        results.append(result)
        metrics.append(metric)
        if args.progress:
            print(f"completed {condition.condition_id}", flush=True)
    apply_cross_run_metrics(metrics)
    apply_paired_run_metrics(metrics)
    write_metrics_file(metrics, metrics_output)
    protocol_paths = write_conversation_protocols(results, protocol_log_dir)

    first_case_key = (test_case_keys or [DEFAULT_TEST_CASE])[0]
    first_case = get_test_case(first_case_key)
    artifacts = write_network_research_artifacts(
        first_case.scenario["start_time_min"],
        run_dir,
        picture_dir=network_picture_dir,
    )
    manifest_path = write_experiment_manifest(
        conditions,
        run_dir,
        num_turns=args.num_turns,
        speech_engine="full_speech",
        tts_engine=args.tts_engine,
        asr_engine=args.asr_engine,
        speech_scope="both",
        agent_b_plugin=args.agent_b_plugin,
        configuration={
            **configured,
            "metric_data_dependencies": serializable_metric_dependency_report({
                **configured,
                "tts_engine": args.tts_engine,
                "asr_engine": args.asr_engine,
                "paired_audio_text_runs": args.paired_audio_text,
                "batch_enabled": True,
            }),
            "job_file": args.job_file,
            "iterations": args.iterations,
            "invalid_route_limit": args.invalid_route_limit,
            "constraint_miss_limit": args.constraint_miss_limit,
            "max_turn_elapsed_sec": args.max_turn_elapsed_sec,
            "calculation_max_time_sec": args.calculation_max_time_sec,
            "agent_a_type": args.agent_a_type,
            "model_provider": args.model_provider,
            "model_name": args.model_name,
            "model_base_url": args.model_base_url,
            "model_device": args.model_device,
            "model_timeout_sec": args.model_timeout_sec,
            "model_max_new_tokens": args.model_max_new_tokens,
            "model_max_input_tokens": args.model_max_input_tokens,
            "maximum_progressive_constraints": args.maximum_progressive_constraints,
            "minimum_compared_routes": args.minimum_compared_routes,
            "require_constraint_retention": args.require_constraint_retention,
            "acceptable_duration_ratio": args.acceptable_duration_ratio,
            "minimum_stage_suboptimal_options": args.minimum_stage_suboptimal_options,
            "require_stage_suboptimal_options": args.require_stage_suboptimal_options,
        },
    )
    write_metric_phase_logs(metrics, phase_log_dir)

    print(f"Run folder: {run_dir}")
    print(f"Metrics: {len(metrics)} rows -> {metrics_output}")
    print(f"Protocols: {len(protocol_paths)} files -> {protocol_log_dir}")
    print(f"Artifacts: manifest={manifest_path}; network={artifacts['network_json']}; graph={artifacts['network_graph']}")
    print(f"Logs: phases={phase_log_dir}; sessions={session_log_dir}; audio={speech_audio_dir}")


if __name__ == "__main__":
    main()
