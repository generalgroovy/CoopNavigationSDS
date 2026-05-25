"""Batch experiment entry point for automatic evaluation of speech-dialog conditions."""
import argparse

from minillama.agent_b.config import AGENT_B_PLUGIN
from minillama.agent_b.config import DEFAULT_SPEECH_PATTERN
from minillama.agent_b.config import RUN_MODE, SPEECH_ASR_ENGINE, SPEECH_AUDIO_DIR, SPEECH_ENGINE, SPEECH_INCOMING_ENABLED, SPEECH_OUTGOING_ENABLED, SPEECH_PLAYBACK_ENABLED, SPEECH_REALTIME_ENABLED, SPEECH_SCOPE, SPEECH_TTS_ENGINE
from minillama.agent_b.plugin_registry import AgentBPluginConfig
from minillama.agent_a.config import PERSONAS
from minillama.controller.config import AGENT_A_TRANSFER_TOLERANCE, NETWORK_PICTURE_DIR, NUM_TURNS, RESEARCH_LOG_DIR, SESSION_LOG_DIR, SESSION_LOG_PROFILE
from minillama.controller.runner import ExperimentRunner, build_condition_grid, write_metrics_file
from minillama.evaluation.research_artifacts import write_conversation_protocols, write_experiment_manifest, write_metric_phase_logs, write_network_research_artifacts
from minillama.test_cases.config import DEFAULT_TEST_CASE
from minillama.test_cases.test_cases import TEST_CASES, get_test_case


def parse_csv_arg(value, all_values=None):
    """Parse a comma-separated CLI argument into a list."""
    if not value:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    if any(item.lower() == "all" for item in items):
        return list(all_values or [])
    return items


def parse_bool_flag(value):
    """Parse CLI booleans for speech-stage toggles."""
    normalized = value.lower()
    if normalized in {"1", "true", "on", "yes"}:
        return True
    if normalized in {"0", "false", "off", "no"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def main():
    """Run the configured experiment grid and write metrics output."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-b-plugin", default=AGENT_B_PLUGIN, help="Agent B plugin: minillama, simple, large language model alias, or package.module:factory.")
    parser.add_argument("--run-mode", default=RUN_MODE, choices=("pure_text", "speech"), help="Run as direct text exchange or through text-to-speech and automatic speech recognition.")
    parser.add_argument("--model-provider", choices=("transformers", "openai"))
    parser.add_argument("--test-cases", default="morning_peak_cross_city,midday_transfer,evening_outbound,late_event")
    parser.add_argument("--personas", default="focused_commuter")
    parser.add_argument("--speech-patterns", default=DEFAULT_SPEECH_PATTERN)
    parser.add_argument("--speech-engine", default=SPEECH_ENGINE, choices=("patterned", "file"), help="Speech backend: text-pattern simulator or generated wave files.")
    parser.add_argument("--tts-engine", default=SPEECH_TTS_ENGINE, choices=("", "patterned", "file", "loopback"), help="Text-to-speech stage backend. Empty uses --speech-engine.")
    parser.add_argument("--asr-engine", default=SPEECH_ASR_ENGINE, choices=("", "patterned", "file", "loopback"), help="Automatic speech recognition stage backend. Empty uses --speech-engine.")
    parser.add_argument("--speech-audio-dir", default=SPEECH_AUDIO_DIR, help="Directory for generated speech wave files and transcript artifacts.")
    parser.add_argument("--speech-incoming", type=parse_bool_flag, default=SPEECH_INCOMING_ENABLED, help="Enable incoming automatic speech recognition transcript processing.")
    parser.add_argument("--speech-outgoing", type=parse_bool_flag, default=SPEECH_OUTGOING_ENABLED, help="Enable outgoing text-to-speech verbalization processing.")
    parser.add_argument("--speech-playback", type=parse_bool_flag, default=SPEECH_PLAYBACK_ENABLED, help="Play generated wave files during file-engine runs.")
    parser.add_argument("--speech-real-time", type=parse_bool_flag, default=SPEECH_REALTIME_ENABLED, help="Wait for each spoken turn before automatic speech recognition transcript delivery.")
    parser.add_argument("--speech-scope", default=SPEECH_SCOPE, choices=("both", "agent_a", "agenta", "agent_b", "agentb", "none"))
    parser.add_argument("--speech-enabled", action="store_true", help="Shortcut: enable incoming and outgoing speech for both agents.")
    parser.add_argument("--model-params", default="greedy")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--num-turns", type=int, default=NUM_TURNS)
    parser.add_argument("--agent-a-transfer-tolerance", type=int, default=AGENT_A_TRANSFER_TOLERANCE, choices=(0, 1, 2), help="Extra line changes Agent A accepts over the constraint-aware startup route.")
    parser.add_argument("--output", default="automatic_eval_metrics.xlsx")
    parser.add_argument("--metrics-log-dir", default=None, help="Directory for per-phase metric JSONL files.")
    parser.add_argument("--research-log-dir", default=RESEARCH_LOG_DIR, help="Directory for network/model research logs.")
    parser.add_argument("--protocol-log-dir", default=None, help="Directory for detailed conversation protocol artifacts. Empty uses a conversation_protocols folder inside --research-log-dir.")
    parser.add_argument("--network-picture-dir", default=NETWORK_PICTURE_DIR, help="Directory for generated network graph SVG.")
    parser.add_argument("--log-profile", default=SESSION_LOG_PROFILE, choices=("off", "startup", "runtime", "full"), help="Structured batch logging level.")
    parser.add_argument("--log-dir", default=SESSION_LOG_DIR, help="Directory for optional batch JSONL/session logs.")
    parser.add_argument("--progress", action="store_true", help="Print each completed condition id.")
    args = parser.parse_args()

    if args.speech_enabled:
        args.run_mode = "speech"
        args.speech_incoming = True
        args.speech_outgoing = True
        args.speech_scope = "both"
    if args.run_mode == "pure_text":
        args.speech_incoming = False
        args.speech_outgoing = False
        args.speech_playback = False
        args.speech_real_time = False
        args.speech_scope = "none"

    test_case_keys = parse_csv_arg(args.test_cases, TEST_CASES)
    conditions = list(build_condition_grid(
        test_case_keys=test_case_keys,
        persona_keys=parse_csv_arg(args.personas, PERSONAS),
        speech_pattern_keys=parse_csv_arg(args.speech_patterns, ["clean", "hesitant", "compressed", "noisy_station"]),
        model_param_keys=parse_csv_arg(args.model_params),
        iterations=args.iterations,
    ))

    agent_b_config = AgentBPluginConfig(args.agent_b_plugin)
    if agent_b_config.needs_model:
        from minillama.model.model_runtime import create_model_adapter

        model_adapter = create_model_adapter(args.model_provider) if args.model_provider else create_model_adapter()
    else:
        model_adapter = None
    runner = ExperimentRunner(
        model_adapter,
        args.num_turns,
        agent_b_plugin_key=args.agent_b_plugin,
        run_mode=args.run_mode,
        speech_incoming_enabled=args.speech_incoming,
        speech_outgoing_enabled=args.speech_outgoing,
        speech_scope=args.speech_scope,
        speech_engine=args.speech_engine,
        tts_engine=args.tts_engine,
        asr_engine=args.asr_engine,
        speech_audio_dir=args.speech_audio_dir,
        speech_playback_enabled=args.speech_playback,
        speech_realtime_enabled=args.speech_real_time,
        transfer_tolerance=args.agent_a_transfer_tolerance,
        log_profile=args.log_profile,
        log_dir=args.log_dir,
    )
    results = []
    metrics = []
    for condition in conditions:
        result, metric = runner.run_condition(condition)
        results.append(result)
        metrics.append(metric)
        if args.progress:
            print(f"completed {condition.condition_id}", flush=True)
    write_metrics_file(metrics, args.output)
    protocol_log_dir = args.protocol_log_dir or f"{args.research_log_dir}/conversation_protocols"
    protocol_paths = write_conversation_protocols(results, protocol_log_dir)

    first_case_key = (test_case_keys or [DEFAULT_TEST_CASE])[0]
    first_case = get_test_case(first_case_key)
    artifacts = write_network_research_artifacts(
        first_case.scenario["start_time_min"],
        args.research_log_dir,
        picture_dir=args.network_picture_dir,
    )
    manifest_path = write_experiment_manifest(
        conditions,
        args.research_log_dir,
        num_turns=args.num_turns,
        speech_engine=args.speech_engine,
        tts_engine=args.tts_engine or args.speech_engine,
        asr_engine=args.asr_engine or args.speech_engine,
        speech_scope=args.speech_scope,
        agent_b_plugin=args.agent_b_plugin,
    )
    phase_log_dir = args.metrics_log_dir or f"{args.research_log_dir}/metrics_by_phase"
    write_metric_phase_logs(metrics, phase_log_dir)

    print(f"wrote {len(metrics)} metric rows to {args.output}")
    print(f"wrote {len(protocol_paths)} conversation protocol folders to {protocol_log_dir}")
    print(f"wrote metric phase logs to {phase_log_dir}")
    print(f"wrote experiment manifest to {manifest_path}")
    print(f"wrote network data to {artifacts['network_json']}")
    print(f"wrote network graph to {artifacts['network_graph']}")


if __name__ == "__main__":
    main()
