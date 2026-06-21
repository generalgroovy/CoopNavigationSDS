"""Shared configuration-pipeline, logging, and metric-dependency model."""

from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
import platform
import shutil

from coop_navigation_sds.EvaluationMetrics.catalog import (
    CORE_METRIC_KEYS,
    DEFAULT_METRIC_CONFIG,
    METRIC_DISPLAY_NAMES,
    METRIC_FAMILY_SPECS,
)


PIPELINE_PHASES = (
    ("network", "Network / Scenario / Constraints"),
    ("agent_a", "Agent A"),
    ("agent_b", "Agent B"),
    ("audio_tts", "Audio Persona / TTS"),
    ("asr", "ASR"),
    ("metrics_logging", "Metrics / Logging"),
    ("batch_results", "Batch / Results"),
)

LOGGED_DATA_FIELDS = {
    "network": (
        ("configuration", "Resolved experiment configuration"),
        ("network_graph", "Network nodes, services, timings, and SVG"),
        ("scenario", "Start, destination, departure time, and constraints"),
        ("optimal_route", "Precalculated routes for validity, time, and three progressive constraint layers"),
        ("stage_viability", "Viable alternatives for each conversation stage"),
    ),
    "agent_a": (
        ("agent_a_utterances", "Agent A generated and heard utterances"),
        ("constraint_states", "Revealed and retained constraint states"),
        ("caller_decisions", "Critiques, acceptance, and closure decisions"),
    ),
    "agent_b": (
        ("agent_b_utterances", "Agent B generated and heard utterances"),
        ("route_candidates", "Parsed candidates, validity, duration, and novelty"),
        ("dialogue_states", "Stage transitions, repairs, and warnings"),
    ),
    "audio_tts": (
        ("audio_files", "Per-turn WAV files and combined dialogue audio"),
        ("tts_trace", "Outgoing text, engine, persona, prosody, and latency"),
        ("audio_signal_features", "Duration, clipping, silence, and loudness evidence"),
        ("endpoint_events", "Endpoint and overlap event annotations"),
        ("nisqa_model", "Local NISQA model evidence"),
        ("dnsmos_model", "Local DNSMOS model evidence"),
    ),
    "asr": (
        ("asr_raw_transcript", "Raw recognizer output"),
        ("asr_transcript_edits", "Token misinterpretations and normalization corrections"),
        ("asr_transcript", "Normalized transcript consumed by the listener"),
        ("asr_reference_text", "Synthesized source text for error analysis"),
        ("asr_timing", "Recognition latency and audio duration"),
        ("asr_confidence", "Recognizer confidence values"),
    ),
    "metrics_logging": (
        ("phase_timings", "Generation, synthesis, recognition, NLU, and policy timing"),
        ("runtime_events", "Ordered phase events and failure diagnostics"),
        ("metric_inputs", "Immutable retrospective metric evidence"),
        ("conversation_transcript", "Complete said/heard transcript"),
        ("final_outcome", "Task completion and constraint satisfaction"),
    ),
    "batch_results": (
        ("cross_run_records", "Multiple completed metric rows"),
        ("pair_metadata", "Audio/text pair identifiers and run type"),
        ("pair_comparison", "Paired outcome and repair deltas"),
        ("analysis_workbook", "Phase-grouped XLSX and JSONL outputs"),
    ),
}

FIELD_LABELS = {
    key: label
    for fields in LOGGED_DATA_FIELDS.values()
    for key, label in fields
}

PHASE_METRIC_FIELDS = {
    "user_simulation": ("agent_a_utterances", "constraint_states", "caller_decisions", "route_candidates"),
    "audio_input": ("audio_files", "tts_trace", "audio_signal_features", "phase_timings"),
    "asr": ("asr_reference_text", "asr_raw_transcript", "asr_transcript_edits", "asr_transcript", "asr_timing"),
    "nlu": ("asr_transcript", "route_candidates", "constraint_states"),
    "dialogue_state_tracking": ("dialogue_states", "constraint_states", "route_candidates"),
    "dialogue_management": ("dialogue_states", "route_candidates", "caller_decisions"),
    "backend_task_execution": ("route_candidates", "network_graph", "optimal_route"),
    "nlg": ("agent_b_utterances", "route_candidates", "optimal_route"),
    "tts": ("tts_trace", "audio_files", "asr_transcript"),
    "task_outcome": ("final_outcome", "route_candidates", "optimal_route", "constraint_states"),
    "whole_dialogue": ("conversation_transcript", "phase_timings", "runtime_events", "final_outcome"),
    "metric_validity": ("cross_run_records",),
}

SPECIAL_METRIC_FIELDS = {
    "audio_endpointing_latency": ("endpoint_events",),
    "audio_early_endpoint_rate": ("endpoint_events",),
    "audio_late_endpoint_rate": ("endpoint_events",),
    "audio_end_of_utterance_error": ("endpoint_events",),
    "audio_overlap_rate": ("endpoint_events",),
    "audio_interruption_rate": ("endpoint_events",),
    "audio_barge_in_success_rate": ("endpoint_events",),
    "asr_confidence_calibration_error": ("asr_confidence",),
    "tts_nisqa": ("audio_files", "nisqa_model"),
    "tts_dnsmos": ("audio_files", "dnsmos_model"),
}


def _metric_phase(metric_key):
    for family in METRIC_FAMILY_SPECS:
        if any(key == metric_key for key, _label in family["metrics"]):
            return family["key"]
    return "whole_dialogue"


def metric_required_fields(metric_key):
    """Return concrete immutable trace fields required by one metric."""
    return SPECIAL_METRIC_FIELDS.get(metric_key, PHASE_METRIC_FIELDS[_metric_phase(metric_key)])


def collected_data_fields(config):
    """Resolve which data fields the selected run can actually capture."""
    run_type = str(config.get("run_type", "audio_variant"))
    audio = run_type != "text_only"
    collected = {
        "configuration", "network_graph", "scenario", "optimal_route", "stage_viability",
        "agent_a_utterances", "constraint_states", "caller_decisions",
        "agent_b_utterances", "route_candidates", "dialogue_states",
        "asr_raw_transcript", "asr_transcript_edits", "asr_transcript", "asr_reference_text", "asr_timing",
        "phase_timings", "runtime_events", "metric_inputs", "conversation_transcript",
        "final_outcome", "analysis_workbook",
    }
    if audio:
        collected.update({"audio_files", "tts_trace", "audio_signal_features"})
    if config.get("batch_enabled") or config.get("paired_audio_text_runs"):
        collected.update({"cross_run_records", "pair_metadata", "pair_comparison"})
    if str(config.get("asr_engine")) in {"sapi", "faster_whisper", "qwen3_asr", "sherpa_onnx"}:
        collected.add("asr_confidence")
    if os.environ.get("COOP_NAVIGATION_SDS_NISQA_MODEL"):
        collected.add("nisqa_model")
    if os.environ.get("COOP_NAVIGATION_SDS_DNSMOS_MODEL"):
        collected.add("dnsmos_model")
    return collected


def metric_dependency_report(config, metric_config=None):
    """Return metric-by-field availability used by GUI, preflight, and manifests."""
    enabled = {**DEFAULT_METRIC_CONFIG, **dict(metric_config or config.get("metric_config") or {})}
    collected = collected_data_fields(config)
    metrics = {}
    for key, label in METRIC_DISPLAY_NAMES.items():
        required = tuple(metric_required_fields(key))
        missing = tuple(field for field in required if field not in collected)
        metrics[key] = {
            "label": label,
            "phase": _metric_phase(key),
            "required_fields": required,
            "missing_fields": missing,
            "available": not missing,
            "enabled": bool(enabled.get(key)),
            "core": key in CORE_METRIC_KEYS,
        }
    return {"collected_fields": collected, "metrics": metrics}


def serializable_metric_dependency_report(config, metric_config=None):
    """Return the dependency report in stable JSON-ready form."""
    report = metric_dependency_report(config, metric_config)
    return {
        "collected_fields": sorted(report["collected_fields"]),
        "metrics": report["metrics"],
    }


@dataclass(frozen=True)
class ComponentStatus:
    key: str
    available: bool
    reason: str


def component_status(kind, key, config=None):
    """Return offline availability without importing or downloading models."""
    config = config or {}
    system = platform.system()
    if kind == "model":
        if key == "transformers":
            model_name = str(config.get("model_name", "")).strip()
            if Path(model_name).exists():
                return ComponentStatus(key, True, "Local model directory found")
            try:
                from huggingface_hub import try_to_load_from_cache
                cached = try_to_load_from_cache(model_name, "config.json")
                ready = isinstance(cached, str) and Path(cached).is_file()
            except Exception:
                ready = False
            return ComponentStatus(key, ready, "Local cache found" if ready else "Cache the selected model before runtime")
        if key == "openai_compatible":
            ready = bool(config.get("model_api_key") and config.get("model_base_url"))
            return ComponentStatus(key, ready, "Credentials configured" if ready else "Configure API key and service URL")
        if key == "ollama":
            ready = bool(config.get("model_base_url") and config.get("model_name"))
            return ComponentStatus(key, ready, "Local service configured" if ready else "Configure local Ollama service and model")
    if key == "file":
        return ComponentStatus(key, True, "Built-in deterministic control")
    if key == "sapi":
        ready = system == "Windows" and shutil.which("powershell") is not None
        return ComponentStatus(key, ready, "Windows System.Speech" if ready else "Windows System.Speech is unavailable")
    modules = {
        "chattts": "ChatTTS", "piper": "piper", "coqui": "TTS.api",
        "faster_whisper": "faster_whisper", "vosk": "vosk",
        "qwen3_asr": "qwen_asr", "sherpa_onnx": "sherpa_onnx",
    }
    try:
        from coop_navigation_sds.DialogManagement.provider_runtime import resolve_provider_python
        isolated_python = resolve_provider_python(
            key,
            explicit=config.get(f"{kind}_python_executable", ""),
            environment_dir=config.get("provider_environment_dir", ".speech-providers"),
        )
    except (FileNotFoundError, OSError, ValueError):
        isolated_python = None
    if key == "espeak_ng":
        from coop_navigation_sds.DialogManagement.speech_pipeline import resolve_espeak_executable
        ready = bool(resolve_espeak_executable(config.get("tts_executable", "")))
        return ComponentStatus(key, ready, "Local executable found" if ready else "Install eSpeak NG or configure its executable")
    if key == "whisper_cpp":
        from coop_navigation_sds.DialogManagement.whisper_cpp_runtime import whisper_cpp_ready
        ready, reason, _resolved = whisper_cpp_ready(
            executable=config.get("asr_executable", ""),
            model=config.get("asr_model", ""),
            vad_model=config.get("asr_vad_model", ""),
            environment_dir=config.get("provider_environment_dir", ".speech-providers"),
        )
        return ComponentStatus(key, ready, reason)
    module = modules.get(key)
    ready = bool(isolated_python or (module and importlib.util.find_spec(module)))
    reason = "Python provider installed" if ready else f"Install or configure the {key} provider"
    if key in {"chattts", "piper", "coqui", "faster_whisper", "vosk", "qwen3_asr", "sherpa_onnx"}:
        model_key = "tts_model" if kind == "tts" else "asr_model"
        model_value = str(config.get(model_key, "")).strip()
        model = Path(model_value) if model_value else None
        ready = ready and model is not None and model.exists()
        reason = "Provider and local model found" if ready else f"Configure a local {key} model path"
    return ComponentStatus(key, ready, reason)


def optimal_route_preview(config):
    """Calculate a concise selected-condition baseline for the startup GUI."""
    try:
        from coop_navigation_sds.TransportNetwork.constraints import layered_optimal_routes
        from coop_navigation_sds.TransportNetwork.network import rebuild_network
        from coop_navigation_sds.TransportNetwork.test_cases import get_test_case

        rebuild_network(int(config.get("network_seed", 42)))
        case = get_test_case(config["test_case_key"]).with_persona(config["persona_key"])
        modes = config.get("agent_a_ticket_modes", "metro,tram")
        if isinstance(modes, str):
            modes = tuple(part.strip() for part in modes.split(","))
        case = case.with_scenario_overrides(
            ticket_modes=tuple(modes),
            max_walking_min=int(config.get("agent_a_max_walking_min", 10)),
            acceptable_duration_ratio=float(config.get("acceptable_duration_ratio", 1.5)),
        )
        layers = layered_optimal_routes(case.scenario, case.persona, max_constraints=3)
        available = [layer for layer in layers if layer["available"]]
        if not available:
            return {"available": False, "summary": "No viable route for the selected constraints."}
        route = available[-1]
        def change_text(layer):
            count = layer["line_change_count"]
            return f"{count} {'change' if count == 1 else 'changes'}"

        summary = "\n".join(
            (
                f"{layer['label']}: {layer['path_text']} | "
                f"{layer['duration_min']} min | {change_text(layer)}"
            ) if layer["available"] else f"{layer['label']}: unavailable"
            for layer in layers
        )
        return {
            "available": True,
            "route": route["route"],
            "duration_min": route["duration_min"],
            "changes": route["line_change_count"],
            "layers": layers,
            "summary": summary,
        }
    except Exception as exc:
        return {"available": False, "summary": f"Preview unavailable: {exc}"}
