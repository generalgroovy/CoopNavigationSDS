"""Shared configuration-pipeline, logging, and metric-dependency model."""

from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
import platform
import shutil
import subprocess
from urllib import error, request

from coop_navigation_sds.Configuration.assets import faster_whisper_model_ready
from coop_navigation_sds.EvaluationMetrics.catalog import (
    METRIC_DISPLAY_NAMES,
    METRIC_FAMILY_SPECS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _transformers_assets_ready(model_name):
    """Check config, tokenizer, and weights without importing Transformers."""
    model_name = str(model_name or "").strip()
    prepared = PROJECT_ROOT / ".speech-providers" / "models" / "huggingface" / model_name.replace("/", "--")
    candidates = [Path(model_name).expanduser(), prepared]
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        config_ready = (candidate / "config.json").is_file()
        tokenizer_ready = any(
            (candidate / name).is_file()
            for name in (
                "tokenizer.json", "tokenizer_config.json", "tokenizer.model",
                "spiece.model", "vocab.json",
            )
        )
        weights_ready = any(
            path.is_file() and path.stat().st_size > 0
            for pattern in ("*.safetensors", "pytorch_model*.bin", "model*.bin")
            for path in candidate.glob(pattern)
        )
        if config_ready and tokenizer_ready and weights_ready:
            return True, str(candidate.resolve())
    try:
        from huggingface_hub import try_to_load_from_cache
        config = try_to_load_from_cache(model_name, "config.json")
        tokenizer = next(
            (
                path for name in ("tokenizer.json", "tokenizer_config.json", "tokenizer.model")
                if isinstance((path := try_to_load_from_cache(model_name, name)), str)
            ),
            None,
        )
        weights = next(
            (
                path for name in ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json")
                if isinstance((path := try_to_load_from_cache(model_name, name)), str)
            ),
            None,
        )
        if all(isinstance(path, str) and Path(path).is_file() for path in (config, tokenizer, weights)):
            return True, "Hugging Face cache"
    except Exception:
        pass
    return False, "missing config, tokenizer, or model weights"


def _speech_assets_ready(key, model):
    """Validate the minimum local asset signature required by a speech engine."""
    if model is None or not model.exists():
        return False
    if model.is_file():
        return model.stat().st_size > 0
    files = [path for path in model.rglob("*") if path.is_file() and path.stat().st_size > 0]
    names = {path.name.casefold() for path in files}
    suffixes = {path.suffix.casefold() for path in files}
    if key == "chattts":
        legacy_weights = {"vocos.pt", "dvae.pt", "gpt.pt"}
        current_weights = {
            "decoder.safetensors", "dvae.safetensors", "embed.safetensors",
            "vocos.safetensors", "model.safetensors",
        }
        return "config.json" in names and bool(
            legacy_weights.intersection(names) or current_weights.intersection(names)
        ) and bool({".pt", ".safetensors"}.intersection(suffixes))
    if key == "piper":
        return any(path.suffix.casefold() == ".onnx" for path in files)
    if key == "faster_whisper":
        return "config.json" in names and bool({"model.bin", "model.safetensors"} & names)
    if key == "vosk":
        relative_parts = {part.casefold() for path in files for part in path.relative_to(model).parts}
        return {"am", "conf", "graph"}.issubset(relative_parts)
    if key == "qwen3_asr":
        return "config.json" in names and bool({".safetensors", ".bin"} & suffixes)
    if key == "sherpa_onnx":
        return any("token" in name for name in names) and ".onnx" in suffixes
    return bool(files)


PIPELINE_PHASES = (
    ("network", "Network / Scenario / Constraints"),
    ("agent_a", "Agent A"),
    ("agent_b", "Agent B"),
    ("audio_tts", "Audio Persona / TTS"),
    ("asr", "ASR"),
    ("metrics_logging", "Metrics / Logging"),
    ("batch_results", "Batch / Results"),
)

EXECUTION_PHASE_CONTRACT = (
    ("preflight", "Scenario and backend preflight", "resolved configuration", "validated scenario, staged optima, provider readiness"),
    ("agent_policy", "Agent policy and language generation", "private memory and heard state", "intended utterance and prompt audit"),
    ("tts", "Text-to-speech", "intended utterance and audio persona", "waveform and synthesis diagnostics"),
    ("asr", "Automatic speech recognition", "generated waveform", "raw transcript and recognition diagnostics"),
    ("nlu", "Normalization and language understanding", "raw transcript", "listener input, semantic frame, transcript edits"),
    ("dialogue_management", "Dialogue state and management", "semantic frame and private memory", "state transition, candidate evaluation, next speaker"),
    ("capture", "Immutable evidence capture", "phase outputs and timings", "protocol and metric inputs"),
    ("evaluation", "Retrospective evaluation and export", "completed immutable evidence", "metrics, calculations, tables, and manifest"),
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
        ("audio_reference_pairs", "Aligned clean-reference and synthesized WAV pairs"),
        ("licensed_polqa_scores", "Scores from a licensed POLQA implementation"),
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
    "tts_nisqa": ("audio_files", "nisqa_model"),
    "tts_dnsmos": ("audio_files", "dnsmos_model"),
    "tts_pesq": ("audio_files", "audio_reference_pairs"),
    "tts_polqa": ("audio_files", "audio_reference_pairs", "licensed_polqa_scores"),
    "tts_stoi": ("audio_files", "audio_reference_pairs"),
    "tts_si_sdr": ("audio_files", "audio_reference_pairs"),
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
    if config.get("batch_enabled"):
        collected.add("cross_run_records")
    if config.get("paired_audio_text_runs"):
        collected.update({"pair_metadata", "pair_comparison"})
    if str(config.get("asr_engine")) in {"sapi", "faster_whisper", "qwen3_asr", "sherpa_onnx"}:
        collected.add("asr_confidence")
    if os.environ.get("COOP_NAVIGATION_SDS_NISQA_MODEL"):
        collected.add("nisqa_model")
    if os.environ.get("COOP_NAVIGATION_SDS_DNSMOS_MODEL"):
        collected.add("dnsmos_model")
    if config.get("tts_reference_audio_manifest"):
        collected.add("audio_reference_pairs")
    if config.get("polqa_result_manifest"):
        collected.add("licensed_polqa_scores")
    return collected


def metric_dependency_report(config):
    """Return metric-by-field availability used by GUI, preflight, and manifests."""
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
            "obligatory": True,
            "enabled": not missing,
        }
    return {"collected_fields": collected, "metrics": metrics}


def serializable_metric_dependency_report(config):
    """Return the dependency report in stable JSON-ready form."""
    report = metric_dependency_report(config)
    return {
        "collected_fields": sorted(report["collected_fields"]),
        "metrics": report["metrics"],
    }


def experiment_pipeline_contract(config):
    """Return the selected phase contract and evidence readiness for manifests."""
    report = metric_dependency_report(config)
    metrics_by_phase = {}
    for metric in report["metrics"].values():
        phase = metric["phase"]
        row = metrics_by_phase.setdefault(phase, {"total": 0, "calculable": 0})
        row["total"] += 1
        row["calculable"] += int(bool(metric["available"]))
    return {
        "selected_components": {
            "agent_a": config.get("agent_a_type"),
            "agent_b": config.get("agent_b_plugin"),
            "language_model": config.get("model_name"),
            "text_to_speech": config.get("tts_engine"),
            "automatic_speech_recognition": config.get("asr_engine"),
        },
        "phases": [
            {
                "index": index,
                "key": key,
                "label": label,
                "input": phase_input,
                "output": phase_output,
            }
            for index, (key, label, phase_input, phase_output) in enumerate(
                EXECUTION_PHASE_CONTRACT,
                start=1,
            )
        ],
        "captured_fields": list(report["collected_fields"]),
        "metric_readiness_by_phase": metrics_by_phase,
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
            if config.get("allow_model_download"):
                return ComponentStatus(key, True, "Model may be downloaded by Transformers during execution")
            ready, location = _transformers_assets_ready(model_name)
            return ComponentStatus(
                key,
                ready,
                f"Complete local model found at {location}"
                if ready else f"Local model is incomplete: {location}",
            )
        if key == "openai_compatible":
            ready = bool(config.get("model_api_key") and config.get("model_base_url"))
            return ComponentStatus(key, ready, "Credentials configured" if ready else "Configure API key and service URL")
        if key == "llama_cpp":
            base_url = str(config.get("model_base_url") or "").rstrip("/")
            try:
                with request.urlopen(f"{base_url}/models", timeout=0.5) as response:
                    ready = 200 <= int(getattr(response, "status", 200)) < 300
            except (OSError, ValueError, error.URLError, TimeoutError):
                ready = False
            return ComponentStatus(
                key,
                ready,
                "Local llama.cpp endpoint responded" if ready else "Start the configured llama.cpp server",
            )
        if key == "ollama":
            try:
                from coop_navigation_sds.NaturalLanguageGeneration.models import ensure_ollama_ready
                ensure_ollama_ready(
                    config.get("model_base_url"),
                    config.get("model_name"),
                    autostart=False,
                    timeout_sec=0.5,
                    models_dir=config.get("model_store_dir"),
                )
            except Exception as exc:
                return ComponentStatus(key, False, str(exc))
            return ComponentStatus(key, True, "Local Ollama service and model responded")
    if key == "file":
        return ComponentStatus(key, True, "Built-in deterministic control")
    if key == "sapi":
        ready = system == "Windows" and shutil.which("powershell") is not None
        return ComponentStatus(key, ready, "Windows System.Speech" if ready else "Windows System.Speech is unavailable")
    modules = {
        "chattts": "ChatTTS", "piper": "piper",
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
    if isolated_python and module:
        try:
            probe_module = module.split(".", 1)[0]
            probe = subprocess.run(
                [
                    str(isolated_python),
                    "-c",
                    (
                        "import importlib.util,sys;"
                        f"sys.exit(0 if importlib.util.find_spec({probe_module!r}) else 1)"
                    ),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=20,
                check=False,
            )
            provider_ready = probe.returncode == 0
        except (OSError, subprocess.SubprocessError):
            provider_ready = False
    else:
        try:
            provider_ready = bool(module and importlib.util.find_spec(module))
        except (ImportError, ModuleNotFoundError, ValueError):
            provider_ready = False
    ready = provider_ready
    reason = "Python provider import succeeded" if ready else f"Install or configure the {key} provider"
    if key in {"chattts", "piper", "faster_whisper", "vosk", "qwen3_asr", "sherpa_onnx"}:
        model_key = "tts_model" if kind == "tts" else "asr_model"
        model_value = str(config.get(model_key, "")).strip()
        model = Path(model_value) if model_value else None
        if key == "faster_whisper":
            asset_ready, resolved_model = faster_whisper_model_ready(model_value)
        else:
            asset_ready = _speech_assets_ready(key, model)
            resolved_model = model_value
        ready = ready and asset_ready
        reason = (
            f"Provider import and local model checks passed: {resolved_model}"
            if ready
            else f"Configure a readable local {key} model path"
        )
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
                f"{index}. {layer['label']}:\n"
                f"   {layer['duration_min']} min, {change_text(layer)}\n"
                f"   {layer['path_text']}"
            ) if layer["available"] else f"{index}. {layer['label']}: unavailable"
            for index, layer in enumerate(layers, start=1)
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


def route_layer_comparison(layers, selected_index):
    """Return edge and cost differences between one optimum and its predecessor."""
    layers = list(layers or ())
    if not 0 <= int(selected_index) < len(layers):
        return None
    selected_index = int(selected_index)
    selected = layers[selected_index]
    previous = layers[selected_index - 1] if selected_index > 0 else None

    def edges(layer):
        return frozenset(
            (
                step.get("from"),
                step.get("to"),
                step.get("line") or step.get("mode") or "walking",
            )
            for step in (layer or {}).get("steps", ())
            if step.get("from") and step.get("to")
        )

    selected_edges = edges(selected)
    previous_edges = edges(previous)
    retained = selected_edges if previous is None else selected_edges & previous_edges
    return {
        "selected": selected,
        "previous": previous,
        "retained_edges": retained,
        "added_edges": frozenset() if previous is None else selected_edges - previous_edges,
        "removed_edges": previous_edges - selected_edges,
        "duration_delta_min": (
            selected.get("duration_min", 0) - previous.get("duration_min", 0)
            if previous else None
        ),
        "line_change_delta": (
            selected.get("line_change_count", 0) - previous.get("line_change_count", 0)
            if previous else None
        ),
    }
