"""Interactive startup controller for one headless speech-dialog experiment."""
import importlib.util
import logging
import platform
import shutil
import subprocess
from contextlib import nullcontext
from pathlib import Path
from urllib.parse import urlparse

try:
    from huggingface_hub.utils import logging as hf_logging
except ModuleNotFoundError:
    hf_logging = None

from coop_navigation_sds.Configuration.runtime import (
    AGENT_A_TRANSFER_TOLERANCE,
    CONSTRAINT_MISS_LIMIT,
    INVALID_ROUTE_LIMIT,
    MAXIMUM_PROGRESSIVE_CONSTRAINTS,
    MINIMUM_COMPARED_ROUTES,
    NUM_TURNS,
    RESULTS_DIR,
    REQUIRE_CONSTRAINT_RETENTION,
    SESSION_LOG_PROFILE,
    SESSION_NAME,
)
from coop_navigation_sds.Configuration.scenarios import DEFAULT_TEST_CASE
from coop_navigation_sds.Configuration.component_catalog import (
    apply_speech_engine_profiles,
    resolve_prepared_asset_path,
    speech_engine_profile,
    startup_choices,
)
from coop_navigation_sds.Configuration.settings import load_run_settings, save_run_settings
from coop_navigation_sds.Configuration.schema import (
    resolve_results_root,
    runtime_environment_metadata,
    sanitized_config,
)
from coop_navigation_sds.Configuration.run_identity import single_run_label
from coop_navigation_sds.Configuration.speech import (
    AGENT_B_PLUGIN,
    DEFAULT_SPEECH_PATTERN,
    SPEECH_ASR_ENGINE,
    SPEECH_PLAYBACK_ENABLED,
    SPEECH_REALTIME_ENABLED,
    SPEECH_TTS_ENGINE,
)
from coop_navigation_sds.Configuration.experimental_defaults import (
    DEFAULT_ASR_AMBIGUOUS_END_SILENCE_MS,
    DEFAULT_ASR_BABBLE_TIMEOUT_SEC,
    DEFAULT_ASR_BEAM_SIZE,
    DEFAULT_ASR_END_SILENCE_MS,
    DEFAULT_ASR_INITIAL_SILENCE_SEC,
    DEFAULT_ASR_TIMEOUT_SEC,
    DEFAULT_MAX_UTTERANCE_SEC,
    DEFAULT_MIN_UTTERANCE_SEC,
    DEFAULT_TTS_TIMEOUT_SEC,
)
from coop_navigation_sds.Configuration.travel import (
    ALLOW_MODEL_DOWNLOAD,
    CHAT_API_KEY,
    CHAT_BASE_URL,
    CHAT_MODEL,
    CHAT_TIMEOUT_SEC,
    DEVICE,
    GENERATION_MAX_TIME_SEC,
    MAX_INPUT_TOKENS,
    MAX_NEW_TOKENS,
    MODEL,
    MODEL_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    ACCEPTABLE_DURATION_RATIO,
    MIN_STAGE_SUBOPTIMAL_OPTIONS,
    REQUIRE_STAGE_SUBOPTIMAL_OPTIONS,
)
from coop_navigation_sds.NaturalLanguageGeneration.models import (
    available_model_provider_keys,
    ensure_ollama_ready,
    matching_model_profile,
    model_adapter_runtime_metadata,
    model_profile_defaults,
)
from coop_navigation_sds.NaturalLanguageGeneration.assistant.plugin_registry import (
    AgentBPluginConfig,
    create_agent_b_plugin,
)
from coop_navigation_sds.NaturalLanguageGeneration.caller.config import DEFAULT_PERSONA, LLM_AGENT_A
from coop_navigation_sds.NaturalLanguageGeneration.caller.responder import (
    AGENT_A_MINILLAMA,
    LLMAgentAResponder,
    TemplateAgentAResponder,
    agent_a_uses_model,
    normalize_agent_a_type,
)
from coop_navigation_sds.DialogManagement.manager import DEFAULT_MAX_TURN_ELAPSED_SEC, HARD_MAX_TURN_ELAPSED_SEC, DialogManager
from coop_navigation_sds.DialogManagement.provider_runtime import resolve_provider_python
from coop_navigation_sds.DialogManagement.whisper_cpp_runtime import whisper_cpp_ready
from coop_navigation_sds.DialogManagement.result import NullEventQueue
from coop_navigation_sds.ResultsAndArtifacts.artifacts import (
    create_execution_run_dir,
    write_single_run_research_outputs,
)
from coop_navigation_sds.ResultsAndArtifacts.logging import MonitoringEventQueue, SessionLogger
from coop_navigation_sds.EvaluationMetrics.metrics import (
    METRIC_DISPLAY_NAMES,
)
from coop_navigation_sds.EvaluationMetrics.catalog import (
    PHASE_DISPLAY_NAMES,
    global_metric_key,
)
from coop_navigation_sds.DialogManagement.speech_pipeline import (
    SpeechPipelineConfig,
    SpeechPipelineError,
    SpeechTransport,
    available_asr_engine_keys,
    available_tts_engine_keys,
    platform_default_asr_engine,
    platform_default_tts_engine,
    resolve_espeak_executable,
)
from coop_navigation_sds.TextToSpeech.personas import (
    DEFAULT_AGENT_A_AUDIO_PERSONA,
    DEFAULT_AGENT_B_AUDIO_PERSONA,
    audio_persona_keys,
    synthesis_values,
)
from coop_navigation_sds.TransportNetwork.constraints import normalize_objective_mode
from coop_navigation_sds.TransportNetwork.test_cases import get_test_case
from coop_navigation_sds.Configuration.pipeline import (
    optimal_route_preview,
    serializable_metric_dependency_report,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
if hf_logging is not None:
    hf_logging.set_verbosity_warning()


CONSOLE_VIEW_CHOICES = ("compact", "transcript", "debug", "quiet")


class ConsoleEventSink:
    """Print a readable speech transcript and retrospective metric report."""

    def __init__(self, view="compact"):
        self.turn = 0
        self._last_speech_payload = None
        self._system_messages = set()
        self.view = str(view or "compact").strip().lower()
        if self.view not in CONSOLE_VIEW_CHOICES:
            self.view = "compact"

    def put(self, event):
        if not event:
            return
        kind = event[0]
        if kind == "telemetry" and len(event) >= 3 and event[1] == "speech":
            if self.view not in {"quiet"}:
                self._print_speech_turn(event[2])
        elif kind == "telemetry" and len(event) >= 3 and event[1] == "memory":
            if self.view == "debug":
                self._print_memory_state(event[2])
        elif kind == "telemetry" and len(event) >= 3 and event[1] == "phase_timing":
            if self.view == "debug":
                self._print_debug_event("PHASE TIMING", event[2])
        elif kind == "phase" and len(event) >= 2:
            if self.view == "debug":
                self._print_debug_event("PHASE", event[1])
        elif kind == "metrics" and len(event) >= 2:
            self._print_run_summary(event[1])
        elif kind == "metric_results" and len(event) >= 2:
            self._print_metric_results(event[1])
        elif kind == "configuration" and len(event) >= 2:
            self._print_configuration(event[1])
        elif kind == "candidate" and len(event) >= 2:
            if self.view == "debug":
                self._print_debug_event("CANDIDATE", event[1])
        elif kind == "warning" and len(event) >= 2:
            print(f"\nWARNING: {event[1]}", flush=True)
        elif kind == "stage" and len(event) >= 2:
            if self.view == "debug":
                self._print_debug_event("STAGE", event[1])
        elif kind == "system" and len(event) >= 2:
            self._print_system_message(event[1])

    def _print_speech_turn(self, payload):
        if (
            not isinstance(payload, dict)
            or payload is self._last_speech_payload
        ):
            return
        self._last_speech_payload = payload
        self.turn += 1
        speaker = payload.get("speaker", "Agent")
        listener = "Agent B" if speaker == "Agent A" else "Agent A"
        intended = payload.get("generated_text") or ""
        spoken = payload.get("outgoing_text") or payload.get("generated_text") or ""
        raw = payload.get("raw_asr_transcript") or (
            payload.get("diagnostics") or {}
        ).get("raw_asr_transcript") or payload.get("incoming_transcript") or ""
        understood = payload.get("incoming_transcript") or ""
        misinterpretations = payload.get("misinterpreted_tokens") or (
            payload.get("diagnostics") or {}
        ).get("misinterpreted_tokens") or []
        corrections = payload.get("transcript_corrections") or (
            payload.get("diagnostics") or {}
        ).get("transcript_corrections") or []
        print(f"\n--- Turn {self.turn}: {speaker} -> {listener} ---", flush=True)
        if intended.strip() != spoken.strip():
            self._print_transcript_row("INTENDED:", intended)
        self._print_transcript_row("TTS SPEECH:", spoken)
        self._print_transcript_row("ASR HEARD:", raw)
        if understood.strip() != raw.strip():
            self._print_transcript_row("AGENT INPUT:", understood)
        if misinterpretations:
            self._print_transcript_row(
                "TTS -> ASR:",
                self._format_token_changes(misinterpretations, quote_source=True),
            )
        if corrections:
            self._print_transcript_row(
                "ASR -> INPUT:",
                self._format_token_changes(corrections),
            )

    @staticmethod
    def _print_transcript_row(label, value):
        print(f"{label:<15}{value}", flush=True)

    def _print_memory_state(self, payload):
        if not isinstance(payload, dict):
            return
        turn = payload.get("turn", "?")
        reason = payload.get("reason", "memory")
        snapshots = payload.get("snapshots") or {}
        additions = payload.get("additions") or {}
        print(f"MEMORY T{turn} ({reason}):", flush=True)
        for agent in ("Agent A", "Agent B"):
            memory = snapshots.get(agent) or {}
            route = memory.get("current_route_summary") or "none"
            constraints = ", ".join(memory.get("active_constraints") or []) or "none"
            focus = memory.get("pending_focus") or "unknown"
            heard = memory.get("latest_heard") or "none"
            task = self._format_task_variables(memory)
            print(
                f"  {agent}: task={task}; focus={focus}; route={route}; constraints={constraints}; heard={heard}",
                flush=True,
            )
            agent_additions = additions.get(agent) or {}
            if agent_additions:
                print(
                    f"  {agent} new: {self._format_memory_additions(agent_additions)}",
                    flush=True,
                )

    @staticmethod
    def _format_memory_additions(additions):
        rendered = []
        for key, value in additions.items():
            if isinstance(value, list):
                text = ", ".join(str(item) for item in value) or "none"
            else:
                text = str(value)
            rendered.append(f"{key}={text}")
        return "; ".join(rendered)

    @staticmethod
    def _format_task_variables(memory):
        values = memory.get("task_variables") or {}
        missing = memory.get("missing_task_variables") or []
        start = values.get("start_station") or "unknown"
        destination = values.get("destination_station") or "unknown"
        minutes = values.get("start_time_min")
        try:
            time_text = f"{int(minutes) // 60:02d}:{int(minutes) % 60:02d}" if minutes is not None else "unknown"
        except (TypeError, ValueError):
            time_text = "unknown"
        suffix = f"; missing={', '.join(missing)}" if missing else ""
        return f"start={start}; destination={destination}; time={time_text}{suffix}"

    @staticmethod
    def _format_token_changes(changes, quote_source=False):
        if not changes:
            return "none"
        rendered = []
        for change in changes:
            source = " ".join(change.get("source_tokens") or []) or "[none]"
            target = " ".join(change.get("target_tokens") or []) or "[none]"
            displayed_source = f"'{source}'" if quote_source and source != "[none]" else source
            rendered.append(f"{displayed_source} -> {target}")
        return "; ".join(rendered)

    @staticmethod
    def _print_run_summary(summary):
        print("\n=== Conversation And Task Summary ===", flush=True)
        for line in str(summary).strip().splitlines():
            if line.strip():
                print(line, flush=True)

    def _print_system_message(self, message):
        text = str(message)
        if text in self._system_messages:
            return
        self._system_messages.add(text)
        print(f"[system] {text}", flush=True)

    @staticmethod
    def _print_debug_event(label, payload):
        print(f"{label}: {payload}", flush=True)

    @staticmethod
    def _print_configuration(config):
        dependency_report = None
        if isinstance(config, dict):
            dependency_report = config.get("__metric_dependency_report")
        print("\n=== Pre-Experiment Overview ===", flush=True)
        print("[Configuration]", flush=True)
        for label, value in config.items():
            if str(label).startswith("__"):
                continue
            print(f"{label}: {value}", flush=True)
        if dependency_report:
            ConsoleEventSink._print_metric_dependency_overview(dependency_report)

    @staticmethod
    def _print_metric_dependency_overview(report):
        print("\n[Metric Evidence Plan]", flush=True)
        collected = set(report.get("collected_fields") or [])
        print(f"Captured values: {len(collected)} trace fields", flush=True)
        phase_rows = {}
        for key, item in (report.get("metrics") or {}).items():
            phase = item.get("phase", "unknown")
            phase_rows.setdefault(phase, {"obligatory": 0, "available": 0, "missing": set()})
            if item.get("obligatory"):
                phase_rows[phase]["obligatory"] += 1
                if item.get("available"):
                    phase_rows[phase]["available"] += 1
                else:
                    phase_rows[phase]["missing"].update(item.get("missing_fields") or [])
        for phase, row in phase_rows.items():
            missing = ", ".join(sorted(row["missing"])) if row["missing"] else "none"
            print(
                f"{PHASE_DISPLAY_NAMES.get(phase, phase.replace('_', ' ').title())}: "
                f"{row['available']}/{row['obligatory']} calculable; missing fields: {missing}",
                flush=True,
            )

    @staticmethod
    def _print_metric_results(metric):
        print("\n=== Post-Experiment Metric Overview ===", flush=True)
        for phase, values in getattr(metric, "metric_families", {}).items():
            calculable = []
            unavailable = 0
            for key, value in values.items():
                if key in {
                    "available",
                    "coverage_rate",
                    "available_metric_count",
                    "configured_metric_count",
                }:
                    continue
                full_key = global_metric_key(phase, key)
                label = METRIC_DISPLAY_NAMES.get(full_key, key.replace("_", " ").title())
                if value is None:
                    unavailable += 1
                else:
                    calculable.append(f"{label}={ConsoleEventSink._format_metric_value(value)}")
            title = PHASE_DISPLAY_NAMES.get(phase, phase.replace("_", " ").title())
            total = len(calculable) + unavailable
            values_text = " | ".join(calculable) if calculable else "none"
            print(
                f"{title} [{len(calculable)}/{total} calculable]: "
                f"{values_text}"
                + (f" | unavailable={unavailable}" if unavailable else ""),
                flush=True,
            )
        print(
            "Detailed formulas, operands, substitutions, ranges, and unavailable reasons "
            "are stored in retrospective_metrics.json and metrics_long.csv/jsonl.",
            flush=True,
        )

    @staticmethod
    def _format_metric_value(value):
        if value is None:
            return "not available"
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)


def build_agent_a_responder(model_adapter, llm_agent_a=LLM_AGENT_A, agent_a_type=None):
    """Create the configured Agent A responder."""
    normalized_type = normalize_agent_a_type(agent_a_type, llm_agent_a)
    if agent_a_uses_model(normalized_type):
        if model_adapter is None:
            raise RuntimeError(
                f"Agent A implementation '{normalized_type}' requires a loaded model adapter."
            )
        return LLMAgentAResponder(model_adapter)
    return TemplateAgentAResponder()


def agent_a_model_integrity(config, model_adapter):
    """Return explicit Agent A model-use evidence for protocol output."""
    agent_a_type = normalize_agent_a_type((config or {}).get("agent_a_type"))
    uses_model = agent_a_uses_model(agent_a_type)
    profile = (config or {}).get("model_profile")
    expected_profile = "tinyllama_1b_transformers" if agent_a_type == "tinyllama" else profile
    return {
        "agent_a_type": agent_a_type,
        "uses_model": uses_model,
        "adapter_loaded": model_adapter is not None,
        "model_profile": profile,
        "expected_model_profile": expected_profile,
        "valid": (not uses_model) or (model_adapter is not None and profile == expected_profile),
    }


def configure_model_adapter_runtime(model_adapter, calculation_max_time_sec):
    """Apply a local generation budget without changing service timeouts."""
    if model_adapter is None:
        return
    budget = max(0.5, float(calculation_max_time_sec or GENERATION_MAX_TIME_SEC))
    if hasattr(model_adapter, "max_time_sec"):
        model_adapter.max_time_sec = budget


def default_run_config():
    """Return the complete default configuration for one speech experiment."""
    agent_a_audio = synthesis_values(DEFAULT_AGENT_A_AUDIO_PERSONA)
    agent_b_audio = synthesis_values(DEFAULT_AGENT_B_AUDIO_PERSONA)
    default_model_name = (
        CHAT_MODEL
        if MODEL_PROVIDER in {"openai", "openai_compatible"}
        else OLLAMA_MODEL if MODEL_PROVIDER == "ollama" else MODEL
    )
    return {
        "test_case_key": DEFAULT_TEST_CASE,
        "persona_key": DEFAULT_PERSONA,
        "agent_b_plugin": AGENT_B_PLUGIN,
        "model_profile": matching_model_profile(MODEL_PROVIDER, default_model_name),
        "model_provider": MODEL_PROVIDER,
        "model_name": default_model_name,
        "model_api_key": CHAT_API_KEY or "",
        "model_base_url": (
            CHAT_BASE_URL
            if MODEL_PROVIDER in {"openai", "openai_compatible"}
            else OLLAMA_BASE_URL if MODEL_PROVIDER == "ollama" else ""
        ),
        "model_device": DEVICE,
        "model_timeout_sec": 180.0 if MODEL_PROVIDER == "ollama" else CHAT_TIMEOUT_SEC,
        "model_max_new_tokens": MAX_NEW_TOKENS,
        "model_max_input_tokens": MAX_INPUT_TOKENS,
        "allow_model_download": ALLOW_MODEL_DOWNLOAD,
        "model_service_autostart": True,
        "num_turns": NUM_TURNS,
        "invalid_route_limit": INVALID_ROUTE_LIMIT,
        "constraint_miss_limit": CONSTRAINT_MISS_LIMIT,
        "clarification_max_attempts": 2,
        "dialogue_stagnation_limit": 2,
        "agent_a_transfer_tolerance": AGENT_A_TRANSFER_TOLERANCE,
        "agent_a_ticket_modes": "metro,tram",
        "agent_a_max_walking_min": 10,
        "agent_a_max_delay_risk": "high",
        "agent_a_max_transfer_risk": "medium",
        "network_seed": 42,
        "maximum_progressive_constraints": MAXIMUM_PROGRESSIVE_CONSTRAINTS,
        "minimum_compared_routes": MINIMUM_COMPARED_ROUTES,
        "require_constraint_retention": REQUIRE_CONSTRAINT_RETENTION,
        "acceptable_duration_ratio": ACCEPTABLE_DURATION_RATIO,
        "minimum_stage_suboptimal_options": MIN_STAGE_SUBOPTIMAL_OPTIONS,
        "require_stage_suboptimal_options": REQUIRE_STAGE_SUBOPTIMAL_OPTIONS,
        "agent_a_objective_mode": "shortest_valid_route_with_constraints",
        "max_turn_elapsed_sec": DEFAULT_MAX_TURN_ELAPSED_SEC,
        "calculation_max_time_sec": GENERATION_MAX_TIME_SEC,
        "agent_a_type": "userlm" if LLM_AGENT_A else AGENT_A_MINILLAMA,
        "llm_agent_a": LLM_AGENT_A,
        "speech_pattern_key": DEFAULT_SPEECH_PATTERN,
        "tts_engine": SPEECH_TTS_ENGINE or platform_default_tts_engine(),
        "asr_engine": SPEECH_ASR_ENGINE or platform_default_asr_engine(),
        "speech_playback_enabled": SPEECH_PLAYBACK_ENABLED,
        "speech_realtime_enabled": SPEECH_REALTIME_ENABLED,
        "agent_a_audio_persona": DEFAULT_AGENT_A_AUDIO_PERSONA,
        "agent_b_audio_persona": DEFAULT_AGENT_B_AUDIO_PERSONA,
        "agent_a_custom_audio": False,
        "agent_b_custom_audio": False,
        **{f"agent_a_{key}": value for key, value in agent_a_audio.items()},
        **{f"agent_b_{key}": value for key, value in agent_b_audio.items()},
        "tts_device": "auto",
        "tts_model": (
            ".speech-providers/models/chattts"
            if (SPEECH_TTS_ENGINE or platform_default_tts_engine()) == "chattts"
            else ""
        ),
        "tts_executable": "",
        "tts_python_executable": "",
        "tts_timeout_sec": DEFAULT_TTS_TIMEOUT_SEC,
        "asr_language": "en-US",
        "asr_model": "small.en",
        "asr_device": "auto",
        "asr_compute_type": "default",
        "asr_executable": "",
        "asr_python_executable": "",
        "asr_vad_model": "",
        "asr_timeout_sec": DEFAULT_ASR_TIMEOUT_SEC,
        "asr_beam_size": DEFAULT_ASR_BEAM_SIZE,
        "asr_initial_silence_sec": DEFAULT_ASR_INITIAL_SILENCE_SEC,
        "asr_babble_timeout_sec": DEFAULT_ASR_BABBLE_TIMEOUT_SEC,
        "asr_end_silence_ms": DEFAULT_ASR_END_SILENCE_MS,
        "asr_ambiguous_end_silence_ms": DEFAULT_ASR_AMBIGUOUS_END_SILENCE_MS,
        "asr_domain_normalization_enabled": True,
        "asr_domain_similarity_threshold": 0.86,
        "min_utterance_sec": DEFAULT_MIN_UTTERANCE_SEC,
        "max_utterance_sec": DEFAULT_MAX_UTTERANCE_SEC,
        "provider_environment_dir": ".speech-providers",
        "gui_font_size": 11,
        "console_view": "compact",
        "log_profile": SESSION_LOG_PROFILE,
        "paired_audio_text_runs": True,
        "results_root": RESULTS_DIR,
    }


def normalize_run_config(config):
    """Validate and complete a mandatory full-speech run configuration."""
    supplied = dict(config)
    if "results_root" not in supplied and supplied.get("protocol_log_dir"):
        supplied["results_root"] = supplied.pop("protocol_log_dir")
    normalized = {**default_run_config(), **supplied}
    normalized["agent_a_type"] = normalize_agent_a_type(
        supplied.get("agent_a_type"),
        supplied.get("llm_agent_a", normalized.get("llm_agent_a", False)),
    )
    normalized["llm_agent_a"] = agent_a_uses_model(normalized["agent_a_type"])
    if normalized["agent_a_type"] == "tinyllama":
        normalized.update(model_profile_defaults("tinyllama_1b_transformers"))
        normalized["model_profile"] = "tinyllama_1b_transformers"
    selected_profile = str(normalized.get("model_profile") or "custom").strip()
    if selected_profile != "custom" and "model_profile" in supplied:
        profile_values = model_profile_defaults(selected_profile)
        if not profile_values:
            raise ValueError(f"Unknown language-model condition '{selected_profile}'.")
        normalized.update(profile_values)
    if str(normalized.get("agent_b_plugin") or "").strip().lower() in {"", "minillama"}:
        normalized["agent_b_plugin"] = "llm"
    normalized["model_provider"] = str(
        normalized.get("model_provider") or MODEL_PROVIDER
    ).strip().lower()
    if normalized["model_provider"] == "openai":
        normalized["model_provider"] = "openai_compatible"
    if normalized["model_provider"] not in available_model_provider_keys():
        raise ValueError(
            f"Unsupported language-model provider '{normalized['model_provider']}'. "
            f"Allowed: {available_model_provider_keys()}"
        )
    normalized["model_name"] = str(normalized.get("model_name") or "").strip()
    normalized["model_profile"] = (
        selected_profile
        if selected_profile != "custom"
        else matching_model_profile(normalized["model_provider"], normalized["model_name"])
    )
    if normalized["model_provider"] == "ollama" and normalized["model_name"] == "llama3.2:3b":
        normalized["model_name"] = "llama3.2:latest"
    normalized["model_service_autostart"] = bool(normalized.get("model_service_autostart", True))
    normalized["model_api_key"] = str(normalized.get("model_api_key") or "").strip()
    normalized["model_base_url"] = str(normalized.get("model_base_url") or "").strip()
    normalized["model_device"] = str(normalized.get("model_device") or DEVICE).strip()
    normalized["model_timeout_sec"] = max(
        0.5, min(600.0, float(normalized["model_timeout_sec"]))
    )
    normalized["model_max_new_tokens"] = max(
        1, min(2048, int(normalized["model_max_new_tokens"]))
    )
    normalized["model_max_input_tokens"] = max(
        128, min(131072, int(normalized["model_max_input_tokens"]))
    )
    normalized.pop("allow_tts_model_download", None)
    normalized["allow_model_download"] = bool(normalized.get("allow_model_download", ALLOW_MODEL_DOWNLOAD))
    normalized["maximum_progressive_constraints"] = max(
        0, min(6, int(normalized["maximum_progressive_constraints"]))
    )
    ticket_modes = [
        mode.strip().lower()
        for mode in str(normalized.get("agent_a_ticket_modes") or "").split(",")
        if mode.strip().lower() in {"metro", "tram", "bus"}
    ]
    if len(set(ticket_modes)) != 2:
        raise ValueError("Agent A must have exactly two distinct tickets: metro, tram, or bus.")
    normalized["agent_a_ticket_modes"] = ",".join(dict.fromkeys(ticket_modes))
    normalized["agent_a_max_walking_min"] = max(
        0, min(30, int(normalized["agent_a_max_walking_min"]))
    )
    for key in ("agent_a_max_delay_risk", "agent_a_max_transfer_risk"):
        normalized[key] = str(normalized[key]).strip().lower()
        if normalized[key] not in {"low", "medium", "high"}:
            raise ValueError(f"{key} must be low, medium, or high.")
    normalized["network_seed"] = max(0, min(2147483647, int(normalized["network_seed"])))
    normalized["clarification_max_attempts"] = max(
        1, min(6, int(normalized["clarification_max_attempts"]))
    )
    normalized["dialogue_stagnation_limit"] = max(
        1, min(10, int(normalized["dialogue_stagnation_limit"]))
    )
    normalized["minimum_compared_routes"] = max(
        1, min(10, int(normalized["minimum_compared_routes"]))
    )
    normalized["require_constraint_retention"] = bool(
        normalized["require_constraint_retention"]
    )
    normalized["acceptable_duration_ratio"] = max(
        1.0, min(3.0, float(normalized["acceptable_duration_ratio"]))
    )
    normalized["minimum_stage_suboptimal_options"] = max(
        0, min(10, int(normalized["minimum_stage_suboptimal_options"]))
    )
    normalized["require_stage_suboptimal_options"] = bool(
        normalized["require_stage_suboptimal_options"]
    )
    normalized["tts_engine"] = str(
        normalized.get("tts_engine") or platform_default_tts_engine()
    ).strip().lower()
    normalized["asr_engine"] = str(
        normalized.get("asr_engine") or platform_default_asr_engine()
    ).strip().lower()
    profile_seed = {
        "tts_engine": normalized["tts_engine"],
        "asr_engine": normalized["asr_engine"],
    }
    profile_values = apply_speech_engine_profiles(profile_seed, replace=True)
    for stage in ("tts", "asr"):
        if f"{stage}_engine" not in supplied:
            continue
        for key, value in profile_values.items():
            if key.startswith(f"{stage}_") and key not in supplied:
                normalized[key] = value
    for stage in ("tts", "asr"):
        engine = normalized[f"{stage}_engine"]
        profile = speech_engine_profile(stage, engine)
        model_key = f"{stage}_model"
        current_model = resolve_prepared_asset_path(normalized.get(model_key))
        normalized[model_key] = current_model
        known_other_defaults = {
            str(other.get(model_key) or "")
            for other_engine in (
                ("sapi", "chattts", "piper", "espeak_ng", "coqui", "file")
                if stage == "tts"
                else ("sapi", "faster_whisper", "vosk", "whisper_cpp", "qwen3_asr", "sherpa_onnx", "file")
            )
            if other_engine != engine
            for other in (speech_engine_profile(stage, other_engine),)
            if other.get(model_key)
        }
        if (
            not current_model
            or current_model in known_other_defaults
            or (stage == "asr" and engine != "faster_whisper" and current_model == "small.en")
        ):
            normalized[model_key] = profile.get(model_key, current_model)
    for key in (
        "tts_model", "tts_executable", "tts_python_executable",
        "asr_model", "asr_executable", "asr_python_executable", "asr_vad_model",
        "provider_environment_dir",
    ):
        normalized[key] = resolve_prepared_asset_path(normalized.get(key))
    if normalized["tts_engine"] not in available_tts_engine_keys():
        raise SpeechPipelineError(
            f"Unsupported text-to-speech engine '{normalized['tts_engine']}'.",
            {"allowed": available_tts_engine_keys()},
        )
    if normalized["asr_engine"] not in available_asr_engine_keys():
        raise SpeechPipelineError(
            f"Unsupported automatic speech recognition engine '{normalized['asr_engine']}'.",
            {"allowed": available_asr_engine_keys()},
        )
    normalized["max_turn_elapsed_sec"] = min(
        HARD_MAX_TURN_ELAPSED_SEC,
        max(0.5, float(normalized["max_turn_elapsed_sec"])),
    )
    normalized["calculation_max_time_sec"] = min(
        HARD_MAX_TURN_ELAPSED_SEC,
        max(0.5, float(normalized["calculation_max_time_sec"])),
    )
    normalized["min_utterance_sec"] = max(0.1, float(normalized["min_utterance_sec"]))
    normalized["max_utterance_sec"] = max(
        normalized["min_utterance_sec"],
        float(normalized["max_utterance_sec"]),
    )
    for agent in ("agent_a", "agent_b"):
        fallback = DEFAULT_AGENT_A_AUDIO_PERSONA if agent == "agent_a" else DEFAULT_AGENT_B_AUDIO_PERSONA
        available = audio_persona_keys("caller" if agent == "agent_a" else "assistant")
        persona_key = str(normalized.get(f"{agent}_audio_persona") or fallback)
        normalized[f"{agent}_audio_persona"] = persona_key if persona_key in available else fallback
        normalized[f"{agent}_custom_audio"] = False
        profile_values = synthesis_values(normalized[f"{agent}_audio_persona"], fallback)
        for field, value in profile_values.items():
            if (
                normalized["tts_engine"] == "chattts"
                and field in {"seed", "temperature", "top_p"}
                and f"{agent}_{field}" in supplied
            ):
                continue
            normalized[f"{agent}_{field}"] = value
        normalized[f"{agent}_speech_rate"] = max(-10, min(10, int(normalized[f"{agent}_speech_rate"])))
        normalized[f"{agent}_volume"] = max(0, min(100, int(normalized[f"{agent}_volume"])))
        normalized[f"{agent}_pitch_semitones"] = max(-12, min(12, int(normalized[f"{agent}_pitch_semitones"])))
        normalized[f"{agent}_pause_ms"] = max(0, min(2000, int(normalized[f"{agent}_pause_ms"])))
        normalized[f"{agent}_speed"] = max(0.25, min(4.0, float(normalized[f"{agent}_speed"])))
        normalized[f"{agent}_temperature"] = max(0.01, min(2.0, float(normalized[f"{agent}_temperature"])))
        normalized[f"{agent}_top_p"] = max(0.01, min(1.0, float(normalized[f"{agent}_top_p"])))
        normalized[f"{agent}_top_k"] = max(1, min(1000, int(normalized[f"{agent}_top_k"])))
        normalized[f"{agent}_oral_level"] = max(0, min(9, int(normalized[f"{agent}_oral_level"])))
        normalized[f"{agent}_laugh_level"] = max(0, min(2, int(normalized[f"{agent}_laugh_level"])))
        normalized[f"{agent}_break_level"] = max(0, min(7, int(normalized[f"{agent}_break_level"])))
        emphasis = str(normalized[f"{agent}_emphasis"]).lower()
        normalized[f"{agent}_emphasis"] = emphasis if emphasis in {"none", "reduced", "moderate", "strong"} else "none"
    normalized["tts_timeout_sec"] = max(1.0, min(600.0, float(normalized["tts_timeout_sec"])))
    normalized["asr_timeout_sec"] = max(1.0, min(600.0, float(normalized["asr_timeout_sec"])))
    normalized["tts_python_executable"] = str(
        normalized.get("tts_python_executable") or ""
    ).strip()
    normalized["asr_python_executable"] = str(
        normalized.get("asr_python_executable") or ""
    ).strip()
    normalized["provider_environment_dir"] = str(
        normalized.get("provider_environment_dir") or ".speech-providers"
    ).strip()
    normalized["asr_beam_size"] = max(1, min(20, int(normalized["asr_beam_size"])))
    normalized["asr_initial_silence_sec"] = max(
        0.1, min(30.0, float(normalized["asr_initial_silence_sec"]))
    )
    normalized["asr_babble_timeout_sec"] = max(
        0.1, min(30.0, float(normalized["asr_babble_timeout_sec"]))
    )
    normalized["asr_end_silence_ms"] = max(
        100, min(10000, int(normalized["asr_end_silence_ms"]))
    )
    normalized["asr_ambiguous_end_silence_ms"] = max(
        normalized["asr_end_silence_ms"],
        min(15000, int(normalized["asr_ambiguous_end_silence_ms"])),
    )
    normalized["asr_domain_normalization_enabled"] = bool(
        normalized["asr_domain_normalization_enabled"]
    )
    normalized["asr_domain_similarity_threshold"] = max(
        0.70, min(1.0, float(normalized["asr_domain_similarity_threshold"]))
    )
    # Per-metric switches are obsolete. Every registered metric is evaluated.
    normalized.pop("metric_config", None)
    normalized.pop("metric_tiers", None)
    normalized["agent_a_objective_mode"] = normalize_objective_mode(normalized.get("agent_a_objective_mode"))
    normalized["console_view"] = str(normalized.get("console_view") or "compact").strip().lower()
    if normalized["console_view"] not in CONSOLE_VIEW_CHOICES:
        raise ValueError(f"console_view must be one of {CONSOLE_VIEW_CHOICES}.")
    normalized["log_profile"] = str(normalized.get("log_profile") or SESSION_LOG_PROFILE).strip().lower()
    if normalized["log_profile"] not in {"off", "startup", "runtime", "full"}:
        raise ValueError("log_profile must be off, startup, runtime, or full.")
    normalized["results_root"] = resolve_results_root(
        str(normalized.get("results_root") or RESULTS_DIR).strip()
    )
    return normalized


def select_run_config():
    """Show the startup-only configuration window."""
    defaults = normalize_run_config(load_run_settings({}))
    choices = startup_choices(
        AGENT_B_PLUGIN,
        defaults,
        operational_only=True,
    )
    from coop_navigation_sds.Configuration.gui import StartupConfigDialog

    selected = StartupConfigDialog(
        choices,
        defaults,
        validator=validate_run_config_for_start,
    ).show()
    if selected is None:
        return None
    normalized = normalize_run_config(selected)
    save_run_settings(normalized)
    return normalized


def _require_module(module_name, implementation):
    if importlib.util.find_spec(module_name) is None:
        raise ValueError(
            f"{implementation} is selected, but Python module '{module_name}' "
            "is not installed in the interpreter running this program."
        )


def _provider_python(config, stage, engine):
    explicit = config[f"{stage}_python_executable"]
    try:
        return resolve_provider_python(
            engine,
            explicit=explicit,
            environment_dir=config["provider_environment_dir"],
        )
    except FileNotFoundError as exc:
        raise ValueError(str(exc)) from exc


def _require_module_or_provider(config, stage, engine, module_name, implementation):
    provider_python = _provider_python(config, stage, engine)
    if provider_python is not None:
        try:
            probe = subprocess.run(
                [str(provider_python), "-c", f"import importlib; importlib.import_module({module_name!r})"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=60,
            )
        except OSError as exc:
            raise ValueError(
                f"{implementation} provider process cannot start: {exc}"
            ) from exc
        if probe.returncode != 0:
            raise ValueError(f"{implementation} provider cannot initialize: import failed")
        return
    _require_module(module_name, implementation)


def _validate_transformers_model(model_name, allow_download):
    """Confirm that an offline local-model selection can actually be loaded."""
    model_path = Path(model_name)
    if model_path.exists():
        if not model_path.is_dir():
            raise ValueError("The local Transformers model path must be a directory.")
        return
    if allow_download:
        return
    try:
        from huggingface_hub import scan_cache_dir

        cached_ids = {repo.repo_id for repo in scan_cache_dir().repos}
    except Exception as exc:
        raise ValueError(
            "The selected Transformers model cannot be verified in the local cache. "
            "Install it before runtime or select a local model directory."
        ) from exc
    if model_name not in cached_ids:
        raise ValueError(
            f"Transformers model '{model_name}' is not present in the local cache. "
            "Install it before runtime or select a local model directory."
        )


def validate_run_config_for_start(config):
    """Validate implementation-specific requirements before closing the GUI."""
    normalized = normalize_run_config(config)
    agent_b = AgentBPluginConfig(normalized["agent_b_plugin"])
    needs_model = agent_b.needs_model or agent_a_uses_model(normalized["agent_a_type"])
    if needs_model:
        provider = normalized["model_provider"]
        if not normalized["model_name"]:
            raise ValueError("The selected language-model provider requires a model identifier.")
        if provider == "transformers":
            _require_module("torch", "Local Transformers")
            _require_module("transformers", "Local Transformers")
            _validate_transformers_model(
                normalized["model_name"],
                normalized["allow_model_download"],
            )
        elif provider == "openai_compatible":
            hostname = urlparse(normalized["model_base_url"]).hostname
            if not normalized["model_api_key"] and hostname not in {"localhost", "127.0.0.1", "::1"}:
                raise ValueError(
                    "ChatGPT/OpenAI-compatible models require an API key. "
                    "Enter it in the API key field or set OPENAI_API_KEY."
                )
            if not normalized["model_base_url"]:
                raise ValueError("The OpenAI-compatible provider requires a service URL.")
        elif provider == "llama_cpp":
            if not normalized["model_base_url"]:
                raise ValueError("llama.cpp requires its local OpenAI-compatible API URL.")
            hostname = urlparse(normalized["model_base_url"]).hostname
            if hostname not in {"localhost", "127.0.0.1", "::1"}:
                raise ValueError("The llama.cpp backend must use a local loopback service URL.")
        elif provider == "ollama":
            if not normalized["model_base_url"]:
                raise ValueError("The Ollama provider requires its API URL.")
            try:
                ensure_ollama_ready(
                    normalized["model_base_url"],
                    normalized["model_name"],
                    autostart=normalized["model_service_autostart"],
                    timeout_sec=min(normalized["model_timeout_sec"], 180.0),
                    warm_model=True,
                )
            except RuntimeError as exc:
                raise ValueError(str(exc)) from exc

    tts_engine = normalized["tts_engine"]
    if tts_engine == "sapi" and platform.system() != "Windows":
        raise ValueError("Windows SAPI text-to-speech is available only on Windows.")
    if tts_engine == "chattts":
        if _provider_python(normalized, "tts", tts_engine) is None:
            _require_module("torch", "ChatTTS")
            _require_module("ChatTTS", "ChatTTS")
        if not normalized["tts_model"] or not Path(normalized["tts_model"]).exists():
            raise ValueError("ChatTTS assets are missing. Run the environment preparation command.")
    if tts_engine == "piper":
        _require_module_or_provider(normalized, "tts", tts_engine, "piper", "Piper")
        if not Path(normalized["tts_model"]).is_file():
            raise ValueError("Piper requires a readable local ONNX voice model path.")
    elif tts_engine == "espeak_ng":
        executable = resolve_espeak_executable(normalized["tts_executable"])
        if not executable:
            raise ValueError("eSpeak NG requires `espeak-ng` on PATH or an explicit executable path.")
    elif tts_engine == "coqui":
        _require_module_or_provider(normalized, "tts", tts_engine, "TTS.api", "Coqui TTS")
        if not normalized["tts_model"]:
            raise ValueError("Coqui requires a local path or model identifier.")
        if not Path(normalized["tts_model"]).exists():
            raise ValueError("Coqui model assets are not available locally.")

    asr_engine = normalized["asr_engine"]
    if asr_engine == "sapi" and platform.system() != "Windows":
        raise ValueError("Windows SAPI recognition is available only on Windows.")
    if asr_engine == "faster_whisper":
        provider_python = _provider_python(normalized, "asr", asr_engine)
        _require_module_or_provider(
            normalized,
            "asr",
            asr_engine,
            "faster_whisper",
            "Faster-Whisper",
        )
        if not Path(normalized["asr_model"]).exists():
            raise ValueError("Faster-Whisper requires a local CTranslate2 model directory.")
    elif asr_engine == "vosk":
        if normalized["asr_model"] == "small.en":
            raise ValueError(
                "Vosk requires a local model directory. The value 'small.en' is "
                "a Whisper model name and cannot be used by Vosk."
            )
        model_path = Path(normalized["asr_model"])
        if normalized["asr_model"] and not model_path.is_dir():
            raise ValueError("Vosk requires a readable local model directory.")
        _require_module_or_provider(normalized, "asr", asr_engine, "vosk", "Vosk")
    elif asr_engine == "whisper_cpp":
        ready, _message, resolved = whisper_cpp_ready(
            executable=normalized["asr_executable"],
            model=normalized["asr_model"],
            vad_model=normalized["asr_vad_model"],
            environment_dir=normalized["provider_environment_dir"],
        )
        if not ready:
            raise ValueError(
                "whisper.cpp requires a readable GGML model file and a whisper-cli "
                "or main executable. Set them in the ASR fields, PATH, or "
                ".speech-providers/providers.json."
            )
    elif asr_engine == "qwen3_asr":
        provider_python = _provider_python(normalized, "asr", asr_engine)
        _require_module_or_provider(normalized, "asr", asr_engine, "qwen_asr", "Qwen3-ASR")
        if not Path(normalized["asr_model"]).exists():
            raise ValueError("Qwen3-ASR model assets are not available locally.")
    elif asr_engine == "sherpa_onnx":
        if not Path(normalized["asr_model"]).is_dir():
            raise ValueError("sherpa-onnx requires a readable local model directory.")
        _require_module_or_provider(normalized, "asr", asr_engine, "sherpa_onnx", "sherpa-onnx")

    from coop_navigation_sds.TransportNetwork.constraints import stage_viability_report
    from coop_navigation_sds.TransportNetwork.network import rebuild_network
    rebuild_network(normalized["network_seed"])
    test_case = configured_test_case(normalized)
    viability = stage_viability_report(test_case.scenario, test_case.persona)
    if not viability["all_stage_requirements_satisfied"]:
        failed = [stage["stage"] for stage in viability["stages"] if not stage["requirement_satisfied"]]
        raise ValueError(
            "The selected network and caller constraints do not provide the configured "
            f"number of alternatives for conversation stage(s) {failed}. Change the "
            "network seed, tickets, walking limit, duration ratio, or alternative requirement."
        )

    results_root = Path(normalized["results_root"])
    results_root.mkdir(parents=True, exist_ok=True)
    return normalized


def configured_test_case(run_config):
    """Build the standardized case with all caller and stage overrides."""
    return (
        get_test_case(run_config["test_case_key"])
        .with_persona(run_config["persona_key"])
        .with_scenario_overrides(
            acceptable_duration_ratio=run_config["acceptable_duration_ratio"],
            min_stage_suboptimal_options=run_config["minimum_stage_suboptimal_options"],
            require_stage_suboptimal_options=run_config["require_stage_suboptimal_options"],
            maximum_progressive_constraints=run_config["maximum_progressive_constraints"],
            minimum_compared_routes=run_config["minimum_compared_routes"],
            require_constraint_retention=run_config["require_constraint_retention"],
            maximum_dialog_turns=run_config["num_turns"],
            clarification_max_attempts=run_config["clarification_max_attempts"],
            dialogue_stagnation_limit=run_config["dialogue_stagnation_limit"],
            ticket_modes=tuple(run_config["agent_a_ticket_modes"].split(",")),
            max_walking_min=run_config["agent_a_max_walking_min"],
            max_delay_probability={"low": 0.24, "medium": 0.44, "high": 1.0}[run_config["agent_a_max_delay_risk"]],
            max_transfer_miss_probability={"low": 0.24, "medium": 0.44, "high": 1.0}[run_config["agent_a_max_transfer_risk"]],
        )
    )


def prepare_execution_run_config(run_config):
    """Create one shallow result folder before any runtime stage starts."""
    normalized = normalize_run_config(run_config)
    if not normalized.get("execution_run_dir"):
        run_dir = create_execution_run_dir(
            normalized["results_root"],
            label=single_run_label(normalized),
        )
        normalized["execution_run_dir"] = str(run_dir)
        normalized["speech_audio_dir"] = str(run_dir)
    return normalized


def build_dialog_runtime(event_queue, model_adapter, run_config):
    """Build agents, speech stages, and dialogue orchestration."""
    run_config = normalize_run_config(run_config)
    from coop_navigation_sds.TransportNetwork.network import rebuild_network
    rebuild_network(run_config["network_seed"])
    configure_model_adapter_runtime(model_adapter, run_config["calculation_max_time_sec"])
    test_case = configured_test_case(run_config)
    agent_b_plugin = create_agent_b_plugin(run_config["agent_b_plugin"], model_adapter)
    speech_transport = SpeechTransport(
        config=SpeechPipelineConfig(
            pattern_key=run_config["speech_pattern_key"],
            tts_engine=run_config["tts_engine"],
            asr_engine=run_config["asr_engine"],
            audio_dir=run_config["speech_audio_dir"],
            agent_a_audio_persona=run_config["agent_a_audio_persona"],
            agent_b_audio_persona=run_config["agent_b_audio_persona"],
            agent_a_custom_audio=run_config["agent_a_custom_audio"],
            agent_b_custom_audio=run_config["agent_b_custom_audio"],
            playback_enabled=run_config["speech_playback_enabled"],
            realtime_enabled=run_config["speech_realtime_enabled"],
            agent_a_words_per_minute=int(run_config["agent_a_words_per_minute"]),
            agent_b_words_per_minute=int(run_config["agent_b_words_per_minute"]),
            agent_a_voice=run_config["agent_a_voice"],
            agent_b_voice=run_config["agent_b_voice"],
            agent_a_speech_rate=int(run_config["agent_a_speech_rate"]),
            agent_b_speech_rate=int(run_config["agent_b_speech_rate"]),
            agent_a_volume=int(run_config["agent_a_volume"]),
            agent_b_volume=int(run_config["agent_b_volume"]),
            agent_a_pitch_semitones=int(run_config["agent_a_pitch_semitones"]),
            agent_b_pitch_semitones=int(run_config["agent_b_pitch_semitones"]),
            agent_a_pause_ms=int(run_config["agent_a_pause_ms"]),
            agent_b_pause_ms=int(run_config["agent_b_pause_ms"]),
            agent_a_emphasis=run_config["agent_a_emphasis"],
            agent_b_emphasis=run_config["agent_b_emphasis"],
            agent_a_language=run_config["agent_a_language"],
            agent_b_language=run_config["agent_b_language"],
            agent_a_speed=float(run_config["agent_a_speed"]),
            agent_b_speed=float(run_config["agent_b_speed"]),
            agent_a_temperature=float(run_config["agent_a_temperature"]),
            agent_b_temperature=float(run_config["agent_b_temperature"]),
            agent_a_top_p=float(run_config["agent_a_top_p"]),
            agent_b_top_p=float(run_config["agent_b_top_p"]),
            agent_a_top_k=int(run_config["agent_a_top_k"]),
            agent_b_top_k=int(run_config["agent_b_top_k"]),
            agent_a_seed=int(run_config["agent_a_seed"]),
            agent_b_seed=int(run_config["agent_b_seed"]),
            agent_a_oral_level=int(run_config["agent_a_oral_level"]),
            agent_b_oral_level=int(run_config["agent_b_oral_level"]),
            agent_a_laugh_level=int(run_config["agent_a_laugh_level"]),
            agent_b_laugh_level=int(run_config["agent_b_laugh_level"]),
            agent_a_break_level=int(run_config["agent_a_break_level"]),
            agent_b_break_level=int(run_config["agent_b_break_level"]),
            agent_a_reference_audio=run_config["agent_a_reference_audio"],
            agent_b_reference_audio=run_config["agent_b_reference_audio"],
            agent_a_reference_text=run_config["agent_a_reference_text"],
            agent_b_reference_text=run_config["agent_b_reference_text"],
            tts_device=run_config["tts_device"],
            tts_model=run_config["tts_model"],
            tts_executable=run_config["tts_executable"],
            tts_python_executable=run_config["tts_python_executable"],
            tts_timeout_sec=float(run_config["tts_timeout_sec"]),
            asr_language=run_config["asr_language"],
            asr_model=run_config["asr_model"],
            asr_device=run_config["asr_device"],
            asr_compute_type=run_config["asr_compute_type"],
            asr_executable=run_config["asr_executable"],
            asr_python_executable=run_config["asr_python_executable"],
            asr_vad_model=run_config["asr_vad_model"],
            asr_timeout_sec=float(run_config["asr_timeout_sec"]),
            asr_beam_size=int(run_config["asr_beam_size"]),
            asr_initial_silence_sec=float(run_config["asr_initial_silence_sec"]),
            asr_babble_timeout_sec=float(run_config["asr_babble_timeout_sec"]),
            asr_end_silence_ms=int(run_config["asr_end_silence_ms"]),
            asr_ambiguous_end_silence_ms=int(run_config["asr_ambiguous_end_silence_ms"]),
            asr_domain_normalization_enabled=run_config["asr_domain_normalization_enabled"],
            asr_domain_similarity_threshold=float(run_config["asr_domain_similarity_threshold"]),
            min_utterance_sec=float(run_config["min_utterance_sec"]),
            max_utterance_sec=float(run_config["max_utterance_sec"]),
            provider_environment_dir=run_config["provider_environment_dir"],
        )
    )
    manager = DialogManager(
        test_case,
        agent_b_plugin,
        int(run_config["num_turns"]),
        speech_transport=speech_transport,
        agent_a_responder=build_agent_a_responder(
            model_adapter,
            llm_agent_a=run_config["llm_agent_a"],
            agent_a_type=run_config["agent_a_type"],
        ),
        monitor=event_queue if hasattr(event_queue, "log_step") else None,
        invalid_route_limit=int(run_config["invalid_route_limit"]),
        constraint_miss_limit=int(run_config["constraint_miss_limit"]),
        stagnation_limit=int(run_config["dialogue_stagnation_limit"]),
        transfer_tolerance=int(run_config["agent_a_transfer_tolerance"]),
        max_turn_elapsed_sec=float(run_config["max_turn_elapsed_sec"]),
        agent_a_objective_mode=run_config["agent_a_objective_mode"],
    )
    return run_config, test_case, agent_b_plugin, speech_transport, manager


def _segment(event_queue, name, **payload):
    segment = getattr(event_queue, "segment", None)
    return segment(name, **payload) if callable(segment) else nullcontext()


def conversation_worker(event_queue, model_adapter, run_config):
    """Execute one full speech dialogue and write all research outputs."""
    speech_transport = None
    try:
        run_config = prepare_execution_run_config(run_config)
        run_config, test_case, agent_b_plugin, speech_transport, manager = build_dialog_runtime(
            event_queue, model_adapter, run_config
        )
        model_name = getattr(model_adapter, "name", "no-model")
        with _segment(
            event_queue,
            "dialog.run",
            model=model_name,
            provider=run_config["model_provider"] if model_adapter is not None else "none",
            turns=run_config["num_turns"],
            calculation_max_time_sec=run_config["calculation_max_time_sec"],
            agent_a=manager.agent_a_responder.name,
            agent_b=getattr(agent_b_plugin, "name", type(agent_b_plugin).__name__),
            speech_pipeline=speech_transport.description,
        ):
            preview = optimal_route_preview(run_config)
            def layer_change_text(layer):
                count = layer["line_change_count"]
                return f"{count} {'change' if count == 1 else 'changes'}"

            layer_configuration = {
                f"Optimal path [{layer['label']}]": (
                    f"{layer['path_text']} | {layer['duration_min']} min | "
                    f"{layer_change_text(layer)}"
                    if layer["available"] else "unavailable"
                )
                for layer in preview.get("layers", [])
            }
            event_queue.put(("configuration", {
                "Scenario": test_case.name,
                "Agent A": f"{manager.agent_a_responder.name} / {run_config['persona_key']}",
                "Agent B": f"{getattr(agent_b_plugin, 'name', type(agent_b_plugin).__name__)} / {run_config['model_name']}",
                "Audio personas": f"{run_config['agent_a_audio_persona']} / {run_config['agent_b_audio_persona']}",
                "TTS": run_config["tts_engine"],
                "ASR": run_config["asr_engine"],
                "Speech": speech_transport.description,
                **layer_configuration,
                "__metric_dependency_report": serializable_metric_dependency_report(run_config),
                "Results": run_config["execution_run_dir"],
            }))
            health = speech_transport.health_check()
            if not health["ok"]:
                raise SpeechPipelineError("Speech preflight failed.", health)
            result = manager.run(event_queue)
            result.extra["speech_preflight"] = health
            result.extra["agent_a_audio_persona"] = run_config["agent_a_audio_persona"]
            result.extra["agent_b_audio_persona"] = run_config["agent_b_audio_persona"]
            result.extra["resolved_audio_personas"] = {
                "agent_a": speech_transport.config.prosody_for("Agent A"),
                "agent_b": speech_transport.config.prosody_for("Agent B"),
            }
            result.extra["metric_data_dependencies"] = serializable_metric_dependency_report(run_config)
            model_roles = []
            if AgentBPluginConfig(run_config["agent_b_plugin"]).needs_model:
                model_roles.append("agent_b")
            if agent_a_uses_model(run_config["agent_a_type"]):
                model_roles.append("agent_a")
            result.extra["model_backend"] = model_adapter_runtime_metadata(
                model_adapter,
                provider=run_config["model_provider"],
                profile=run_config.get("model_profile", "custom"),
                roles=model_roles,
            )
            result.extra["agent_a_model_integrity"] = agent_a_model_integrity(
                run_config,
                model_adapter,
            )
            result.extra["resolved_run_config"] = sanitized_config(run_config)
            result.extra["runtime_environment"] = runtime_environment_metadata()
            paths = write_single_run_research_outputs(
                result,
                test_case.scenario,
                run_config["results_root"],
                run_dir=run_config["execution_run_dir"],
            )
            event_queue.put(("metric_results", paths["metric"]))
            event_queue.put(("system", f"Results: {run_config['execution_run_dir']}"))
            return result, paths
    except SpeechPipelineError as exc:
        logging.error("Speech pipeline failed: %s", exc)
        event_queue.put(("warning", f"Speech pipeline failed: {exc}"))
        event_queue.put(("warning", f"Diagnostics: {exc.diagnostics}"))
        return None
    except Exception as exc:
        logging.exception("Conversation failed")
        event_queue.put(("warning", f"Conversation stopped: {exc}"))
        return None
    finally:
        if speech_transport is not None:
            speech_transport.close()
        close = getattr(event_queue, "close", None)
        if callable(close):
            close()


def _load_model_or_fallback(run_config, event_queue):
    agent_b_config = AgentBPluginConfig(run_config["agent_b_plugin"])
    agent_a_model = agent_a_uses_model(run_config["agent_a_type"])
    if not agent_b_config.needs_model and not agent_a_model:
        return None
    roles = []
    if agent_a_model:
        roles.append(f"Agent A {run_config['agent_a_type']}")
    if agent_b_config.needs_model:
        roles.append("Agent B")
    event_queue.put((
        "system",
        f"Loading language model for {', '.join(roles)}: "
        f"{run_config['model_name']} through {run_config['model_provider']}.",
    ))
    try:
        from coop_navigation_sds.NaturalLanguageGeneration.model_runtime import create_model_adapter

        with _segment(
            event_queue,
            "model.load",
            model_provider=run_config["model_provider"],
            model_name=run_config["model_name"],
        ):
            return create_model_adapter(
                run_config["model_provider"],
                model_name=run_config["model_name"],
                api_key=run_config["model_api_key"],
                base_url=run_config["model_base_url"],
                timeout_sec=run_config["model_timeout_sec"],
                device=run_config["model_device"],
                max_new_tokens=run_config["model_max_new_tokens"],
                max_input_tokens=run_config["model_max_input_tokens"],
                allow_model_download=run_config["allow_model_download"],
            )
    except Exception as exc:
        logging.exception("Model loading failed")
        raise RuntimeError(
            f"Preflight failed: local model assets for {run_config['model_name']} are unavailable: {exc}"
        ) from exc


def main():
    """Configure and execute one experiment without a runtime GUI."""
    run_config = select_run_config()
    if run_config is None:
        return
    run_config = prepare_execution_run_config(run_config)
    run_dir = Path(run_config["execution_run_dir"])

    logger = None
    if run_config["log_profile"] != "off":
        logger = SessionLogger(SESSION_NAME, run_dir, profile=run_config["log_profile"])
    event_queue = MonitoringEventQueue(ConsoleEventSink(run_config["console_view"]), logger)
    model_adapter = _load_model_or_fallback(run_config, event_queue)
    conversation_worker(event_queue, model_adapter, run_config)


if __name__ == "__main__":
    main()
