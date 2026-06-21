"""Startup-only Tk configuration window.

The dialog closes before model loading and experiment execution. Runtime output
is written to the run result folder; there is intentionally no live GUI.
"""
import tkinter as tk
import tkinter.font as tkfont
from tkinter import messagebox, ttk

from coop_navigation_sds.Configuration.settings import default_settings_path
from coop_navigation_sds.Configuration.component_catalog import speech_engine_profile
from coop_navigation_sds.Configuration.experimental_defaults import numeric_range
from coop_navigation_sds.NaturalLanguageGeneration.caller.personas import get_persona, preference_text
from coop_navigation_sds.NaturalLanguageGeneration.models import (
    model_provider_defaults,
)
from coop_navigation_sds.EvaluationMetrics.catalog import (
    metric_calculation_method,
)
from coop_navigation_sds.EvaluationMetrics.metrics import (
    METRIC_FAMILY_SPECS,
    metric_config_with_defaults,
    metric_tier_config_with_defaults,
)
from coop_navigation_sds.Configuration.pipeline import (
    FIELD_LABELS,
    LOGGED_DATA_FIELDS,
    component_status,
    metric_dependency_report,
    optimal_route_preview,
)


COMBINED_GUI_PHASES = (
    ("network_task", "Network and Task"),
    ("agent_a", "Agent A"),
    ("agent_b_nlg", "Agent B and Natural-Language Generation"),
    ("tts", "Text-to-Speech"),
    ("asr", "Automatic Speech Recognition"),
    ("nlu", "Natural-Language Understanding"),
    ("dialogue_management", "Dialogue Management"),
    ("results", "Results, Logging, and Batch"),
)

GUI_METRIC_FAMILIES = {
    "network_task": ("backend_task_execution", "task_outcome"),
    "agent_a": ("user_simulation",),
    "agent_b_nlg": ("nlg",),
    "tts": ("audio_input", "tts"),
    "asr": ("asr",),
    "nlu": ("nlu",),
    "dialogue_management": ("dialogue_state_tracking", "dialogue_management", "whole_dialogue"),
    "results": ("metric_validity",),
}


SETTING_HELP = {
    "test_case_key": "Selects the network scenario, start, destination, and travel conditions.",
    "persona_key": "Controls Agent A's dialogue behavior and private travel preferences.",
    "agent_b_plugin": "Selects the deterministic assistant, local language model, or custom plugin factory.",
    "agent_a_objective_mode": "Defines whether Agent A requires any valid route, the shortest route, or staged constraint satisfaction.",
    "agent_a_type": "Selects the deterministic staged caller policy or the model-backed UserLM caller.",
    "model_provider": "Language-model runtime used by the configurable Agent B implementation.",
    "model_name": "Provider-specific model identifier or local model path.",
    "model_api_key": "Credential sent to the selected OpenAI-compatible service. It is not needed for local Transformers or Ollama.",
    "model_base_url": "Address of the selected model service. Use the service's API root, not a web interface URL.",
    "model_device": "Hardware used for local Transformers inference. Use cpu unless CUDA or another accelerator is installed and verified.",
    "model_timeout_sec": "Maximum seconds to wait for one response from an API or Ollama service before the turn fails.",
    "model_service_autostart": "Start a locally installed Ollama service during preflight when the configured loopback endpoint is not running.",
    "model_max_new_tokens": "Maximum generated response length. Higher values allow more natural route explanations but can increase latency.",
    "num_turns": "Maximum total dialogue turns before the experiment stops.",
    "invalid_route_limit": "Stops the dialogue after this many invalid route proposals.",
    "constraint_miss_limit": "Stops the dialogue after this many proposals that miss already stated constraints.",
    "clarification_max_attempts": "Number of targeted hearing-repair turns allowed before the agents reset and request the three trip facts separately.",
    "agent_a_transfer_tolerance": "Additional transfers Agent A accepts beyond the constraint-aware baseline.",
    "agent_a_ticket_modes": "Exactly two public transport tickets available to Agent A. The third public mode is unavailable.",
    "agent_a_max_walking_min": "Maximum cumulative walking time accepted across the complete route.",
    "agent_a_max_delay_risk": "Highest acceptable whole-route delay class. Risk is reported only as low, medium, or high.",
    "agent_a_max_transfer_risk": "Highest acceptable missed-connection risk class for a transfer.",
    "network_seed": "Reproducible seed controlling station mode pairs, service selection, travel times, transfer times, and demand profiles.",
    "maximum_progressive_constraints": "Maximum number of private travel requirements Agent A can reveal. They are introduced one at a time only after the previous goal is satisfied.",
    "minimum_compared_routes": "Minimum number of distinct valid route candidates discussed before Agent A normally chooses one.",
    "require_constraint_retention": "A replacement route must preserve every previously satisfied stated constraint.",
    "acceptable_duration_ratio": "Maximum allowed route duration divided by the optimal duration. For example, 1.5 accepts routes under fifty percent longer than optimal.",
    "minimum_stage_suboptimal_options": "Required viable non-best alternatives verified before the dialogue starts.",
    "require_stage_suboptimal_options": "Fails scenario preflight when a stage lacks the configured alternative routes.",
    "max_turn_elapsed_sec": "Maximum recorded and enforced processing time for one turn.",
    "calculation_max_time_sec": "Maximum generation or route-calculation budget for one agent response.",
    "tts_engine": "Speech synthesizer used to create the actual audio heard by the other agent. Changing it changes the acoustic experimental condition.",
    "asr_engine": "Recognizer that converts synthesized audio into the transcript the listening agent actually receives.",
    "tts_device": "Requested device for optional local neural synthesis backends.",
    "tts_model": "Provider-specific text-to-speech model identifier or local voice path.",
    "tts_executable": "Optional synthesizer executable path. Empty uses the operating-system PATH.",
    "asr_language": "Recognition language or locale, such as en-US.",
    "asr_model": "Provider-specific recognition model identifier or local model path.",
    "asr_device": "Recognition device, such as auto, cpu, cuda, or cuda:0.",
    "asr_compute_type": "Faster-Whisper compute type, such as default, int8, or float16.",
    "asr_executable": "Optional whisper.cpp executable path. Empty searches for whisper-cli.",
    "asr_vad_model": "Optional whisper.cpp voice-activity-detection model path.",
    "asr_beam_size": "Number of recognition hypotheses explored by compatible recognizers.",
    "asr_end_silence_ms": "Pause length required before recognition treats an utterance as complete.",
    "asr_ambiguous_end_silence_ms": "Longer pause required when recognition considers an utterance incomplete or ambiguous; raising it prevents cutoff during natural pauses.",
    "max_utterance_sec": "Maximum synthesized utterance duration; increase it when long route instructions are being truncated.",
    "asr_domain_normalization_enabled": "Conservatively repairs close station, line, and route-word recognition errors after any selected recognizer; the raw transcript remains logged.",
    "asr_domain_similarity_threshold": "Minimum text similarity for automatic domain-term repair. Higher values reduce corrections; lower values tolerate more recognition variation.",
    "speech_pattern_key": "Controlled speaking condition applied before synthesis, such as clean speech, hesitation, pauses, or dropped words.",
    "speech_playback_enabled": "Plays each synthesized utterance through the system audio output.",
    "speech_realtime_enabled": "Waits for playback to finish before the listening agent receives its transcript.",
    "protocol_log_dir": "Parent folder for run results. Each execution creates one self-contained timestamped subfolder.",
}

PROSODY_HELP = {
    "audio_persona": "Named, reproducible voice condition. It fixes speaking rate, volume, pitch, and pause behavior without exposing backend-specific controls.",
    "temperature": "ChatTTS variation level. Lower values produce more repeatable speech; higher values increase acoustic variation.",
    "top_p": "ChatTTS sampling range. Lower values restrict variation; higher values allow a broader set of speech-code choices.",
    "seed": "ChatTTS random seed used to reproduce the same sampled speaker condition.",
}


class ToolTip:
    """Small hover tooltip for configuration controls."""

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.window = None
        widget.bind("<Enter>", self.show, add="+")
        widget.bind("<Leave>", self.hide, add="+")

    def show(self, _event=None):
        if self.window or not self.text:
            return
        self.window = tk.Toplevel(self.widget)
        self.window.wm_overrideredirect(True)
        x = self.widget.winfo_rootx() + 18
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.window.wm_geometry(f"+{x}+{y}")
        ttk.Label(
            self.window,
            text=self.text,
            padding=(7, 4),
            relief="solid",
            borderwidth=1,
            wraplength=360,
        ).pack()

    def hide(self, _event=None):
        if self.window:
            self.window.destroy()
            self.window = None


class StartupConfigDialog:
    """Collect one interactive run configuration."""

    def __init__(self, choices, defaults, validator=None):
        self.choices = choices
        self.defaults = defaults
        self.validator = validator
        self.result = None
        self.root = tk.Tk()
        self.root.title("CoopNavigationSDS Experiment Configuration")
        self.root.minsize(980, 700)
        self.root.after_idle(self._maximize_window)
        self.persona_detail = tk.StringVar()
        self.dynamic_frames = {}
        self._last_tts_engine = str(defaults["tts_engine"])
        self._last_asr_engine = str(defaults["asr_engine"])
        self._tts_model_by_engine = {self._last_tts_engine: str(defaults.get("tts_model", ""))}
        self._asr_model_by_engine = {self._last_asr_engine: str(defaults.get("asr_model", ""))}
        self.pipeline_summary_vars = {key: tk.StringVar() for key, _label in COMBINED_GUI_PHASES}
        self.phase_metric_summary_vars = {key: tk.StringVar() for key, _label in COMBINED_GUI_PHASES}
        self.phase_status_widgets = {}
        self.phase_metric_panels = {}
        self.optimal_route_text = tk.StringVar(value="Calculating selected-condition baseline...")
        self.logging_status_text = tk.StringVar()
        self.vars = {
            "test_case_key": tk.StringVar(value=defaults["test_case_key"]),
            "persona_key": tk.StringVar(value=defaults["persona_key"]),
            "agent_b_plugin": tk.StringVar(value=defaults["agent_b_plugin"]),
            "model_provider": tk.StringVar(value=defaults["model_provider"]),
            "model_name": tk.StringVar(value=defaults["model_name"]),
            "model_api_key": tk.StringVar(value=defaults["model_api_key"]),
            "model_base_url": tk.StringVar(value=defaults["model_base_url"]),
            "model_device": tk.StringVar(value=defaults["model_device"]),
            "model_timeout_sec": tk.DoubleVar(value=defaults["model_timeout_sec"]),
            "model_max_new_tokens": tk.IntVar(value=defaults["model_max_new_tokens"]),
            "model_service_autostart": tk.BooleanVar(value=defaults.get("model_service_autostart", True)),
            "agent_a_type": tk.StringVar(value=defaults["agent_a_type"]),
            "agent_a_objective_mode": tk.StringVar(value=defaults["agent_a_objective_mode"]),
            "num_turns": tk.IntVar(value=defaults["num_turns"]),
            "invalid_route_limit": tk.IntVar(value=defaults["invalid_route_limit"]),
            "constraint_miss_limit": tk.IntVar(value=defaults["constraint_miss_limit"]),
            "clarification_max_attempts": tk.IntVar(value=defaults["clarification_max_attempts"]),
            "agent_a_transfer_tolerance": tk.IntVar(value=defaults["agent_a_transfer_tolerance"]),
            "agent_a_ticket_modes": tk.StringVar(value=defaults["agent_a_ticket_modes"]),
            "agent_a_max_walking_min": tk.IntVar(value=defaults["agent_a_max_walking_min"]),
            "agent_a_max_delay_risk": tk.StringVar(value=defaults["agent_a_max_delay_risk"]),
            "agent_a_max_transfer_risk": tk.StringVar(value=defaults["agent_a_max_transfer_risk"]),
            "network_seed": tk.IntVar(value=defaults["network_seed"]),
            "maximum_progressive_constraints": tk.IntVar(value=defaults["maximum_progressive_constraints"]),
            "minimum_compared_routes": tk.IntVar(value=defaults["minimum_compared_routes"]),
            "require_constraint_retention": tk.BooleanVar(value=defaults["require_constraint_retention"]),
            "acceptable_duration_ratio": tk.DoubleVar(value=defaults["acceptable_duration_ratio"]),
            "minimum_stage_suboptimal_options": tk.IntVar(value=defaults["minimum_stage_suboptimal_options"]),
            "require_stage_suboptimal_options": tk.BooleanVar(value=defaults["require_stage_suboptimal_options"]),
            "max_turn_elapsed_sec": tk.DoubleVar(value=defaults["max_turn_elapsed_sec"]),
            "calculation_max_time_sec": tk.DoubleVar(value=defaults["calculation_max_time_sec"]),
            "speech_pattern_key": tk.StringVar(value=defaults["speech_pattern_key"]),
            "tts_engine": tk.StringVar(value=defaults["tts_engine"]),
            "asr_engine": tk.StringVar(value=defaults["asr_engine"]),
            "speech_playback_enabled": tk.BooleanVar(value=defaults["speech_playback_enabled"]),
            "speech_realtime_enabled": tk.BooleanVar(value=defaults["speech_realtime_enabled"]),
            "tts_device": tk.StringVar(value=defaults["tts_device"]),
            "tts_model": tk.StringVar(value=defaults["tts_model"]),
            "tts_executable": tk.StringVar(value=defaults["tts_executable"]),
            "asr_language": tk.StringVar(value=defaults["asr_language"]),
            "asr_model": tk.StringVar(value=defaults["asr_model"]),
            "asr_device": tk.StringVar(value=defaults["asr_device"]),
            "asr_compute_type": tk.StringVar(value=defaults["asr_compute_type"]),
            "asr_executable": tk.StringVar(value=defaults["asr_executable"]),
            "asr_vad_model": tk.StringVar(value=defaults["asr_vad_model"]),
            "asr_beam_size": tk.IntVar(value=defaults["asr_beam_size"]),
            "asr_initial_silence_sec": tk.DoubleVar(value=defaults["asr_initial_silence_sec"]),
            "asr_babble_timeout_sec": tk.DoubleVar(value=defaults["asr_babble_timeout_sec"]),
            "asr_end_silence_ms": tk.IntVar(value=defaults["asr_end_silence_ms"]),
            "asr_ambiguous_end_silence_ms": tk.IntVar(value=defaults["asr_ambiguous_end_silence_ms"]),
            "min_utterance_sec": tk.DoubleVar(value=defaults["min_utterance_sec"]),
            "max_utterance_sec": tk.DoubleVar(value=defaults["max_utterance_sec"]),
            "asr_domain_normalization_enabled": tk.BooleanVar(value=defaults["asr_domain_normalization_enabled"]),
            "asr_domain_similarity_threshold": tk.DoubleVar(value=defaults["asr_domain_similarity_threshold"]),
            "agent_a_audio_persona": tk.StringVar(value=defaults["agent_a_audio_persona"]),
            "agent_b_audio_persona": tk.StringVar(value=defaults["agent_b_audio_persona"]),
            "agent_a_temperature": tk.DoubleVar(value=defaults["agent_a_temperature"]),
            "agent_b_temperature": tk.DoubleVar(value=defaults["agent_b_temperature"]),
            "agent_a_top_p": tk.DoubleVar(value=defaults["agent_a_top_p"]),
            "agent_b_top_p": tk.DoubleVar(value=defaults["agent_b_top_p"]),
            "agent_a_seed": tk.IntVar(value=defaults["agent_a_seed"]),
            "agent_b_seed": tk.IntVar(value=defaults["agent_b_seed"]),
            "protocol_log_dir": tk.StringVar(value=defaults["protocol_log_dir"]),
            "gui_font_size": tk.IntVar(value=defaults.get("gui_font_size", 11)),
            "paired_audio_text_runs": tk.BooleanVar(value=defaults.get("paired_audio_text_runs", True)),
        }
        metric_tiers = metric_tier_config_with_defaults(defaults.get("metric_tiers"))
        metric_defaults = metric_config_with_defaults(
            defaults.get("metric_config"),
            metric_tiers,
        )
        self.metric_vars = {
            key: tk.BooleanVar(value=metric_defaults[key])
            for family in METRIC_FAMILY_SPECS
            for key, _label in family["metrics"]
        }
        self.metric_tier_vars = {
            key: tk.StringVar(value=metric_tiers[key])
            for family in METRIC_FAMILY_SPECS
            for key, _label in family["metrics"]
        }
        self.metric_enabled_widgets = {}
        self._build()
        self._refresh_persona_detail()
        self._refresh_conditional_sections()
        self.root.protocol("WM_DELETE_WINDOW", self.cancel)

    def _maximize_window(self):
        """Fill the available desktop while retaining native window controls."""
        try:
            self.root.state("zoomed")
        except tk.TclError:
            width = self.root.winfo_screenwidth()
            height = self.root.winfo_screenheight()
            self.root.geometry(f"{width}x{height}+0+0")

    def _build(self):
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        family = "Segoe UI" if self.root.tk.call("tk", "windowingsystem") == "win32" else "DejaVu Sans"
        size = int(self.defaults.get("gui_font_size", 11))
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont"):
            tkfont.nametofont(name).configure(family=family, size=size)
        style.configure("TLabelframe.Label", font=(family, size, "bold"))
        style.configure("Pipeline.TLabelframe.Label", font=(family, size + 1, "bold"))
        style.configure("Heading.TLabel", font=(family, size + 3, "bold"))
        style.configure("TButton", padding=(10, 4))
        style.configure("TCheckbutton", padding=(2, 2))

        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        shell = ttk.Frame(self.root, padding=(7, 6, 7, 2))
        shell.grid(row=0, column=0, sticky="nsew")
        shell.rowconfigure(1, weight=1)
        shell.columnconfigure(0, weight=1)

        heading = ttk.Frame(shell)
        heading.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        heading.columnconfigure(0, weight=1)
        ttk.Label(
            heading,
            text="Cooperative Route-Dialogue Experiment",
            style="Heading.TLabel",
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            heading,
            text="Experiments are local-only. Prepare providers and model assets before starting a run.",
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))
        ttk.Label(
            heading,
            text=f"Saved settings: {default_settings_path()}",
        ).grid(row=2, column=0, sticky="w", pady=(2, 0))

        workspace = ttk.LabelFrame(shell, text="Experiment pipeline", padding=5)
        workspace.grid(row=1, column=0, sticky="nsew")
        workspace.rowconfigure(0, weight=1)
        workspace.columnconfigure(0, weight=1)
        self._build_combined_pipeline(workspace)

        actions = ttk.Frame(self.root)
        actions.grid(row=1, column=0, sticky="ew", padx=7, pady=(4, 6))
        actions.columnconfigure(0, weight=1)
        ttk.Label(
            actions,
            text="Preflight validates local components and metric evidence before execution.",
        ).grid(row=0, column=0, sticky="w")
        ttk.Button(actions, text="Cancel", command=self.cancel).grid(row=0, column=1, padx=(8, 8))
        ttk.Button(actions, text="Start Experiment", command=self.start).grid(row=0, column=2)

        for key in (
            "test_case_key", "persona_key", "agent_a_type", "agent_b_plugin",
            "model_provider", "model_name", "tts_engine", "asr_engine",
            "agent_a_audio_persona", "agent_b_audio_persona", "protocol_log_dir",
        ):
            self.vars[key].trace_add("write", self._schedule_pipeline_refresh)
        for variable in self.metric_vars.values():
            variable.trace_add("write", self._schedule_pipeline_refresh)
        self.root.after_idle(self._refresh_pipeline_overview)

    def _build_combined_pipeline(self, parent):
        content = self._vertical_pipeline(parent)
        content.columnconfigure(0, weight=1, uniform="phase_column")
        content.columnconfigure(1, weight=1, uniform="phase_column")
        cards = {
            key: self._combined_phase_card(content, index, key, title)
            for index, (key, title) in enumerate(COMBINED_GUI_PHASES)
        }

        network = cards["network_task"]
        self._combo(network, 2, "Scenario", "test_case_key", self.choices["test_case_keys"])
        self._number(network, 3, "Network seed", "network_seed", 0, 2147483647)
        self._combo(network, 4, "Tickets", "agent_a_ticket_modes", ("metro,tram", "metro,bus", "tram,bus"))
        self._number(network, 5, "Walking limit", "agent_a_max_walking_min", 0, 30)
        self._number(network, 6, "Duration ratio", "acceptable_duration_ratio", 1.0, 3.0, 0.05)
        ttk.Label(network, textvariable=self.optimal_route_text, wraplength=680, justify="left").grid(
            row=7, column=0, columnspan=3, sticky="ew", pady=(5, 0)
        )

        caller = cards["agent_a"]
        self._combo(caller, 2, "Persona", "persona_key", self.choices["persona_keys"], on_change=self._refresh_persona_detail)
        self._combo(caller, 3, "Implementation", "agent_a_type", self.choices["agent_a_types"], on_change=self._refresh_agent_selection)
        self._combo(caller, 4, "Objective", "agent_a_objective_mode", self.choices["agent_a_objective_modes"])
        ttk.Label(caller, textvariable=self.persona_detail, wraplength=680, justify="left").grid(
            row=5, column=0, columnspan=3, sticky="ew", pady=(5, 0)
        )

        assistant = cards["agent_b_nlg"]
        self._combo(
            assistant, 2, "Policy", "agent_b_plugin", self.choices["agent_b_plugins"],
            editable=True, on_change=self._refresh_agent_selection,
        )
        self.agent_b_model_controls = ttk.Frame(assistant)
        self.agent_b_model_controls.grid(row=3, column=0, columnspan=3, sticky="ew")
        self.agent_b_model_controls.columnconfigure(1, weight=1)
        self._combo(
            self.agent_b_model_controls, 0, "Model provider", "model_provider",
            self.choices["model_providers"], on_change=self._select_model_provider,
        )
        self._entry(self.agent_b_model_controls, 1, "Model", "model_name")
        self.dynamic_frames["model"] = self._collapsible(assistant, 4, "Implementation settings")
        self.agent_b_model_advanced = self.dynamic_frames["model"].master

        tts = cards["tts"]
        self._combo(tts, 2, "Engine", "tts_engine", self.choices["tts_engines"], on_change=self._select_tts_engine)
        self._combo(tts, 3, "Agent A audio persona", "agent_a_audio_persona", self.choices["agent_a_audio_personas"], on_change=self._select_audio_persona)
        self._combo(tts, 4, "Agent B audio persona", "agent_b_audio_persona", self.choices["agent_b_audio_personas"], on_change=self._select_audio_persona)
        self._check(tts, 5, "Play synthesized speech", "speech_playback_enabled")
        self._number(tts, 6, "Maximum utterance seconds", "max_utterance_sec", 5, 40, 1)
        self.dynamic_frames["tts"] = self._collapsible(tts, 7, "Implementation settings")

        asr = cards["asr"]
        self._combo(asr, 2, "Engine", "asr_engine", self.choices["asr_engines"], on_change=self._select_asr_engine)
        self._entry(asr, 3, "Language", "asr_language")
        self.dynamic_frames["asr"] = self._collapsible(asr, 4, "Implementation settings")

        nlu = cards["nlu"]
        self._check(nlu, 2, "Normalize transit terms", "asr_domain_normalization_enabled")
        self._number(nlu, 3, "Normalization similarity", "asr_domain_similarity_threshold", 0.70, 1.0, 0.01)

        dialogue = cards["dialogue_management"]
        self._number(dialogue, 2, "Maximum turns", "num_turns", 1, 100)
        self._number(dialogue, 3, "Invalid route limit", "invalid_route_limit", 1, 20)
        self._number(dialogue, 4, "Constraint miss limit", "constraint_miss_limit", 1, 20)
        self._number(dialogue, 5, "Clarification attempts", "clarification_max_attempts", 1, 6)
        self._number(dialogue, 6, "Progressive constraints", "maximum_progressive_constraints", 0, 6)
        self._number(dialogue, 7, "Routes compared", "minimum_compared_routes", 1, 10)
        self._check(dialogue, 8, "Retain earlier constraints", "require_constraint_retention")

        results = cards["results"]
        self._entry(results, 2, "Results root", "protocol_log_dir")
        self._check(results, 3, "Pair audio with text control", "paired_audio_text_runs")
        self._number(results, 4, "Interface font size", "gui_font_size", 9, 16)
        self._build_logging_controls(results, 5)

        for key, card in cards.items():
            self._attach_phase_metrics(card, key, 80)

    def _combined_phase_card(self, parent, index, key, title):
        card = ttk.LabelFrame(parent, text=f"{index + 1}. {title}", padding=7, style="Pipeline.TLabelframe")
        card.grid(row=index // 2, column=index % 2, sticky="nsew", padx=3, pady=3)
        card.columnconfigure(1, weight=1)
        status = ttk.Label(card, textvariable=self.pipeline_summary_vars[key], wraplength=680, justify="left")
        status.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 5))
        self.phase_status_widgets[key] = status
        ttk.Separator(card).grid(row=1, column=0, columnspan=3, sticky="ew", pady=(0, 4))
        return card

    def _attach_phase_metrics(self, card, phase_key, row):
        ttk.Separator(card).grid(row=row, column=0, columnspan=3, sticky="ew", pady=(7, 4))
        host = ttk.Frame(card)
        host.grid(row=row + 1, column=0, columnspan=3, sticky="ew")
        host.columnconfigure(0, weight=1)
        ttk.Label(host, textvariable=self.phase_metric_summary_vars[phase_key]).grid(row=0, column=0, sticky="w")
        toggle = ttk.Button(
            host,
            text="Metrics",
            command=lambda target=host, key=phase_key: self._expand_phase_metrics(target, key),
        )
        toggle.grid(row=0, column=1, sticky="e", padx=(8, 0))
        self.phase_metric_panels[phase_key] = {"host": host, "toggle": toggle, "viewport": None}

    def _expand_phase_metrics(self, host, phase_key):
        panel = self.phase_metric_panels[phase_key]
        if panel["viewport"] is not None:
            panel["viewport"].grid()
            panel["toggle"].configure(
                text="Close metrics",
                command=lambda key=phase_key: self._collapse_phase_metrics(key),
            )
            return

        viewport = ttk.Frame(host)
        viewport.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(5, 0))
        viewport.rowconfigure(0, weight=1)
        viewport.columnconfigure(0, weight=1)
        canvas = tk.Canvas(viewport, height=240, highlightthickness=0)
        scrollbar = ttk.Scrollbar(viewport, orient="vertical", command=canvas.yview)
        content = ttk.Frame(canvas, padding=(2, 0, 5, 2))
        content.columnconfigure(0, weight=1)
        window = canvas.create_window((0, 0), window=content, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        content.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(window, width=event.width))
        canvas.bind("<MouseWheel>", lambda event: canvas.yview_scroll(int(-event.delta / 120), "units"))
        canvas.bind("<Button-4>", lambda _event: canvas.yview_scroll(-1, "units"))
        canvas.bind("<Button-5>", lambda _event: canvas.yview_scroll(1, "units"))

        row = 0
        family_keys = set(GUI_METRIC_FAMILIES[phase_key])
        for family in METRIC_FAMILY_SPECS:
            if family["key"] not in family_keys:
                continue
            ttk.Label(content, text=family["title"], style="Pipeline.TLabelframe.Label").grid(
                row=row, column=0, columnspan=3, sticky="w", pady=(4, 2)
            )
            row += 1
            for metric_key, label in family["metrics"]:
                self._build_metric_row(content, row, metric_key, label)
                row += 1
        panel["viewport"] = viewport
        panel["toggle"].configure(
            text="Close metrics",
            command=lambda key=phase_key: self._collapse_phase_metrics(key),
        )
        self._schedule_pipeline_refresh()

    def _collapse_phase_metrics(self, phase_key):
        panel = self.phase_metric_panels[phase_key]
        panel["viewport"].grid_remove()
        panel["toggle"].configure(
            text="Metrics",
            command=lambda target=panel["host"], key=phase_key: self._expand_phase_metrics(target, key),
        )

    def _build_logging_controls(self, parent, row):
        logging_body = self._collapsible(parent, row, "Logged data overview")
        self.logging_overview = tk.Text(logging_body, height=8, wrap="word", state="disabled", borderwidth=0)
        self.logging_overview.grid(row=0, column=0, columnspan=2, sticky="nsew")
        dependency_body = self._collapsible(parent, row + 1, "Metric-data dependencies")
        dependency_body.columnconfigure(0, weight=1)
        self.dependency_loaded = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            dependency_body, text="Load dependency matrix", variable=self.dependency_loaded,
            command=self._schedule_pipeline_refresh,
        ).grid(row=0, column=0, sticky="w")
        self.dependency_tree = ttk.Treeview(dependency_body, show="tree headings", height=12)
        self.dependency_tree.heading("#0", text="Logged data field")
        self.dependency_tree.column("#0", width=260, stretch=False)
        self.dependency_scrollbar = ttk.Scrollbar(
            dependency_body, orient="horizontal", command=self.dependency_tree.xview
        )
        self.dependency_tree.configure(xscrollcommand=self.dependency_scrollbar.set)
        self.dependency_tree.grid(row=1, column=0, sticky="nsew")
        self.dependency_scrollbar.grid(row=2, column=0, sticky="ew")
        self.dependency_tree.grid_remove()
        self.dependency_scrollbar.grid_remove()
    def _vertical_pipeline(self, parent):
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)
        canvas = tk.Canvas(parent, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        content = ttk.Frame(canvas, padding=4)
        content.columnconfigure(0, weight=1)
        window = canvas.create_window((0, 0), window=content, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        content.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(window, width=event.width))
        canvas.bind("<MouseWheel>", lambda event: canvas.yview_scroll(int(-event.delta / 120), "units"))
        return content

    def _collapsible(self, parent, row, title):
        host = ttk.Frame(parent)
        host.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        host.columnconfigure(0, weight=1)
        body = ttk.Frame(host)
        body.columnconfigure(1, weight=1)
        visible = tk.BooleanVar(value=False)
        def toggle():
            if visible.get():
                body.grid(row=1, column=0, sticky="ew")
            else:
                body.grid_remove()
        ttk.Checkbutton(host, text=title, variable=visible, command=toggle).grid(row=0, column=0, sticky="w")
        body.grid(row=1, column=0, sticky="ew")
        body.grid_remove()
        return body

    def _schedule_pipeline_refresh(self, *_args):
        pending = getattr(self, "_pipeline_refresh_job", None)
        if pending:
            self.root.after_cancel(pending)
        self._pipeline_refresh_job = self.root.after(120, self._refresh_pipeline_overview)

    def _current_config_snapshot(self):
        config = {}
        for key, variable in self.vars.items():
            try:
                config[key] = variable.get()
            except tk.TclError:
                pass
        config["metric_config"] = {key: variable.get() for key, variable in self.metric_vars.items()}
        return config

    def _refresh_pipeline_overview(self):
        self._pipeline_refresh_job = None
        config = self._current_config_snapshot()
        tts = component_status("tts", config.get("tts_engine"), config)
        asr = component_status("asr", config.get("asr_engine"), config)
        model = component_status("model", config.get("model_provider"), config)
        if str(config.get("agent_b_plugin", "")).lower() not in {"", "llm", "minillama"} and str(config.get("agent_a_type", "")).lower() != "userlm":
            model = type(model)(model.key, True, "No language model required")
        report = metric_dependency_report(config)
        enabled = [item for item in report["metrics"].values() if item["enabled"]]
        unavailable = [item for item in enabled if not item["available"]]
        for key, widget in self.metric_enabled_widgets.items():
            if report["metrics"][key]["available"]:
                self._refresh_metric_state(key)
            else:
                widget.state(["disabled"])
        summaries = {
            "network_task": "",
            "agent_a": "",
            "agent_b_nlg": "" if model.available else f"Unavailable: {model.reason}",
            "tts": "" if tts.available else f"Unavailable: {tts.reason}",
            "asr": "" if asr.available else f"Unavailable: {asr.reason}",
            "nlu": "",
            "dialogue_management": "",
            "results": "",
        }
        for key, value in summaries.items():
            self.pipeline_summary_vars[key].set(value)
            if value:
                self.phase_status_widgets[key].grid()
            else:
                self.phase_status_widgets[key].grid_remove()
        for phase_key, family_keys in GUI_METRIC_FAMILIES.items():
            phase_metrics = [
                item for item in report["metrics"].values()
                if item["phase"] in family_keys
            ]
            phase_enabled = sum(item["enabled"] for item in phase_metrics)
            phase_blocked = sum(item["enabled"] and not item["available"] for item in phase_metrics)
            self.phase_metric_summary_vars[phase_key].set(
                f"Metrics: {phase_enabled} enabled" + (f", {phase_blocked} blocked" if phase_blocked else "")
            )
        self.optimal_route_text.set(optimal_route_preview(config)["summary"])
        lines = []
        collected = report["collected_fields"]
        for phase, fields in LOGGED_DATA_FIELDS.items():
            labels = [f"{'[x]' if key in collected else '[ ]'} {label}" for key, label in fields]
            lines.append(f"{phase.replace('_', ' ').title()}:\n  " + "\n  ".join(labels))
        if unavailable:
            lines.append("Warnings:\n  " + "\n  ".join(
                f"{item['label']}: missing {', '.join(FIELD_LABELS.get(key, key) for key in item['missing_fields'])}"
                for item in unavailable
            ))
        self.logging_overview.configure(state="normal")
        self.logging_overview.delete("1.0", "end")
        self.logging_overview.insert("1.0", "\n\n".join(lines))
        self.logging_overview.configure(state="disabled")
        if self.dependency_loaded.get():
            self.dependency_tree.grid()
            self.dependency_scrollbar.grid()
            self._refresh_dependency_tree(report)
        else:
            self.dependency_tree.grid_remove()
            self.dependency_scrollbar.grid_remove()

    def _refresh_dependency_tree(self, report):
        tree = self.dependency_tree
        metrics = [key for key, item in report["metrics"].items() if item["enabled"]]
        tree.configure(columns=metrics)
        for key in metrics:
            item = report["metrics"][key]
            tree.heading(key, text=("OK " if item["available"] else "BLOCKED ") + item["label"])
            tree.column(key, width=150, anchor="center", stretch=False)
        tree.delete(*tree.get_children())
        required = {key: set(report["metrics"][key]["required_fields"]) for key in metrics}
        for phase, fields in LOGGED_DATA_FIELDS.items():
            parent = tree.insert("", "end", text=phase.replace("_", " ").title(), open=True)
            for field, label in fields:
                values = ["required" if field in required[key] else "" for key in metrics]
                status = "collected" if field in report["collected_fields"] else "missing"
                tree.insert(parent, "end", text=f"{label} [{status}]", values=values)

    def _help(self, widget, key, text=None):
        base_key = key.removeprefix("agent_a_").removeprefix("agent_b_")
        ToolTip(
            widget,
            text
            or SETTING_HELP.get(key)
            or PROSODY_HELP.get(base_key)
            or f"Configures {key.replace('_', ' ')}.",
        )

    def _combo(self, parent, row, label, key, values, editable=False, on_change=None):
        label_widget = ttk.Label(parent, text=label)
        label_widget.grid(row=row, column=0, sticky="w", padx=(0, 6), pady=2)
        widget = ttk.Combobox(
            parent,
            textvariable=self.vars[key],
            values=list(values),
            state="normal" if editable else "readonly",
        )
        widget.grid(row=row, column=1, sticky="ew", pady=2)
        if on_change:
            widget.bind("<<ComboboxSelected>>", lambda _event: on_change())
        self._help(label_widget, key)
        self._help(widget, key)
        return widget

    def _number(self, parent, row, label, key, minimum, maximum, increment=1):
        minimum, maximum, increment = numeric_range(key, (minimum, maximum, increment))
        value_var = self.vars[key]
        slider_var = tk.DoubleVar(value=float(value_var.get()))

        def snap(value):
            raw = float(value)
            snapped = round((raw - minimum) / increment) * increment + minimum
            snapped = min(max(snapped, minimum), maximum)
            if isinstance(value_var, tk.IntVar):
                value_var.set(int(round(snapped)))
            else:
                value_var.set(round(snapped, 4))

        def sync_slider(*_args):
            try:
                slider_var.set(float(value_var.get()))
            except (TypeError, tk.TclError, ValueError):
                return

        label_widget = ttk.Label(parent, text=label)
        label_widget.grid(row=row, column=0, sticky="w", padx=(0, 6), pady=2)
        control = ttk.Frame(parent)
        control.grid(row=row, column=1, sticky="ew", pady=2)
        control.columnconfigure(0, weight=0)
        control.columnconfigure(1, weight=1)
        widget = ttk.Spinbox(
            control,
            textvariable=value_var,
            from_=minimum,
            to=maximum,
            increment=increment,
            width=10,
        )
        widget.grid(row=0, column=0, sticky="w")
        slider = ttk.Scale(
            control,
            from_=minimum,
            to=maximum,
            variable=slider_var,
            command=snap,
            orient="horizontal",
        )
        slider.grid(row=0, column=1, sticky="ew", padx=(6, 0))
        value_var.trace_add("write", sync_slider)
        self._help(label_widget, key)
        self._help(widget, key)
        self._help(slider, key)
        return widget

    def _entry(self, parent, row, label, key, show=None):
        label_widget = ttk.Label(parent, text=label)
        label_widget.grid(row=row, column=0, sticky="w", padx=(0, 6), pady=2)
        widget = ttk.Entry(parent, textvariable=self.vars[key], show=show)
        widget.grid(row=row, column=1, sticky="ew", pady=2)
        self._help(label_widget, key)
        self._help(widget, key)
        return widget

    def _check(self, parent, row, label, key):
        widget = ttk.Checkbutton(parent, text=label, variable=self.vars[key])
        widget.grid(row=row, column=0, columnspan=2, sticky="w", pady=2)
        self._help(widget, key)
        return widget

    @staticmethod
    def _clear_frame(frame):
        for child in frame.winfo_children():
            child.destroy()

    def _refresh_agent_selection(self):
        self._refresh_conditional_sections()

    def _refresh_speech_selection(self):
        self._refresh_conditional_sections()

    def _select_audio_persona(self):
        self._schedule_pipeline_refresh()

    def _select_tts_engine(self):
        selected = self.vars["tts_engine"].get()
        self._tts_model_by_engine[self._last_tts_engine] = self.vars["tts_model"].get()
        defaults = speech_engine_profile("tts", selected)
        self.vars["tts_model"].set(
            self._tts_model_by_engine.get(selected, defaults.get("tts_model", ""))
        )
        self.vars["tts_executable"].set(defaults.get("tts_executable", ""))
        self.vars["tts_python_executable"].set(defaults.get("tts_python_executable", ""))
        if "tts_timeout_sec" in defaults:
            self.vars["tts_timeout_sec"].set(defaults["tts_timeout_sec"])
        self._last_tts_engine = selected
        self._refresh_speech_selection()

    def _select_asr_engine(self):
        engine = self.vars["asr_engine"].get()
        self._asr_model_by_engine[self._last_asr_engine] = self.vars["asr_model"].get()
        defaults = speech_engine_profile("asr", engine)
        self.vars["asr_model"].set(
            self._asr_model_by_engine.get(engine, defaults.get("asr_model", ""))
        )
        self.vars["asr_executable"].set(defaults.get("asr_executable", ""))
        self.vars["asr_python_executable"].set(defaults.get("asr_python_executable", ""))
        self._last_asr_engine = engine
        self._refresh_speech_selection()

    def _refresh_conditional_sections(self):
        if not self.dynamic_frames:
            return
        self._rebuild_model_settings()
        self._rebuild_tts_settings()
        self._rebuild_asr_settings()

    def _rebuild_model_settings(self):
        frame = self.dynamic_frames.get("model")
        if frame is None:
            return
        self._clear_frame(frame)
        needs_model = (
            self.vars["agent_b_plugin"].get().strip().lower() in {"", "llm", "minillama"}
            or self.vars["agent_a_type"].get().strip().lower() == "userlm"
        )
        if not needs_model:
            self.agent_b_model_controls.grid_remove()
            self.agent_b_model_advanced.grid_remove()
            return
        self.agent_b_model_controls.grid()
        self.agent_b_model_advanced.grid()
        provider = self.vars["model_provider"].get()
        if provider == "transformers":
            self._entry(frame, 0, "Inference device", "model_device")
            self._number(frame, 1, "Maximum response tokens", "model_max_new_tokens", 8, 512, 8)
            ttk.Label(frame, text="Offline only: model and tokenizer must already be cached.", wraplength=270).grid(
                row=2, column=0, columnspan=2, sticky="w"
            )
        elif provider == "openai_compatible":
            self._entry(frame, 1, "Service URL", "model_base_url")
            self._entry(frame, 2, "API key", "model_api_key", show="*")
            self._number(frame, 3, "Request timeout seconds", "model_timeout_sec", 1, 300, 1)
            self._number(frame, 4, "Maximum response tokens", "model_max_new_tokens", 8, 512, 8)
        elif provider == "ollama":
            self._entry(frame, 1, "Ollama API URL", "model_base_url")
            self._check(frame, 2, "Start local Ollama service when needed", "model_service_autostart")
            self._number(frame, 3, "Request timeout seconds", "model_timeout_sec", 1, 300, 1)
            self._number(frame, 4, "Maximum response tokens", "model_max_new_tokens", 8, 512, 8)

    def _rebuild_tts_settings(self):
        frame = self.dynamic_frames.get("tts")
        if frame is None:
            return
        self._clear_frame(frame)
        engine = self.vars["tts_engine"].get()
        if engine == "chattts":
            self._entry(frame, 0, "Local ChatTTS assets", "tts_model")
            self._entry(frame, 1, "Inference device", "tts_device")
            self._number(frame, 2, "Agent A speaker seed", "agent_a_seed", 0, 2147483647)
            self._number(frame, 3, "Agent B speaker seed", "agent_b_seed", 0, 2147483647)
            self._number(frame, 4, "Agent A variation", "agent_a_temperature", 0.01, 1.0, 0.05)
            self._number(frame, 5, "Agent B variation", "agent_b_temperature", 0.01, 1.0, 0.05)
            self._number(frame, 6, "Agent A sampling range", "agent_a_top_p", 0.05, 1.0, 0.05)
            self._number(frame, 7, "Agent B sampling range", "agent_b_top_p", 0.05, 1.0, 0.05)
        elif engine == "piper":
            self._entry(frame, 0, "Piper voice model path", "tts_model")
            self._entry(frame, 1, "Inference device", "tts_device")
        elif engine == "espeak_ng":
            self._entry(frame, 0, "Optional eSpeak NG executable", "tts_executable")
        elif engine == "coqui":
            self._entry(frame, 0, "Coqui model name or path", "tts_model")
            self._entry(frame, 1, "Inference device", "tts_device")
        elif engine == "sapi":
            ttk.Label(
                frame,
                text="Voice, pace, volume, and pauses come from the two selected audio personas.",
                wraplength=430,
                justify="left",
            ).grid(row=0, column=0, columnspan=2, sticky="ew")
        else:
            ttk.Label(
                frame,
                text="The deterministic WAV implementation needs no provider-specific settings.",
                wraplength=430,
                justify="left",
            ).grid(row=0, column=0, columnspan=2, sticky="ew")

    def _rebuild_asr_settings(self):
        frame = self.dynamic_frames.get("asr")
        if frame is None:
            return
        self._clear_frame(frame)
        engine = self.vars["asr_engine"].get()
        if engine == "sapi":
            self._entry(frame, 0, "Recognition language", "asr_language")
            self._number(frame, 1, "End-of-speech pause milliseconds", "asr_end_silence_ms", 500, 6000, 100)
            self._number(frame, 2, "Ambiguous speech pause milliseconds", "asr_ambiguous_end_silence_ms", 1000, 8000, 100)
            next_row = 3
        elif engine == "faster_whisper":
            self._entry(frame, 0, "Whisper model", "asr_model")
            self._entry(frame, 1, "Inference device", "asr_device")
            self._entry(frame, 2, "Compute type", "asr_compute_type")
            self._number(frame, 3, "Recognition beam size", "asr_beam_size", 1, 20)
            self._entry(frame, 4, "Recognition language", "asr_language")
            next_row = 5
        elif engine == "vosk":
            self._entry(frame, 0, "Local Vosk model directory", "asr_model")
            self._entry(frame, 1, "Recognition language", "asr_language")
            next_row = 2
        elif engine == "whisper_cpp":
            self._entry(frame, 0, "whisper.cpp model path", "asr_model")
            self._entry(frame, 1, "whisper-cli executable", "asr_executable")
            self._entry(frame, 2, "Optional voice activity model", "asr_vad_model")
            self._entry(frame, 3, "Recognition language", "asr_language")
            next_row = 4
        elif engine == "qwen3_asr":
            self._entry(frame, 0, "Qwen3-ASR model", "asr_model")
            self._entry(frame, 1, "Inference device", "asr_device")
            self._entry(frame, 2, "Recognition language", "asr_language")
            next_row = 3
        elif engine == "sherpa_onnx":
            self._entry(frame, 0, "sherpa-onnx model directory", "asr_model")
            self._entry(frame, 1, "Recognition language", "asr_language")
            next_row = 2
        else:
            ttk.Label(
                frame,
                text="The deterministic recognizer reads the transcript attached to the generated WAV.",
                wraplength=430,
                justify="left",
            ).grid(row=0, column=0, columnspan=2, sticky="ew")
            next_row = 1
        self._check(frame, next_row, "Repair close transit terms", "asr_domain_normalization_enabled")
        self._number(
            frame,
            next_row + 1,
            "Domain repair similarity",
            "asr_domain_similarity_threshold",
            0.70,
            1.0,
            0.01,
        )

    def _refresh_persona_detail(self):
        persona = get_persona(self.vars["persona_key"].get())
        self.persona_detail.set(f"{persona['description']} {preference_text(persona)}")

    def _select_model_provider(self):
        defaults = model_provider_defaults(self.vars["model_provider"].get())
        self.vars["model_name"].set(defaults["model_name"])
        self.vars["model_base_url"].set(defaults["model_base_url"])
        self.vars["model_timeout_sec"].set(defaults["model_timeout_sec"])
        self._refresh_conditional_sections()

    def _build_metric_row(self, group, row, key, label):
        label_widget = ttk.Label(group, text=label)
        label_widget.grid(row=row, column=0, sticky="w", pady=1)
        enabled = ttk.Checkbutton(
            group, variable=self.metric_vars[key],
            command=lambda metric_key=key: self._refresh_metric_state(metric_key),
        )
        enabled.grid(row=row, column=1, sticky="w", padx=(8, 0))
        tier = ttk.Combobox(
            group, textvariable=self.metric_tier_vars[key],
            values=("core", "supplementary"), state="readonly", width=14,
        )
        tier.grid(row=row, column=2, sticky="w", padx=(8, 0))
        tier.bind("<<ComboboxSelected>>", lambda _event, metric_key=key: self._refresh_metric_state(metric_key))
        self.metric_enabled_widgets[key] = enabled
        tooltip = (
            f"{label}: {metric_calculation_method(key)}. Core forces the metric on; "
            "supplementary metrics are optional."
        )
        for widget in (label_widget, enabled, tier):
            ToolTip(widget, tooltip)
        self._refresh_metric_state(key)

    def _refresh_metric_state(self, metric_key):
        """Apply the core/supplementary rule to one metric row."""
        tier = self.metric_tier_vars[metric_key].get().strip().lower()
        widget = self.metric_enabled_widgets.get(metric_key)
        if tier == "core":
            self.metric_vars[metric_key].set(True)
            if widget is not None:
                widget.state(["disabled"])
        elif widget is not None:
            widget.state(["!disabled"])

    def start(self):
        selected = {key: variable.get() for key, variable in self.vars.items()}
        selected["metric_config"] = {
            key: variable.get() for key, variable in self.metric_vars.items()
        }
        selected["metric_tiers"] = {
            key: variable.get() for key, variable in self.metric_tier_vars.items()
        }
        try:
            self.result = self.validator(selected) if self.validator else selected
        except Exception as exc:
            self.result = None
            messagebox.showerror(
                "Configuration cannot start",
                f"{exc}\n\nChange the highlighted implementation settings and try again.",
                parent=self.root,
            )
            return
        self.root.destroy()

    def cancel(self):
        self.result = None
        self.root.destroy()

    def show(self):
        self.root.mainloop()
        return self.result
