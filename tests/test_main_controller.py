from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO
import inspect
from types import SimpleNamespace
import unittest

from coop_navigation_sds.app import (
    ConsoleEventSink,
    configure_model_adapter_runtime,
    default_run_config,
    normalize_run_config,
)
from coop_navigation_sds.Configuration.gui import StartupConfigDialog
from coop_navigation_sds.Configuration.component_catalog import startup_choices
from coop_navigation_sds.Configuration.experimental_defaults import numeric_range
from coop_navigation_sds.DialogManagement.speech_pipeline import SpeechPipelineError


class MainControllerTests(unittest.TestCase):
    def test_local_generation_budget_does_not_replace_service_timeout(self):
        adapter = SimpleNamespace(max_time_sec=None, timeout_sec=180.0)

        configure_model_adapter_runtime(adapter, 5.0)

        self.assertEqual(adapter.max_time_sec, 5.0)
        self.assertEqual(adapter.timeout_sec, 180.0)

    def test_console_defers_dialog_stages_until_post_run_artifacts(self):
        sink = ConsoleEventSink()
        output = StringIO()

        with redirect_stdout(output):
            sink.put(("stage", "proposal"))
            sink.put(("stage", "proposal"))
            sink.put(("stage", "comparison"))

        text = output.getvalue()
        self.assertEqual(text, "")

    def test_metric_tier_core_forces_enabled_and_supplementary_is_optional(self):
        class Variable:
            def __init__(self, value):
                self.value = value

            def get(self):
                return self.value

            def set(self, value):
                self.value = value

        class Widget:
            def __init__(self):
                self.states = set()

            def state(self, states):
                for state in states:
                    if state.startswith("!"):
                        self.states.discard(state[1:])
                    else:
                        self.states.add(state)

        dialog = StartupConfigDialog.__new__(StartupConfigDialog)
        widget = Widget()
        dialog.metric_vars = {"asr_wer": Variable(False)}
        dialog.metric_tier_vars = {"asr_wer": Variable("core")}
        dialog.metric_enabled_widgets = {"asr_wer": widget}

        dialog._refresh_metric_state("asr_wer")
        self.assertTrue(dialog.metric_vars["asr_wer"].get())
        self.assertIn("disabled", widget.states)

        dialog.metric_tier_vars["asr_wer"].set("supplementary")
        dialog._refresh_metric_state("asr_wer")
        self.assertNotIn("disabled", widget.states)

    def test_run_scripts_exist_for_dialog_and_script_config_modes(self):
        root = Path(__file__).resolve().parents[1]
        self.assertTrue((root / "scripts" / "run_with_config_dialog.py").exists())
        self.assertTrue((root / "scripts" / "run_from_script_config.py").exists())

    def test_default_config_is_full_speech_without_runtime_gui(self):
        config = default_run_config()
        self.assertNotIn("gui_enabled", config)
        self.assertNotIn("run_mode", config)
        self.assertNotIn("speech_scope", config)
        self.assertEqual(config["tts_engine"], "sapi")
        self.assertEqual(config["asr_engine"], "sapi")
        self.assertTrue(config["speech_playback_enabled"])
        self.assertTrue(config["speech_realtime_enabled"])
        self.assertEqual(config["agent_a_type"], "staged")
        self.assertEqual(config["agent_a_audio_persona"], "high_clarity_caller")
        self.assertEqual(config["agent_b_audio_persona"], "high_clarity_operator")
        self.assertGreaterEqual(config["asr_end_silence_ms"], 1500)
        self.assertEqual(config["asr_beam_size"], 8)
        self.assertEqual(config["agent_a_speech_rate"], -3)
        self.assertEqual(config["agent_a_words_per_minute"], 140)
        self.assertEqual(config["agent_b_pause_ms"], 280)
        self.assertGreaterEqual(config["model_max_new_tokens"], 96)
        self.assertEqual(config["max_utterance_sec"], 20.0)
        self.assertEqual(config["agent_b_volume"], 100)
        self.assertIn(config["agent_a_emphasis"], {"none", "reduced", "moderate", "strong"})
        self.assertTrue(config["metric_config"]["task_outcome_completion"])

    def test_metric_switches_are_normalized(self):
        config = default_run_config()
        config["metric_config"] = {"asr_wer": False}
        normalized = normalize_run_config(config)
        self.assertTrue(normalized["metric_config"]["asr_wer"])
        self.assertTrue(normalized["metric_config"]["task_outcome_completion"])

    def test_runtime_lengths_are_clamped(self):
        config = default_run_config()
        config["max_turn_elapsed_sec"] = 99.0
        config["calculation_max_time_sec"] = 99.0
        normalized = normalize_run_config(config)
        self.assertEqual(normalized["max_turn_elapsed_sec"], 20.0)
        self.assertEqual(normalized["calculation_max_time_sec"], 20.0)

    def test_legacy_agent_a_boolean_maps_to_userlm(self):
        normalized = normalize_run_config({"llm_agent_a": True})
        self.assertEqual(normalized["agent_a_type"], "userlm")
        self.assertTrue(normalized["llm_agent_a"])

    def test_named_audio_personas_override_legacy_custom_controls(self):
        config = default_run_config()
        config.update({
            "agent_a_custom_audio": True,
            "agent_b_custom_audio": True,
            "agent_a_speech_rate": 99,
            "agent_a_volume": -5,
            "agent_b_pitch_semitones": 40,
            "agent_b_pause_ms": 9000,
            "agent_b_emphasis": "invalid",
        })
        normalized = normalize_run_config(config)
        defaults = default_run_config()
        self.assertFalse(normalized["agent_a_custom_audio"])
        self.assertFalse(normalized["agent_b_custom_audio"])
        self.assertEqual(normalized["agent_a_speech_rate"], defaults["agent_a_speech_rate"])
        self.assertEqual(normalized["agent_a_volume"], defaults["agent_a_volume"])
        self.assertEqual(normalized["agent_b_pause_ms"], defaults["agent_b_pause_ms"])

    def test_text_bypass_engines_are_rejected(self):
        config = default_run_config()
        config["tts_engine"] = "loopback"
        with self.assertRaises(SpeechPipelineError):
            normalize_run_config(config)

    def test_console_prints_spoken_and_understood_text_plus_metrics(self):
        sink = ConsoleEventSink()
        metric = SimpleNamespace(
            success=True,
            automatic_eval_score=0.9,
            quality_score=0.8,
            pipeline_success_rate=1.0,
            mean_turn_latency_sec=0.4,
            metric_families={
                "asr": {
                    "available": True,
                    "wer": 0.1,
                }
            },
        )
        output = StringIO()

        with redirect_stdout(output):
            sink.put(("telemetry", "speech", {
                "speaker": "Agent A",
                "outgoing_text": "I need Alpha to Echo.",
                "incoming_transcript": "I need Alpha to Echo.",
            }))
            sink.put(("message", "Agent A", "I need Alpha to Echo."))
            sink.put(("metrics", "Messages: 1\nRoute correct: True"))
            sink.put(("metric_results", metric))

        text = output.getvalue()
        self.assertIn("SPEECH: Agent A: I need Alpha to Echo.", text)
        self.assertIn("ASR RAW: Agent B: I need Alpha to Echo.", text)
        self.assertIn("MISINTERPRETATIONS: none", text)
        self.assertIn("CORRECTIONS: none", text)
        self.assertIn("UNDERSTOOD: Agent B: I need Alpha to Echo.", text)
        self.assertEqual(text.count("I need Alpha to Echo."), 3)
        self.assertIn("Conversation And Task Summary", text)
        self.assertIn("[2. Automatic Speech Recognition]", text)
        self.assertIn("WER = 0.1000", text)
        self.assertIn("Formula:", text)

    def test_console_shows_raw_errors_corrections_and_listener_input(self):
        sink = ConsoleEventSink()
        output = StringIO()
        with redirect_stdout(output):
            sink.put(("telemetry", "speech", {
                "speaker": "Agent B",
                "outgoing_text": "Take metro line M1 to Harbor.",
                "raw_asr_transcript": "Take metro line em one to harder.",
                "incoming_transcript": "Take metro line M1 to Harbor.",
                "misinterpreted_tokens": [
                    {"operation": "replace", "source_tokens": ["M1"], "target_tokens": ["em", "one"]},
                    {"operation": "replace", "source_tokens": ["Harbor"], "target_tokens": ["harder"]},
                ],
                "transcript_corrections": [
                    {"operation": "replace", "source_tokens": ["em", "one"], "target_tokens": ["M1"]},
                    {"operation": "replace", "source_tokens": ["harder"], "target_tokens": ["Harbor"]},
                ],
            }))

        text = output.getvalue()
        self.assertIn("ASR RAW: Agent A: Take metro line em one to harder.", text)
        self.assertIn("M1 -> em one", text)
        self.assertIn("em one -> M1", text)
        self.assertIn("UNDERSTOOD: Agent A: Take metro line M1 to Harbor.", text)

    def test_console_prints_retrospective_numeric_operands_and_steps(self):
        sink = ConsoleEventSink()
        metric = SimpleNamespace(
            metric_families={"asr": {"available": True, "wer": 0.25}},
            metric_calculations={
                "asr_wer": {
                    "formula": "(substitutions + deletions + insertions) / reference words",
                    "operands": {
                        "substitutions": 1,
                        "deletions": 0,
                        "insertions": 0,
                        "reference_words": 4,
                    },
                    "substitution": "(1 + 0 + 0) / 4 = 0.25",
                }
            },
        )
        output = StringIO()

        with redirect_stdout(output):
            sink.put(("metric_results", metric))

        text = output.getvalue()
        self.assertIn("substitutions=1", text)
        self.assertIn("Steps: (1 + 0 + 0) / 4 = 0.25", text)

    def test_console_defers_pipeline_phases_during_conversation(self):
        sink = ConsoleEventSink()
        output = StringIO()

        with redirect_stdout(output):
            sink.put(("phase", {
                "turn": 2,
                "speaker": "Agent B",
                "phase": "NLG",
                "text": "Take Alpha to Echo.",
                "latency_sec": 0.125,
            }))
            sink.put(("phase", {
                "turn": 2,
                "speaker": "Agent B",
                "phase": "TTS",
                "text": "Take Alpha to Echo.",
                "engine": "qwen3_tts",
                "latency_sec": 0.5,
            }))
            sink.put(("phase", {
                "turn": 2,
                "speaker": "Agent B",
                "phase": "ASR",
                "text": "Take Alpha to Echo.",
                "engine": "qwen3_asr",
                "latency_sec": 0.25,
            }))
            sink.put(("phase", {
                "turn": 2,
                "speaker": "Agent B",
                "phase": "NLU",
                "parsed_route": ["Alpha", "Echo"],
                "route_valid": True,
                "latency_sec": 0.01,
            }))

        text = output.getvalue()
        self.assertEqual(text, "")

    def test_console_defers_phase_timing_during_conversation(self):
        sink = ConsoleEventSink()
        output = StringIO()

        with redirect_stdout(output):
            sink.put(("telemetry", "phase_timing", {
                "turn": 2,
                "speaker": "Agent B",
                "natural_language_generation_sec": 0.12,
                "text_to_speech_processing_sec": 0.20,
                "audio_duration_sec": 1.50,
                "automatic_speech_recognition_processing_sec": 0.30,
                "natural_language_understanding_sec": 0.04,
                "dialogue_management_sec": 0.05,
                "speech_pipeline_wall_sec": 2.10,
                "observed_turn_sec": 2.31,
                "accounted_processing_sec": 2.31,
            }))

        text = output.getvalue()
        self.assertEqual(text, "")

    def test_configuration_gui_combines_configuration_and_metrics_by_phase(self):
        source = inspect.getsource(StartupConfigDialog)

        self.assertIn('text="Experiment pipeline"', source)
        self.assertIn("_build_combined_pipeline", source)
        self.assertIn("_attach_phase_metrics", source)
        self.assertIn("_expand_phase_metrics", source)
        self.assertIn("_collapse_phase_metrics", source)
        self.assertIn('text="Close metrics"', source)
        self.assertIn('orient="vertical", command=canvas.yview', source)
        self.assertIn("_vertical_pipeline", source)
        self.assertNotIn("ttk.Notebook", source)
        self.assertIn('self.root.state("zoomed")', source)
        self.assertIn("self.root.after_idle(self._maximize_window)", source)
        self.assertIn("Core forces the metric on", source)
        self.assertIn("_refresh_metric_state", source)
        self.assertIn("_refresh_conditional_sections", source)
        self.assertIn("ttk.Scale", source)
        self.assertIn("numeric_range", source)
        self.assertIn('style.configure("TButton"', source)
        self.assertNotIn("laugh_level", source)
        self.assertNotIn("reference_audio", source)

    def test_component_catalog_exposes_plug_and_play_backends(self):
        choices = startup_choices()

        self.assertIn("llm", choices["agent_b_plugins"])
        self.assertIn("simple", choices["agent_b_plugins"])
        self.assertIn("openai_compatible", choices["model_providers"])
        self.assertIn("ollama", choices["model_providers"])
        self.assertIn("chattts", choices["tts_engines"])
        self.assertIn("whisper_cpp", choices["asr_engines"])
        self.assertEqual(numeric_range("asr_end_silence_ms", (0, 1, 1)), (500, 6000, 100))


if __name__ == "__main__":
    unittest.main()
