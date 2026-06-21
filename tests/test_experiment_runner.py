import csv
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import wave
from zipfile import ZipFile

from coop_navigation_sds.DialogManagement.result import DialogResult
from coop_navigation_sds.experiments import ExperimentCondition, ExperimentRunner, build_condition_grid, write_metrics_csv, write_metrics_file
from coop_navigation_sds.ResultsAndArtifacts.artifacts import create_execution_run_dir, write_conversation_protocol, write_conversation_protocols, write_experiment_manifest, write_metric_phase_logs, write_network_research_artifacts, write_single_run_research_outputs
from coop_navigation_sds.NaturalLanguageGeneration.models import ModelParameterSet


def write_test_wav(path, frame_rate=8000, seconds=0.05):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = int(frame_rate * seconds)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(frame_rate)
        handle.writeframes(b"\x00\x00" * frame_count)


class FakeModelAdapter:
    def __init__(self, name="fake-model", model_parameters=None):
        self.name = name
        self.device = "cpu"
        self.model_parameters = model_parameters or ModelParameterSet("greedy", do_sample=False)
        self.max_time_sec = None
        self.calls = []

    def with_model_params(self, model_param_key):
        self.calls.append(model_param_key)
        adapter = FakeModelAdapter(
            name=self.name,
            model_parameters=ModelParameterSet(
                model_param_key,
                do_sample=model_param_key != "greedy",
                temperature=0.7 if model_param_key == "temp0.7" else None,
                top_p=0.9 if model_param_key == "nucleus0.9" else None,
            ),
        )
        adapter.calls = self.calls
        return adapter


class ExperimentRunnerTests(unittest.TestCase):
    @patch("coop_navigation_sds.experiments.create_agent_b_plugin")
    @patch("coop_navigation_sds.experiments.DialogManager")
    @patch("coop_navigation_sds.experiments.get_test_case")
    def test_run_condition_applies_model_params_and_records_audit_values(
        self,
        get_test_case,
        dialog_manager_cls,
        create_agent_b_plugin,
    ):
        base_case = SimpleNamespace(
            key="case",
            name="Case",
            persona_key="focused_commuter",
            scenario_key="scenario",
            scenario={
                "name": "Scenario",
                "start_station": "A",
                "destination_station": "B",
                "start_time_min": 0,
                "transfer_time_min": 2,
            },
        )
        base_case.with_persona = lambda persona_key: base_case
        get_test_case.return_value = base_case

        model_adapter = FakeModelAdapter()
        runner = ExperimentRunner(
            model_adapter,
            num_turns=3,
            agent_b_plugin_key="simple",
            tts_engine="file",
            asr_engine="file",
            speech_playback_enabled=False,
            speech_realtime_enabled=False,
            max_turn_elapsed_sec=4.0,
            calculation_max_time_sec=4.5,
        )
        runner.metric_computer.compute = MagicMock(return_value="metric")

        result = SimpleNamespace(extra={})
        dialog_manager_cls.return_value.run.return_value = result

        condition = ExperimentCondition(
            condition_id="case__focused_commuter__clean__temp0.7__0",
            test_case_key="case",
            persona_key="focused_commuter",
            scenario_key="scenario",
            speech_pattern_key="clean",
            model_param_key="temp0.7",
            iteration=0,
        )

        returned_result, metric = runner.run_condition(condition)

        self.assertIs(returned_result, result)
        self.assertEqual(metric, "metric")
        self.assertEqual(model_adapter.calls, ["temp0.7"])
        create_agent_b_plugin.assert_called_once()
        self.assertEqual(create_agent_b_plugin.call_args.args[0], "simple")
        self.assertEqual(create_agent_b_plugin.call_args.args[1].model_parameters.temperature, 0.7)
        self.assertEqual(create_agent_b_plugin.call_args.args[1].max_time_sec, 4.5)
        self.assertEqual(dialog_manager_cls.call_args.kwargs["max_turn_elapsed_sec"], 4.0)
        self.assertEqual(result.condition_id, condition.condition_id)
        self.assertEqual(result.speech_pattern_key, "clean")
        self.assertEqual(result.extra["model_param_key"], "temp0.7")
        self.assertEqual(result.extra["iteration"], 0)
        self.assertIn("condition_runtime_sec", result.extra)
        self.assertEqual(result.extra["model_parameters"]["do_sample"], True)
        self.assertEqual(result.extra["model_parameters"]["temperature"], 0.7)

    @patch("coop_navigation_sds.experiments.create_agent_b_plugin")
    @patch("coop_navigation_sds.experiments.DialogManager")
    @patch("coop_navigation_sds.experiments.get_test_case")
    def test_run_condition_can_write_runtime_batch_logs(
        self,
        get_test_case,
        dialog_manager_cls,
        create_agent_b_plugin,
    ):
        base_case = SimpleNamespace(
            key="case",
            name="Case",
            persona_key="focused_commuter",
            scenario_key="scenario",
            scenario={
                "name": "Scenario",
                "start_station": "A",
                "destination_station": "B",
                "start_time_min": 0,
                "transfer_time_min": 2,
            },
        )
        base_case.with_persona = lambda persona_key: base_case
        get_test_case.return_value = base_case
        create_agent_b_plugin.return_value = SimpleNamespace(name="plugin")
        runner = ExperimentRunner(
            FakeModelAdapter(),
            num_turns=1,
            agent_b_plugin_key="simple",
            tts_engine="file",
            asr_engine="file",
            speech_playback_enabled=False,
            speech_realtime_enabled=False,
            log_profile="runtime",
            log_dir=tempfile.mkdtemp(),
        )
        runner.metric_computer.compute = MagicMock(return_value="metric")
        dialog_manager_cls.return_value.run.return_value = SimpleNamespace(extra={})

        condition = ExperimentCondition(
            condition_id="case__focused_commuter__clean__greedy__0",
            test_case_key="case",
            persona_key="focused_commuter",
            scenario_key="scenario",
            speech_pattern_key="clean",
            model_param_key="greedy",
            iteration=0,
        )

        runner.run_condition(condition)

        self.assertTrue(list(Path(runner.log_dir).glob("batch-*.jsonl")))

    def test_plugin_config_identifies_model_need(self):
        from coop_navigation_sds.NaturalLanguageGeneration.assistant.plugin_registry import AgentBPluginConfig

        self.assertTrue(AgentBPluginConfig("minillama").needs_model)
        self.assertTrue(AgentBPluginConfig("llm").needs_model)
        self.assertFalse(AgentBPluginConfig("simple").needs_model)

    @patch("coop_navigation_sds.experiments.get_test_case")
    def test_build_condition_grid_caches_test_case_lookups(self, get_test_case):
        get_test_case.side_effect = lambda key: SimpleNamespace(scenario_key=f"scenario:{key}")

        conditions = list(
            build_condition_grid(
                test_case_keys=["alpha", "alpha", "beta"],
                persona_keys=["p"],
                speech_pattern_keys=["clean"],
                model_param_keys=["greedy"],
                iterations=2,
            )
        )

        self.assertEqual(len(conditions), 6)
        self.assertEqual(get_test_case.call_count, 2)
        self.assertEqual(conditions[0].scenario_key, "scenario:alpha")
        self.assertEqual(conditions[-1].scenario_key, "scenario:beta")

    def test_write_metrics_csv_accepts_iterables(self):
        class FakeMetric:
            def __init__(self, row):
                self.row = row

            def as_dict(self):
                return self.row

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.csv"
            write_metrics_csv((FakeMetric({"a": 1, "b": 2}), FakeMetric({"a": 3, "b": 4})), path)

            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(rows, [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}])

    def test_write_metrics_xlsx_and_phase_logs(self):
        class FakeMetric:
            condition_id = "c1"
            test_case_key = "case"
            persona_key = "persona"
            scenario_key = "scenario"
            speech_pattern_key = "clean"
            model_name = "model"
            model_param_key = "greedy"
            metric_families = {
                "runtime": {"available": True, "response_latency_sec": 0.12},
                "asr": {"available": True, "word_error_rate": 0.0},
            }

            def as_dict(self):
                return {"condition_id": self.condition_id, "automatic_eval_score": 0.9}

        with tempfile.TemporaryDirectory() as tmpdir:
            xlsx_path = Path(tmpdir) / "metrics.xlsx"
            log_dir = Path(tmpdir)
            write_metrics_file([FakeMetric()], xlsx_path)
            write_metric_phase_logs([FakeMetric()], log_dir)

            with ZipFile(xlsx_path) as archive:
                names = set(archive.namelist())
                workbook_xml = archive.read("xl/workbook.xml").decode("utf-8")

            runtime_rows = (log_dir / "metric_phase_runtime.jsonl").read_text(encoding="utf-8").splitlines()
            runtime_record = json.loads(runtime_rows[0])

        self.assertIn("xl/workbook.xml", names)
        self.assertIn('name="summary"', workbook_xml)
        self.assertIn('name="runtime"', workbook_xml)
        self.assertEqual(runtime_record["phase"], "runtime")
        self.assertEqual(runtime_record["metrics"]["response_latency_sec"], 0.12)

    def test_write_network_research_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = write_network_research_artifacts(480, Path(tmpdir) / "research", Path(tmpdir) / "graphs")

            network_data = json.loads(paths["network_json"].read_text(encoding="utf-8"))
            graph_text = paths["network_graph"].read_text(encoding="utf-8")

        self.assertGreater(network_data["line_count"], 0)
        self.assertGreater(network_data["station_count"], 0)
        self.assertIn("<svg", graph_text)

    def test_create_execution_run_dir_uses_systematic_unique_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            first = create_execution_run_dir(tmpdir, label="Single Case", timestamp="20260531_120000")
            second = create_execution_run_dir(tmpdir, label="Single Case", timestamp="20260531_120000")

        self.assertEqual(first.name, "20260531_120000_single_case")
        self.assertEqual(second.name, "20260531_120000_single_case_02")

    def test_write_experiment_manifest_documents_scientific_design(self):
        condition = ExperimentCondition(
            condition_id="case__persona__clean__greedy__0",
            test_case_key="case",
            persona_key="persona",
            scenario_key="scenario",
            speech_pattern_key="clean",
            model_param_key="greedy",
            iteration=0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_experiment_manifest(
                [condition],
                tmpdir,
                num_turns=4,
                speech_engine="file",
                tts_engine="file",
                asr_engine="loopback",
                speech_scope="both",
                agent_b_plugin="simple",
            )
            manifest = json.loads(path.read_text(encoding="utf-8"))

        self.assertIn("hypotheses", manifest)
        self.assertIn("independent_variables", manifest)
        self.assertIn("agent_a_audio_persona", manifest["independent_variables"])
        self.assertIn("agent_b_audio_persona", manifest["independent_variables"])
        self.assertEqual(manifest["conditions"][0]["condition_id"], condition.condition_id)
        self.assertEqual(manifest["conditions"][0]["asr_engine"], "loopback")
        self.assertEqual(manifest["controls"]["speech_engine"], "file")
        self.assertEqual(manifest["controls"]["tts_engine"], "file")
        self.assertEqual(manifest["controls"]["asr_engine"], "loopback")

    def test_write_conversation_protocol_creates_verified_research_artifacts(self):
        result = DialogResult(
            condition_id="Case One",
            test_case_key="case",
            persona_key="persona",
            scenario_key="scenario",
            speech_pattern_key="clean",
            model_name="model",
            conversation=[("Agent A", "Need Alpha to Echo."), ("Agent B", "Take Alpha to Echo.")],
            route=["Alpha", "Echo"],
            route_steps=[{"from": "Alpha", "to": "Echo", "line": "Ring"}],
            route_valid=True,
            route_reaches_goal=True,
            route_correct=True,
            route_duration_min=12,
            runtime_sec=1.2,
            metrics_text="Messages: 2",
            extra={
                "messages": 2,
                "speech_turns": [
                    {
                        "speaker": "Agent A",
                        "generated_text": "Need Alpha to Echo.",
                        "outgoing_text": "Need Alpha to Echo.",
                        "incoming_transcript": "Need Alpha to Echo.",
                        "mode": "speech",
                        "pipeline_ok": True,
                    }
                ],
                "timing_turns": [{"turn_latency_sec": 0.4}],
                "phase_timings": [{
                    "turn": 1,
                    "speaker": "Agent A",
                    "natural_language_generation_sec": 0.1,
                    "text_to_speech_processing_sec": 0.1,
                    "audio_duration_sec": 0.2,
                    "automatic_speech_recognition_processing_sec": 0.1,
                    "natural_language_understanding_sec": None,
                    "dialogue_management_sec": None,
                    "speech_pipeline_wall_sec": 0.3,
                    "observed_turn_sec": 0.4,
                    "accounted_processing_sec": 0.4,
                }],
                "agent_turn_segments": [
                    {"turn": 1, "speaker": "Agent A", "turn_elapsed_sec": 0.1},
                    {"turn": 2, "speaker": "Agent B", "turn_elapsed_sec": 0.4},
                ],
                "agent_timing_summary": {
                    "Agent A": {"turn_count": 1, "mean_turn_elapsed_sec": 0.1},
                    "Agent B": {"turn_count": 1, "mean_turn_elapsed_sec": 0.4},
                },
                "nlu_turns": [{"route_valid": True, "route_reaches_goal": True}],
                "runtime_events": [{"phase": "preflight", "event_type": "viability_check", "payload": {}}],
                "preflight_viability": {"constraint_route_available": True},
                "conversation_outcome": "satisfied",
            },
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            turn_audio_path = Path(tmpdir) / "turn_audio" / "agent_a.wav"
            write_test_wav(turn_audio_path)
            result.extra["speech_turns"][0]["audio"] = {"path": str(turn_audio_path)}
            paths = write_conversation_protocol(result, tmpdir)
            protocol_children = [child for child in Path(tmpdir).iterdir() if child.is_dir() and child.name != "turn_audio"]
            batch_paths = write_conversation_protocols([result], Path(tmpdir) / "batch")

            summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
            combined_protocol = json.loads(paths["protocol"].read_text(encoding="utf-8"))
            audio_manifest = json.loads(paths["audio_manifest"].read_text(encoding="utf-8"))
            transcript_text = paths["transcript_txt"].read_text(encoding="utf-8")
            conversation_wav_exists = paths["conversation_wav"].exists()
            verification = json.loads(paths["verification"].read_text(encoding="utf-8"))
            turn_rows = paths["turns"].read_text(encoding="utf-8").splitlines()
            agent_rows = paths["agent_turn_segments"].read_text(encoding="utf-8").splitlines()
            agent_a_rows = paths["agent_a_segments"].read_text(encoding="utf-8").splitlines()
            agent_b_rows = paths["agent_b_segments"].read_text(encoding="utf-8").splitlines()
            agent_summary = json.loads(paths["agent_timing_summary"].read_text(encoding="utf-8"))
            runtime_rows = paths["runtime_events"].read_text(encoding="utf-8").splitlines()
            phase_timing_rows = paths["phase_timing"].read_text(encoding="utf-8").splitlines()
            retrospective_text = paths["retrospective_summary"].read_text(encoding="utf-8")

        self.assertTrue(batch_paths)
        self.assertEqual(protocol_children, [])
        self.assertEqual(summary["condition_id"], "Case One")
        self.assertEqual(combined_protocol["summary"]["condition_id"], "Case One")
        self.assertIn("Agent A", transcript_text)
        self.assertTrue(audio_manifest["created"])
        self.assertTrue(conversation_wav_exists)
        self.assertTrue(verification["verified"])
        self.assertEqual(len(turn_rows), 2)
        self.assertEqual(len(agent_rows), 2)
        self.assertEqual(len(agent_a_rows), 1)
        self.assertEqual(len(agent_b_rows), 1)
        self.assertEqual(agent_summary["Agent B"]["mean_turn_elapsed_sec"], 0.4)
        self.assertEqual(len(runtime_rows), 1)
        self.assertEqual(len(phase_timing_rows), 1)
        self.assertEqual(len(combined_protocol["phase_timing"]), 1)
        self.assertIn("processing seconds", transcript_text)
        self.assertIn("Messages", retrospective_text)

    def test_single_run_research_outputs_compile_metrics_for_analysis(self):
        result = DialogResult(
            condition_id="Case One",
            test_case_key="morning_peak_cross_city",
            persona_key="focused_commuter",
            scenario_key="morning_peak_cross_city",
            speech_pattern_key="clean",
            model_name="model",
            conversation=[("Agent A", "Need Bravo to Harbor."), ("Agent B", "Take Bravo to Alpha to Golf to November to Uniform to Birch to Ivy to Harbor.")],
            route=["Bravo", "Alpha", "Golf", "November", "Uniform", "Birch", "Ivy", "Harbor"],
            route_steps=[
                {"from": "Bravo", "to": "Alpha", "line": "Ring", "travel": 2, "wait": 0, "transfer": 0, "fullness": 43, "delay_probability": 0.2},
                {"from": "Alpha", "to": "Golf", "line": "Ring", "travel": 6, "wait": 0, "transfer": 0, "fullness": 40, "delay_probability": 0.2},
                {"from": "Golf", "to": "November", "line": "Diagonal-SE-6", "travel": 3, "wait": 3, "transfer": 2, "fullness": 42, "delay_probability": 0.2},
                {"from": "November", "to": "Uniform", "line": "Diagonal-SE-6", "travel": 2, "wait": 0, "transfer": 0, "fullness": 45, "delay_probability": 0.2},
                {"from": "Uniform", "to": "Birch", "line": "Diagonal-SE-6", "travel": 2, "wait": 0, "transfer": 0, "fullness": 48, "delay_probability": 0.2},
                {"from": "Birch", "to": "Ivy", "line": "Diagonal-SE-6", "travel": 3, "wait": 0, "transfer": 0, "fullness": 49, "delay_probability": 0.2},
                {"from": "Ivy", "to": "Harbor", "line": "Ring", "travel": 2, "wait": 1, "transfer": 2, "fullness": 47, "delay_probability": 0.2},
            ],
            route_valid=True,
            route_reaches_goal=True,
            route_correct=True,
            route_duration_min=28,
            runtime_sec=1.2,
            metrics_text="Messages: 2",
            extra={
                "messages": 2,
                "route_revisions": 0,
                "best_candidate_turn": 1,
                "reference_duration_min": 28,
                "constraint_duration_min": 28,
                "displayed_line_sequence": ["Ring", "Diagonal-SE-6", "Ring"],
                "displayed_line_changes": 2,
                "reference_line_sequence": ["Ring", "Diagonal-SE-6", "Ring"],
                "reference_line_changes": 2,
                "constraint_line_sequence": ["Ring", "Diagonal-SE-6", "Ring"],
                "constraint_line_changes": 2,
                "constraint_delay_probability": 0.2,
                "warning_count": 0,
                "speech_turns": [],
                "timing_turns": [],
                "nlu_turns": [],
                "runtime_events": [{"phase": "preflight", "event_type": "viability_check", "payload": {}}],
                "preflight_viability": {"constraint_route_available": True},
                "conversation_outcome": "satisfied",
            },
        )
        from coop_navigation_sds.TransportNetwork import get_test_case

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = write_single_run_research_outputs(
                result,
                get_test_case("morning_peak_cross_city").scenario,
                tmpdir,
            )

            metrics_file = paths["metrics_file"]
            phase_dir = paths["phase_log_dir"]

            self.assertTrue(paths["run_dir"].exists())
            self.assertTrue(paths["run_manifest"].exists())
            self.assertEqual(paths["metrics_file"].parent, paths["run_dir"])
            self.assertTrue(metrics_file.exists())
            self.assertEqual(phase_dir, paths["run_dir"])
            self.assertTrue((phase_dir / "metric_phase_summary.jsonl").exists())
            self.assertTrue((phase_dir / "metric_catalog.json").exists())
            self.assertTrue(paths["retrospective_json"].exists())
            self.assertTrue(paths["protocol"]["protocol"].exists())
            self.assertTrue(paths["protocol"]["transcript_txt"].exists())
            self.assertTrue(paths["protocol"]["audio_manifest"].exists())
            self.assertTrue(paths["network_json"].exists())
            self.assertTrue(paths["network_graph"].exists())
            self.assertFalse(any(child.is_dir() for child in paths["run_dir"].iterdir()))


if __name__ == "__main__":
    unittest.main()
