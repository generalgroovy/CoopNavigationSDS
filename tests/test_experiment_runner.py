import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import xml.etree.ElementTree as ElementTree
from unittest.mock import MagicMock, patch
import wave
from zipfile import ZipFile

from coop_navigation_sds.DialogManagement.result import DialogResult
from coop_navigation_sds.Configuration.experiment import ExperimentSpecification
from coop_navigation_sds.experiments import ExperimentCondition, ExperimentRunner, build_condition_grid, condition_configuration_provenance, write_metrics_csv, write_metrics_file
from coop_navigation_sds.ResultsAndArtifacts.artifacts import compact_existing_result_tree, consolidate_completed_runtime_logs, create_execution_run_dir, write_conversation_protocol, write_conversation_protocols, write_experiment_manifest, write_metric_phase_logs, write_network_research_artifacts, write_single_run_research_outputs
from coop_navigation_sds.NaturalLanguageGeneration.models import ModelParameterSet
from coop_navigation_sds.TransportNetwork.constraints import OBJECTIVE_SHORTEST_WITH_CONSTRAINTS


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
    def test_condition_fingerprint_distinguishes_treatments_from_shared_base(self):
        specification = ExperimentSpecification.resolve({"agent_a_type": "tinyllama"})
        common = {
            "test_case_key": "morning_peak_cross_city",
            "persona_key": "focused_commuter",
            "scenario_key": "morning_peak_cross_city",
            "speech_pattern_key": "clean",
            "model_param_key": "greedy",
        }
        ceiling = ExperimentCondition(condition_id="ceiling", **common)
        floor = ExperimentCondition(
            condition_id="floor",
            parameter_values={"speech_performance_band": "floor"},
            **common,
        )

        ceiling_provenance = condition_configuration_provenance(specification, ceiling)
        floor_provenance = condition_configuration_provenance(specification, floor)

        self.assertEqual(
            ceiling_provenance["base_fingerprint_sha256"],
            floor_provenance["base_fingerprint_sha256"],
        )
        self.assertNotEqual(
            ceiling_provenance["fingerprint_sha256"],
            floor_provenance["fingerprint_sha256"],
        )

    def test_experiment_condition_enforces_constraint_aware_shortest_route(self):
        condition = ExperimentCondition(
            condition_id="fixed-objective",
            test_case_key="morning_peak_cross_city",
            persona_key="focused_commuter",
            scenario_key="morning_peak_cross_city",
            speech_pattern_key="clean",
            model_param_key="greedy",
            objective_mode="only_valid_route",
        )

        self.assertEqual(condition.objective_mode, OBJECTIVE_SHORTEST_WITH_CONSTRAINTS)

    def test_shared_tinyllama_condition_reuses_loaded_adapter(self):
        adapter = FakeModelAdapter(name="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        factory = MagicMock()
        runner = ExperimentRunner(
            adapter,
            num_turns=3,
            model_adapter_factory=factory,
        )
        condition = ExperimentCondition(
            condition_id="shared-model",
            test_case_key="morning_peak_cross_city",
            persona_key="focused_commuter",
            scenario_key="morning_peak_cross_city",
            speech_pattern_key="clean",
            model_param_key="greedy",
            agent_b_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        )

        selected = runner._model_adapter_for(condition)

        factory.assert_not_called()
        self.assertEqual(selected.name, adapter.name)

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
        audio_dir = Path(dialog_manager_cls.call_args.kwargs["speech_transport"].config.audio_dir)
        self.assertEqual(audio_dir.parent.name, ".turn_audio")
        self.assertEqual(audio_dir.name, hashlib.sha256(condition.condition_id.encode("utf-8")).hexdigest()[:12])
        self.assertLess(len(str(audio_dir)), len(str(audio_dir.parent / condition.condition_id)))
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
    def test_run_condition_can_keep_agent_a_model_separate_from_agent_b_grid(
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

        agent_b_base = FakeModelAdapter("agent-b-base")
        agent_a_fixed = FakeModelAdapter("agent-a-tinyllama")
        agent_b_condition = FakeModelAdapter("agent-b-comparison")
        factory = MagicMock(return_value=agent_b_condition)
        runner = ExperimentRunner(
            agent_b_base,
            num_turns=3,
            agent_b_plugin_key="llm",
            tts_engine="file",
            asr_engine="file",
            speech_playback_enabled=False,
            speech_realtime_enabled=False,
            agent_a_type="tinyllama",
            model_adapter_factory=factory,
            agent_a_model_adapter=agent_a_fixed,
        )
        runner.metric_computer.compute = MagicMock(return_value="metric")
        result = SimpleNamespace(extra={})
        dialog_manager_cls.return_value.run.return_value = result

        condition = ExperimentCondition(
            condition_id="separate_models",
            test_case_key="case",
            persona_key="focused_commuter",
            scenario_key="scenario",
            speech_pattern_key="clean",
            model_param_key="greedy",
            agent_b_model="agent-b-comparison",
        )

        runner.run_condition(condition)

        factory.assert_called_once_with("agent-b-comparison")
        self.assertEqual(create_agent_b_plugin.call_args.args[1].name, "agent-b-comparison")
        agent_a_responder = dialog_manager_cls.call_args.kwargs["agent_a_responder"]
        self.assertEqual(agent_a_responder.model_adapter.name, "agent-a-tinyllama")
        self.assertEqual(result.extra["agent_b_model"], "agent-b-comparison")
        self.assertEqual(result.extra["model_backend"]["model"], "agent-b-comparison")
        self.assertEqual(result.extra["agent_a_model_backend"]["model"], "agent-a-tinyllama")

    @patch("coop_navigation_sds.experiments.create_agent_b_plugin")
    @patch("coop_navigation_sds.experiments.DialogManager")
    @patch("coop_navigation_sds.experiments.get_test_case")
    def test_run_condition_captures_failure_for_batch_and_allows_next_condition(
        self,
        get_test_case,
        dialog_manager_cls,
        _create_agent_b_plugin,
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
        base_case.with_persona = lambda _persona_key: base_case
        get_test_case.return_value = base_case
        failed = RuntimeError("provider failed")
        successful = SimpleNamespace(extra={}, runtime_sec=0.0)
        dialog_manager_cls.return_value.run.side_effect = [failed, successful]
        runner = ExperimentRunner(
            FakeModelAdapter(),
            num_turns=1,
            agent_b_plugin_key="simple",
            tts_engine="file",
            asr_engine="file",
            speech_playback_enabled=False,
            speech_realtime_enabled=False,
        )
        conditions = [
            ExperimentCondition(
                condition_id=f"condition-{index}",
                test_case_key="case",
                persona_key="focused_commuter",
                scenario_key="scenario",
                speech_pattern_key="clean",
                model_param_key="greedy",
            )
            for index in (1, 2)
        ]

        first, _ = runner.run_condition(conditions[0], compute_metrics=False, capture_failure=True)
        second, _ = runner.run_condition(conditions[1], compute_metrics=False, capture_failure=True)

        self.assertEqual(first.extra["execution_status"], "failed")
        self.assertEqual(first.extra["pipeline_failure"]["message"], "provider failed")
        self.assertFalse(first.route_correct)
        self.assertEqual(second.extra["execution_status"], "completed")
        self.assertEqual(dialog_manager_cls.return_value.run.call_count, 2)

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

    @patch("coop_navigation_sds.experiments.get_test_case")
    def test_build_condition_grid_has_one_fixed_objective(self, get_test_case):
        get_test_case.return_value = SimpleNamespace(scenario_key="scenario")

        conditions = list(build_condition_grid(
            test_case_keys=["case"],
            persona_keys=["persona"],
            speech_pattern_keys=["clean"],
            model_param_keys=["greedy"],
            objective_modes=["only_valid_route", "shortest_valid_route"],
        ))

        self.assertEqual(len(conditions), 1)
        self.assertEqual(conditions[0].objective_mode, OBJECTIVE_SHORTEST_WITH_CONSTRAINTS)

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

            long_rows = list(csv.DictReader((log_dir / "metrics_long.csv").open(encoding="utf-8")))
            runtime_record = next(
                row for row in long_rows
                if row["phase"] == "runtime" and row["metric_name"] == "response_latency_sec"
            )

        self.assertIn("xl/workbook.xml", names)
        self.assertIn('name="summary"', workbook_xml)
        self.assertIn('name="metric_long"', workbook_xml)
        self.assertIn('name="runtime"', workbook_xml)
        self.assertEqual(runtime_record["phase"], "runtime")
        self.assertEqual(float(runtime_record["value_numeric"]), 0.12)
        self.assertEqual({row["phase"] for row in long_rows}, {"runtime", "asr"})
        self.assertFalse((log_dir / "metrics_long.jsonl").exists())

    def test_write_network_research_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = write_network_research_artifacts(480, Path(tmpdir) / "research", Path(tmpdir) / "graphs")

            network_data = json.loads(paths["network_json"].read_text(encoding="utf-8"))
            graph_text = paths["network_graph"].read_text(encoding="utf-8")

        self.assertGreater(network_data["line_count"], 0)
        self.assertGreater(network_data["station_count"], 0)
        self.assertIn("<svg", graph_text)

    def test_network_graph_contains_every_connection_and_external_index(self):
        from coop_navigation_sds.TransportNetwork.network import LINES, line_stop_pairs

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = write_network_research_artifacts(
                480,
                Path(tmpdir) / "research",
                Path(tmpdir) / "graphs",
            )
            root = ElementTree.parse(paths["network_graph"]).getroot()

        namespace = {"svg": "http://www.w3.org/2000/svg"}
        connections = root.findall(".//svg:line[@class='network-connection']", namespace)
        expected = sum(
            len(line_stop_pairs(line_name, data))
            for line_name, data in LINES.items()
        )
        self.assertEqual(len(connections), expected)
        self.assertTrue(
            all(
                edge.get("data-line")
                and edge.get("data-mode")
                and edge.get("data-from")
                and edge.get("data-to")
                for edge in connections
            )
        )
        walking = [edge for edge in connections if edge.get("data-mode") == "walking"]
        self.assertTrue(walking)
        self.assertTrue(all(edge.get("stroke-dasharray") for edge in walking))
        self.assertIn("Line index", "".join(root.itertext()))

        geometries_by_pair = {}
        for edge in connections:
            pair = tuple(sorted((edge.get("data-from"), edge.get("data-to"))))
            geometry = tuple(
                edge.get(attribute)
                for attribute in ("x1", "y1", "x2", "y2")
            )
            geometries_by_pair.setdefault(pair, []).append(geometry)
        self.assertTrue(
            all(
                len(geometries) == len(set(geometries))
                for geometries in geometries_by_pair.values()
            )
        )

        index_panel = root.find(".//svg:rect[@class='line-index-panel']", namespace)
        self.assertIsNotNone(index_panel)
        index_x = float(index_panel.get("x"))
        self.assertLess(
            max(max(float(edge.get("x1")), float(edge.get("x2"))) for edge in connections),
            index_x,
        )

    def test_create_execution_run_dir_uses_systematic_unique_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            first = create_execution_run_dir(tmpdir, label="Single Case", timestamp="20260531_120000")
            second = create_execution_run_dir(tmpdir, label="Single Case", timestamp="20260531_120000")

        self.assertEqual(first.name, "20260531_120000_single_case")
        self.assertEqual(second.name, "20260531_120000_single_case_02")

    def test_create_execution_run_dir_retries_existing_parallel_suffixes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            (base / "20260531_120000_parallel_case").mkdir()
            (base / "20260531_120000_parallel_case_02").mkdir()

            created = create_execution_run_dir(
                base,
                label="Parallel Case",
                timestamp="20260531_120000",
            )

        self.assertEqual(created.name, "20260531_120000_parallel_case_03")

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
            batch_dir = Path(tmpdir) / "batch"
            batch_protocol_count = len(list(batch_dir.glob("*_protocol.json")))
            batch_transcript_count = len(list(batch_dir.glob("*_conversation_transcript.txt")))
            batch_protocol_records = [
                json.loads(line)
                for line in (batch_dir / "conversation_protocols.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            batch_transcript = (batch_dir / "conversation_transcripts.txt").read_text(encoding="utf-8")

            combined_protocol = json.loads(paths["protocol"].read_text(encoding="utf-8"))
            transcript_text = paths["transcript_txt"].read_text(encoding="utf-8")
            conversation_wav_exists = paths["conversation_wav"].exists()

        self.assertTrue(batch_paths)
        self.assertEqual(batch_protocol_count, 0)
        self.assertEqual(batch_transcript_count, 0)
        self.assertEqual(len(batch_protocol_records), 1)
        self.assertIn("=== Conversation 1: Case One ===", batch_transcript)
        self.assertEqual(protocol_children, [])
        self.assertEqual(combined_protocol["summary"]["condition_id"], "Case One")
        self.assertIn("Agent A", transcript_text)
        self.assertTrue(combined_protocol["audio_manifest"]["created"])
        self.assertTrue(conversation_wav_exists)
        self.assertTrue(combined_protocol["verification"]["verified"])
        self.assertEqual(len(combined_protocol["conversation"]), 2)
        self.assertEqual(len(combined_protocol["agent_turn_segments"]), 2)
        self.assertEqual(
            {row["speaker"] for row in combined_protocol["agent_turn_segments"]},
            {"Agent A", "Agent B"},
        )
        self.assertEqual(combined_protocol["agent_timing_summary"]["Agent B"]["mean_turn_elapsed_sec"], 0.4)
        self.assertEqual(len(combined_protocol["runtime_events"]), 1)
        self.assertEqual(len(combined_protocol["phase_timing"]), 1)
        self.assertIn("processing seconds", transcript_text)
        self.assertIn("Messages", combined_protocol["retrospective_summary"])

    def test_interrupted_result_recovery_combines_readable_files_without_deleting_sources(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "results" / "20260703_partial"
            run_dir.mkdir(parents=True)
            (run_dir / "experiment_job.json").write_text("{}", encoding="utf-8")
            wav_path = run_dir / "0001-agent-a.wav"
            txt_path = run_dir / "0001-agent-a.txt"
            write_test_wav(wav_path)
            txt_path.write_text("Alpha to Echo", encoding="utf-8")
            events = [
                {
                    "schema_version": 3,
                    "kind": "program.segment",
                    "name": "batch.condition.start",
                    "payload": {"segment": "batch.condition", "phase": "start", "condition_id": "case-one"},
                },
                {
                    "schema_version": 3,
                    "kind": "system",
                    "name": "telemetry.speech",
                    "payload": {
                        "turn": 1,
                        "speaker": "Agent A",
                        "generated_text": "Alpha to Echo",
                        "outgoing_text": "Alpha to Echo",
                        "raw_asr_transcript": "alpha to echo",
                        "incoming_transcript": "Alpha to Echo",
                        "agent_input_transcript": "Alpha to Echo",
                        "misinterpreted_tokens": [],
                        "transcript_corrections": [],
                        "audio": {"path": str(wav_path), "transcript_path": str(txt_path)},
                    },
                },
            ]
            log_path = run_dir / "batch-case-one.jsonl"
            log_path.write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )

            reports = compact_existing_result_tree(run_dir.parent)
            protocol_lines = [
                json.loads(line)
                for line in (run_dir / "partial_conversation_protocols.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            transcript = (run_dir / "partial_conversation_transcripts.txt").read_text(encoding="utf-8")

            self.assertEqual(reports[0]["status"], "recovered_incomplete_run")
            self.assertEqual(len(protocol_lines), 1)
            self.assertEqual(protocol_lines[0]["preservation_policy"], "original turn files retained because the batch did not finalize")
            self.assertTrue(protocol_lines[0]["source_integrity"])
            self.assertIn("TTS speech: Alpha to Echo", transcript)
            self.assertTrue(wav_path.is_file())
            self.assertTrue(txt_path.is_file())
            self.assertTrue((run_dir / "case-one_partial_conversation.wav").is_file())

    def test_completed_runtime_logs_are_verified_and_consolidated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            event_path = root / "batch-case.jsonl"
            text_path = root / "batch-case.log"
            summary_path = root / "batch-case-summary.json"
            event_path.write_text(json.dumps({"name": "condition.end"}) + "\n", encoding="utf-8")
            text_path.write_text("condition complete\n", encoding="utf-8")
            summary_path.write_text(json.dumps({"events": 1}), encoding="utf-8")

            outputs = consolidate_completed_runtime_logs(root)
            event = json.loads(outputs["events"].read_text(encoding="utf-8"))
            summary = json.loads(outputs["summaries"].read_text(encoding="utf-8"))

            self.assertEqual(event["source_session_file"], event_path.name)
            self.assertEqual(event["event"]["name"], "condition.end")
            self.assertEqual(len(event["source_sha256"]), 64)
            self.assertEqual(summary["summary"]["events"], 1)
            self.assertIn("condition complete", outputs["text"].read_text(encoding="utf-8"))
            self.assertFalse(event_path.exists())
            self.assertFalse(text_path.exists())
            self.assertFalse(summary_path.exists())

    def test_finalized_result_compaction_refreshes_index_and_artifact_inventory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "results" / "completed"
            run_dir.mkdir(parents=True)
            protocol_path = run_dir / "case-one_protocol.json"
            transcript_path = run_dir / "case-one_conversation_transcript.txt"
            wav_path = run_dir / "case-one_conversation.wav"
            protocol_path.write_text(
                json.dumps({"summary": {"condition_id": "case-one"}}),
                encoding="utf-8",
            )
            transcript_path.write_text("Agent A: Alpha to Echo.\n", encoding="utf-8")
            write_test_wav(wav_path)
            (run_dir / "index.jsonl").write_text(
                json.dumps({"condition_id": "case-one", "protocol_file": protocol_path.name}) + "\n",
                encoding="utf-8",
            )
            (run_dir / "run_summary.json").write_text(
                json.dumps({"artifacts": [{"path": protocol_path.name}]}),
                encoding="utf-8",
            )

            reports = compact_existing_result_tree(run_dir.parent)
            index = json.loads((run_dir / "index.jsonl").read_text(encoding="utf-8"))
            summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
            inventory = json.loads(
                (run_dir / summary["artifact_inventory"]).read_text(encoding="utf-8")
            )
            artifact_names = {row["path"] for row in inventory["artifacts"]}

            self.assertEqual(reports[0]["status"], "compacted_finalized_run")
            self.assertFalse(protocol_path.exists())
            self.assertFalse(transcript_path.exists())
            self.assertEqual(index["protocol_file"], "conversation_protocols.jsonl")
            self.assertEqual(index["protocol_record"], 1)
            self.assertEqual(index["transcript_file"], "conversation_transcripts.txt")
            self.assertIn("conversation_protocols.jsonl", artifact_names)
            self.assertIn("conversation_transcripts.txt", artifact_names)
            self.assertNotIn(protocol_path.name, artifact_names)
            self.assertTrue(all(row["sha256"] for row in inventory["artifacts"]))

    def test_interrupted_recovery_rejects_audio_reused_across_conditions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "results" / "partial"
            run_dir.mkdir(parents=True)
            (run_dir / "experiment_job.json").write_text("{}", encoding="utf-8")
            shared_wav = run_dir / "shared.wav"
            write_test_wav(shared_wav)
            for condition_id in ("case-one", "case-two"):
                events = [
                    {
                        "name": "batch.condition.start",
                        "payload": {"condition_id": condition_id},
                    },
                    {
                        "name": "telemetry.speech",
                        "payload": {
                            "turn": 1,
                            "speaker": "Agent A",
                            "generated_text": "Alpha to Echo",
                            "outgoing_text": "Alpha to Echo",
                            "incoming_transcript": "Alpha to Echo",
                            "audio": {"path": str(shared_wav)},
                        },
                    },
                ]
                (run_dir / f"batch-{condition_id}.jsonl").write_text(
                    "".join(json.dumps(event) + "\n" for event in events),
                    encoding="utf-8",
                )

            compact_existing_result_tree(run_dir.parent)
            records = [
                json.loads(line)
                for line in (run_dir / "partial_conversation_protocols.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            self.assertEqual(len(records), 2)
            self.assertTrue(all(record["audio_recovery_status"] == "unavailable_or_ambiguous" for record in records))
            self.assertTrue(all("shared across conditions" in record["audio_manifest"]["reason"] for record in records))
            self.assertEqual(list(run_dir.glob("*_partial_conversation.wav")), [])
            self.assertTrue(shared_wav.is_file())

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
            self.assertTrue((phase_dir / "metrics_long.csv").exists())
            self.assertTrue((phase_dir / "metrics_wide.csv").exists())
            self.assertFalse((phase_dir / "metrics_long.jsonl").exists())
            self.assertFalse((phase_dir / "metric_phase_summary.jsonl").exists())
            self.assertFalse((phase_dir / "metric_catalog.json").exists())
            self.assertFalse((phase_dir / "retrospective_metrics.json").exists())
            long_rows = list(csv.DictReader((phase_dir / "metrics_long.csv").open(encoding="utf-8")))
            self.assertGreater(len(long_rows), 0)
            self.assertTrue({
                "phase",
                "metric_key",
                "value_numeric",
                "available",
                "formula",
                "operands_json",
                "selection_rationale",
            }.issubset(long_rows[0]))
            self.assertTrue(paths["protocol"]["protocol"].exists())
            self.assertTrue(paths["protocol"]["transcript_txt"].exists())
            protocol = json.loads(paths["protocol"]["protocol"].read_text(encoding="utf-8"))
            self.assertIn("audio_manifest", protocol)
            self.assertTrue(paths["network_json"].exists())
            self.assertTrue(paths["network_graph"].exists())
            summary = json.loads(paths["run_summary"].read_text(encoding="utf-8"))
            inventory_path = paths["run_dir"] / summary["artifact_inventory"]
            inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["artifact_count"], inventory["artifact_count"])
            self.assertTrue(all(row["sha256"] for row in inventory["artifacts"]))
            self.assertFalse(any(child.is_dir() for child in paths["run_dir"].iterdir()))


if __name__ == "__main__":
    unittest.main()
