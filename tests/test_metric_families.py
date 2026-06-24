import unittest

from coop_navigation_sds.EvaluationMetrics.catalog import (
    CORE_METRIC_KEYS,
    DEFAULT_METRIC_CONFIG,
    METRIC_KEYS,
    METRIC_FAMILY_SPECS,
    metric_metadata,
)
from coop_navigation_sds.EvaluationMetrics.metrics import obligatory_metric_map

from coop_navigation_sds.DialogManagement.result import DialogResult
from coop_navigation_sds.EvaluationMetrics.metrics import (
    MetricComputer,
    apply_cross_run_metrics,
)
from coop_navigation_sds.TransportNetwork.routes import (
    optimal_time_route,
    route_line_change_count,
    route_line_sequence,
    route_station_sequence,
    route_text_from_steps,
)
from coop_navigation_sds.TransportNetwork import DEFAULT_TEST_CASE, get_test_case


class MetricFamilyTests(unittest.TestCase):
    def test_each_phase_has_at_least_seven_default_metrics(self):
        for family in METRIC_FAMILY_SPECS:
            enabled = [
                key for key, _label in family["metrics"]
                if DEFAULT_METRIC_CONFIG[key]
            ]
            self.assertGreaterEqual(len(enabled), 7, family["key"])

    def test_all_metrics_are_obligatory(self):
        core_key = next(iter(CORE_METRIC_KEYS))
        self.assertTrue(DEFAULT_METRIC_CONFIG[core_key])
        configured = obligatory_metric_map()
        self.assertTrue(configured[core_key])

    def test_metric_catalog_no_longer_exports_tiers(self):
        core_key = next(iter(CORE_METRIC_KEYS))
        self.assertNotIn("tier", metric_metadata(core_key, "whole_dialogue"))

    def test_metric_record_exposes_stage_families(self):
        class FakeNISQAEvaluator:
            def evaluate(self, items):
                self.items = tuple(items)
                return {
                    "status": "available",
                    "score": 4.25,
                    "dimensions": {
                        "overall_mos": 4.25,
                        "noisiness": 4.1,
                        "discontinuity": 4.0,
                        "coloration": 4.2,
                        "loudness": 4.3,
                    },
                    "files": [],
                    "errors": [],
                }

        class FakeDNSMOSEvaluator:
            def evaluate(self, items):
                self.items = tuple(items)
                return {
                    "status": "available",
                    "score": 3.75,
                    "dimensions": {
                        "p808_mos": 3.9,
                        "signal_mos": 3.8,
                        "background_mos": 3.7,
                        "overall_mos": 3.75,
                    },
                    "files": [],
                    "errors": [],
                }

        nisqa_evaluator = FakeNISQAEvaluator()
        dnsmos_evaluator = FakeDNSMOSEvaluator()
        test_case = get_test_case(DEFAULT_TEST_CASE)
        scenario = test_case.scenario
        reference_arrival, reference_steps = optimal_time_route(
            scenario["start_station"],
            scenario["destination_station"],
            scenario["start_time_min"],
            scenario["transfer_time_min"],
        )
        reference_route = route_station_sequence(reference_steps)
        route_text = route_text_from_steps(reference_steps)

        result = DialogResult(
            condition_id="case",
            test_case_key=test_case.key,
            persona_key=test_case.persona_key,
            scenario_key=test_case.scenario_key,
            speech_pattern_key="clean",
            model_name="fake-model",
            conversation=[
                ("Agent A", f"I'm at {scenario['start_station']} at 08:07, going to {scenario['destination_station']}. What route should I take?"),
                ("Agent B", route_text),
            ],
            route=reference_route,
            route_steps=reference_steps,
            route_valid=True,
            route_reaches_goal=True,
            route_correct=True,
            route_duration_min=reference_arrival - scenario["start_time_min"],
            runtime_sec=1.25,
            extra={
                "metric_config": {key: True for key in METRIC_KEYS},
                "messages": 2,
                "model_param_key": "greedy",
                "model_parameters": {"do_sample": False, "temperature": None, "top_p": None},
                "candidate_routes": 1,
                "route_revisions": 0,
                "best_candidate_turn": 1,
                "warning_count": 0,
                "speech_turns": [
                    {
                        "speaker": "Agent B",
                        "generated_text": f"Please I would say {route_text}",
                        "outgoing_text": route_text,
                        "incoming_transcript": route_text,
                        "transcript_corrections": [],
                        "outgoing_enabled": True,
                        "incoming_enabled": True,
                        "mode": "speech",
                        "pipeline_ok": True,
                        "audio": {"path": "turn.wav"},
                        "latency_sec": 0.05,
                    }
                ],
                "timing_turns": [
                    {
                        "speaker": "Agent B",
                        "generation_sec": 0.02,
                        "speech_sec": 0.05,
                        "turn_latency_sec": 0.07,
                    }
                ],
                "nlu_turns": [
                    {
                        "speaker": "Agent B",
                        "text": route_text,
                        "has_station_mentions": True,
                        "parsed_route": reference_route,
                        "route_valid": True,
                        "route_reaches_goal": True,
                    }
                ],
                "agent_timing_summary": {
                    "Agent A": {
                        "turn_count": 1,
                        "word_count": 12,
                        "mean_words_per_turn": 12.0,
                        "total_generation_sec": 0.0,
                        "total_speech_sec": 0.03,
                        "mean_turn_elapsed_sec": 0.03,
                        "max_turn_elapsed_sec": 0.03,
                    },
                    "Agent B": {
                        "turn_count": 1,
                        "word_count": 14,
                        "mean_words_per_turn": 14.0,
                        "total_generation_sec": 0.02,
                        "total_speech_sec": 0.05,
                        "mean_turn_elapsed_sec": 0.07,
                        "max_turn_elapsed_sec": 0.07,
                    },
                },
                "condition_runtime_sec": 0.25,
                "runtime_events": [
                    {
                        "phase": "dialogue_state_tracking",
                        "event_type": "agent_memory_snapshot",
                        "payload": {
                            "turn": 1,
                            "snapshots": {
                                "Agent A": {"current_route": []},
                                "Agent B": {"current_route": []},
                            },
                            "additions": {"Agent A": {"latest_spoken": "Need a route."}},
                        },
                    },
                    {
                        "phase": "dialogue_state_tracking",
                        "event_type": "agent_memory_snapshot",
                        "payload": {
                            "turn": 2,
                            "snapshots": {
                                "Agent A": {"current_route": reference_route},
                                "Agent B": {"current_route": reference_route},
                            },
                            "additions": {
                                "Agent A": {"current_route": reference_route},
                                "Agent B": {"current_route": reference_route},
                            },
                        },
                    },
                ],
            },
        )

        record = MetricComputer(
            nisqa_evaluator=nisqa_evaluator,
            dnsmos_evaluator=dnsmos_evaluator,
        ).compute(result, scenario)
        row = record.as_dict()

        self.assertIn("audio_turn_latency", row)
        self.assertIn("audio_input", record.metric_families)
        self.assertIn("user_simulation", record.metric_families)
        self.assertIn("backend_task_execution", record.metric_families)
        self.assertIn("dialogue_state_tracking", record.metric_families)
        self.assertEqual(row["audio_turn_latency"], 0.07)
        self.assertNotIn("audio_end_of_utterance_error", row)
        self.assertNotIn("audio_overlap_rate", row)
        self.assertIn("asr_wer", row)
        self.assertEqual(row["asr_success_rate"], 1.0)
        self.assertEqual(row["asr_failure_count"], 0)
        self.assertEqual(row["asr_word_error_rate"], 0.0)
        self.assertEqual(row["asr_wer"], 0.0)
        self.assertEqual(row["asr_entity_error_rate"], 0.0)
        self.assertEqual(row["asr_transcript_correction_count"], 0)
        self.assertEqual(row["asr_uncorrected_misinterpretation_count"], 0)
        self.assertEqual(row["nlu_pipeline_input_match_rate"], 1.0)
        self.assertEqual(row["nlu_constraint_extraction_f1"], 1.0)
        self.assertEqual(row["nlu_semantic_frame_accuracy"], 1.0)
        self.assertEqual(row["dialogue_state_shared_state_agreement"], 1.0)
        self.assertEqual(row["dialogue_state_memory_trace_coverage"], 1.0)
        self.assertEqual(row["dialogue_state_memory_update_rate"], 1.0)
        self.assertEqual(row["dialogue_state_route_memory_retention_rate"], 1.0)
        self.assertEqual(row["dialogue_management_premature_answer_rate"], 0.0)
        self.assertGreaterEqual(row["dialogue_management_repair_success_rate"], 0.0)
        self.assertLessEqual(row["dialogue_management_repair_success_rate"], 1.0)
        self.assertEqual(row["agent_b_grounded_proposal_score"], 1.0)
        self.assertEqual(row["agent_b_actionability_score"], 1.0)
        self.assertEqual(row["agent_a_false_acceptance_rate"], 0.0)
        self.assertEqual(row["nlg_faithfulness"], 1.0)
        self.assertEqual(row["tts_success_rate"], 1.0)
        self.assertEqual(row["tts_failure_count"], 0)
        self.assertEqual(row["tts_round_trip_semantic_intelligibility"], 1.0)
        self.assertEqual(row["tts_dnsmos"], 3.75)
        self.assertEqual(row["tts_nisqa"], 4.25)
        self.assertEqual(nisqa_evaluator.items[0]["path"], "turn.wav")
        self.assertEqual(nisqa_evaluator.items[0]["speaker"], "Agent B")
        self.assertEqual(result.extra["nisqa_evaluation"]["status"], "available")
        self.assertEqual(result.extra["dnsmos_evaluation"]["status"], "available")
        self.assertEqual(dnsmos_evaluator.items[0]["speaker"], "Agent B")
        self.assertGreater(row["tts_text_change_rate"], 0.0)
        self.assertEqual(row["speech_incoming_enabled_rate"], 1.0)
        self.assertEqual(row["speech_outgoing_enabled_rate"], 1.0)
        self.assertEqual(row["mean_turn_elapsed_sec"], 0.07)
        self.assertEqual(row["max_turn_elapsed_sec"], 0.07)
        self.assertEqual(row["condition_runtime_sec"], 0.25)
        self.assertEqual(row["speech_duration_total_sec"], 0.05)
        self.assertEqual(row["whole_dialogue_dialogue_cost"], 2)
        self.assertEqual(row["pipeline_mode"], "speech")
        self.assertEqual(row["pipeline_success_rate"], 1.0)
        self.assertEqual(row["pipeline_failure_count"], 0)
        self.assertEqual(row["whole_dialogue_interaction_quality_trajectory"], row["automatic_eval_score"])
        self.assertNotIn("metric_validity_rank_stability", row)
        self.assertIn("route_line_sequence", row)
        self.assertIn("reference_line_sequence", row)
        self.assertGreaterEqual(row["route_line_change_count"], 0)

    def test_route_line_helpers_collapse_consecutive_segments(self):
        steps = [
            {"line": "Red"},
            {"line": "Red"},
            {"line": "Blue"},
            {"line": "Blue"},
            {"line": "Green"},
        ]

        self.assertEqual(route_line_sequence(steps), ["Red", "Blue", "Green"])
        self.assertEqual(route_line_change_count(steps), 2)

    def test_metric_manifest_exposes_complete_family_stack(self):
        titles = [family["title"] for family in METRIC_FAMILY_SPECS]
        self.assertEqual(
            titles,
            [
                "0. User Simulation and Input",
                "1. Audio Input and Turn-Taking",
                "2. Automatic Speech Recognition",
                "3. Spoken-Language Understanding",
                "4. Dialogue State Tracking",
                "5. Dialogue Management and Policy",
                "6. Backend Task Execution and Grounding",
                "7. Natural-Language Generation",
                "8. Text-to-Speech and Audio Output",
                "9. End-to-End Task Outcome",
                "10. Whole-Dialogue Interaction",
                "11. Cross-Run Validity and Robustness",
            ],
        )

        self.assertEqual(
            [family["key"] for family in METRIC_FAMILY_SPECS],
            [
                "user_simulation",
                "audio_input",
                "asr",
                "nlu",
                "dialogue_state_tracking",
                "dialogue_management",
                "backend_task_execution",
                "nlg",
                "tts",
                "task_outcome",
                "whole_dialogue",
                "metric_validity",
            ],
        )

    def test_expanded_catalog_contains_phase_diagnostic_metrics(self):
        keys = {
            key
            for family in METRIC_FAMILY_SPECS
            for key, _label in family["metrics"]
        }

        self.assertIn("asr_numeric_preservation_rate", keys)
        self.assertIn("nlu_joint_frame_accuracy", keys)
        self.assertIn("dialogue_state_candidate_memory_precision", keys)
        self.assertIn("dialogue_state_memory_trace_coverage", keys)
        self.assertIn("dialogue_state_memory_update_rate", keys)
        self.assertIn("dialogue_state_route_memory_retention_rate", keys)
        self.assertIn("dialogue_management_clarification_precision", keys)
        self.assertIn("agent_b_duration_regret", keys)
        self.assertIn("nlg_slot_error_rate", keys)
        self.assertIn("tts_real_time_factor", keys)
        self.assertIn("task_outcome_first_compliant_route_turn", keys)
        self.assertIn("metric_validity_missingness_rate", keys)

    def test_cross_run_metrics_are_populated_after_batch_aggregation(self):
        test_case = get_test_case(DEFAULT_TEST_CASE)
        scenario = test_case.scenario
        records = []
        for iteration, correct in enumerate((True, False)):
            result = DialogResult(
                condition_id=f"condition__{iteration}",
                test_case_key=test_case.key,
                persona_key=test_case.persona_key,
                scenario_key=test_case.scenario_key,
                speech_pattern_key="clean" if iteration == 0 else "mostly_clean",
                model_name="fake-model",
                conversation=[("Agent A", "Need a route.")],
                route=[],
                route_steps=[],
                route_valid=correct,
                route_reaches_goal=correct,
                route_correct=correct,
                route_duration_min=None,
                runtime_sec=0.1,
                extra={"messages": 1, "iteration": iteration},
            )
            records.append(MetricComputer().compute(result, scenario))

        apply_cross_run_metrics(records)

        for record in records:
            validity = record.metric_families["metric_validity"]
            self.assertIsNotNone(validity["success_confidence_interval_low"])
            self.assertIsNotNone(validity["success_confidence_interval_high"])
            self.assertIsNotNone(validity["seed_variance"])
            self.assertIsNotNone(validity["missingness_rate"])
            self.assertNotIn("available", validity)

    def test_legacy_metric_configuration_cannot_filter_metrics(self):
        test_case = get_test_case(DEFAULT_TEST_CASE)
        scenario = test_case.scenario
        result = DialogResult(
            condition_id="metric-config",
            test_case_key=test_case.key,
            persona_key=test_case.persona_key,
            scenario_key=test_case.scenario_key,
            speech_pattern_key="clean",
            model_name="fake-model",
            conversation=[("Agent A", "Need Alpha to Harbor.")],
            route=[],
            route_steps=[],
            route_valid=False,
            route_reaches_goal=False,
            route_correct=False,
            route_duration_min=None,
            runtime_sec=0.1,
            extra={
                "metric_config": {
                    "asr_wer": False,
                    "audio_missing_rate": False,
                    "audio_turn_latency": True,
                },
            },
        )

        row = MetricComputer().compute(result, scenario).as_dict()

        self.assertIn("audio_turn_latency", row)
        self.assertIn("asr_wer", row)
        self.assertIn("audio_missing_rate", row)
        self.assertIn("asr_word_error_rate", row)

    def test_metric_record_exposes_pipeline_failure_case(self):
        test_case = get_test_case(DEFAULT_TEST_CASE)
        scenario = test_case.scenario
        result = DialogResult(
            condition_id="failure-case",
            test_case_key=test_case.key,
            persona_key=test_case.persona_key,
            scenario_key=test_case.scenario_key,
            speech_pattern_key="clean",
            model_name="fake-model",
            conversation=[("Agent A", "Need Bravo to Harbor.")],
            route=[],
            route_steps=[],
            route_valid=False,
            route_reaches_goal=False,
            route_correct=False,
            route_duration_min=None,
            runtime_sec=0.2,
            extra={
                "messages": 1,
                "speech_turns": [
                    {
                        "speaker": "Agent A",
                        "generated_text": "Need Bravo to Harbor.",
                        "outgoing_text": "",
                        "incoming_transcript": "",
                        "outgoing_enabled": True,
                        "incoming_enabled": True,
                        "mode": "speech",
                        "pipeline_ok": False,
                        "failure_reason": "text-to-speech failed",
                    }
                ],
                "timing_turns": [],
                "nlu_turns": [],
                "pipeline_failure": {"message": "text-to-speech failed"},
            },
        )

        row = MetricComputer().compute(result, scenario).as_dict()

        self.assertEqual(row["pipeline_mode"], "speech")
        self.assertEqual(row["pipeline_success_rate"], 0.0)
        self.assertEqual(row["pipeline_failure_count"], 2)
        self.assertEqual(row["tts_success_rate"], 0.0)
        self.assertEqual(row["tts_failure_count"], 1)
        self.assertEqual(row["asr_success_rate"], 0.0)
        self.assertEqual(row["asr_failure_count"], 1)
        self.assertEqual(row["tts_round_trip_semantic_intelligibility"], 0.0)


if __name__ == "__main__":
    unittest.main()
