import unittest

from minillama.controller.dialog_result import DialogResult
from minillama.evaluation.metrics import METRIC_FAMILY_SPECS, MetricComputer
from minillama.model.route_planner import (
    optimal_time_route,
    route_line_change_count,
    route_line_sequence,
    route_station_sequence,
    route_text_from_steps,
)
from minillama.test_cases import DEFAULT_TEST_CASE, get_test_case


class MetricFamilyTests(unittest.TestCase):
    def test_metric_record_exposes_stage_families(self):
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
                "condition_runtime_sec": 0.25,
            },
        )

        row = MetricComputer().compute(result, scenario).as_dict()

        self.assertIn("audio_available", row)
        self.assertFalse(row["audio_available"])
        self.assertIn("audio_snr_db", row)
        self.assertIn("vad_false_alarm_rate", row)
        self.assertIn("diarization_der", row)
        self.assertIn("asr_word_error_rate", row)
        self.assertEqual(row["asr_success_rate"], 1.0)
        self.assertEqual(row["asr_failure_count"], 0)
        self.assertIn("asr_confidence_calibration", row)
        self.assertEqual(row["asr_word_error_rate"], 0.0)
        self.assertEqual(row["asr_entity_wer"], 0.0)
        self.assertIn("slu_intent_error_rate", row)
        self.assertEqual(row["slu_pipeline_input_match_rate"], 1.0)
        self.assertEqual(row["slu_slot_f1"], 1.0)
        self.assertIn("dst_requested_slot_f1", row)
        self.assertEqual(row["dst_joint_goal_accuracy"], 1.0)
        self.assertIn("policy_dialog_act_f1", row)
        self.assertEqual(row["policy_tool_call_exact_match"], 1.0)
        self.assertIn("tool_result_relevance", row)
        self.assertIn("nlg_rouge", row)
        self.assertEqual(row["nlg_constraint_satisfaction_rate"], 1.0)
        self.assertIn("tts_predicted_mos", row)
        self.assertEqual(row["tts_success_rate"], 1.0)
        self.assertEqual(row["tts_failure_count"], 0)
        self.assertEqual(row["tts_intelligibility_wer"], 0.0)
        self.assertGreater(row["tts_text_change_rate"], 0.0)
        self.assertEqual(row["speech_incoming_enabled_rate"], 1.0)
        self.assertEqual(row["speech_outgoing_enabled_rate"], 1.0)
        self.assertIn("runtime_time_to_first_token_sec", row)
        self.assertEqual(row["condition_runtime_sec"], 0.25)
        self.assertEqual(row["runtime_condition_runtime_sec"], 0.25)
        self.assertEqual(row["speech_duration_total_sec"], 0.05)
        self.assertEqual(row["pipeline_mode"], "speech")
        self.assertEqual(row["pipeline_success_rate"], 1.0)
        self.assertEqual(row["pipeline_failure_count"], 0)
        self.assertEqual(row["pipeline_phase_output_dependency_rate"], 1.0)
        self.assertIn("end_to_end_abandonment_rate", row)
        self.assertEqual(row["end_to_end_task_success"], 1.0)
        self.assertIn("posthoc_safety_refusal_precision", row)
        self.assertEqual(row["posthoc_predicted_user_satisfaction"], row["automatic_eval_score"])
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
                "Audio ingress / capture",
                "Voice activity detection and segmentation",
                "Diarization",
                "Automatic speech recognition",
                "Spoken language understanding",
                "Dialog state tracking",
                "Policy and dialog management",
                "Tool / retrieval",
                "Natural language generation",
                "Text-to-speech",
                "Runtime",
                "Pipeline phases",
                "End to end",
                "Post hoc",
            ],
        )

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
        self.assertEqual(row["pipeline_failure_reason"], "text-to-speech failed")


if __name__ == "__main__":
    unittest.main()
