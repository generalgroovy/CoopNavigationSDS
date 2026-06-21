import json
import queue
import tempfile
import unittest
from pathlib import Path

from coop_navigation_sds.NaturalLanguageGeneration.caller.responder import TemplateAgentAResponder
from coop_navigation_sds.NaturalLanguageGeneration.assistant.plugin_registry import SimplePlannerAgentBPlugin, create_agent_b_plugin
from coop_navigation_sds.DialogManagement.speech_pipeline import SpeechPipelineTrace, SpeechSignal, SpeechTransport
from coop_navigation_sds.DialogManagement.speech_pipeline import SpeechPipelineConfig
from coop_navigation_sds.DialogManagement.manager import DialogManager, constraint_gap_missed
from coop_navigation_sds.DialogManagement.result import NullEventQueue
from coop_navigation_sds.ResultsAndArtifacts.logging import MonitoringEventQueue, SessionLogger
from coop_navigation_sds.TransportNetwork import DEFAULT_TEST_CASE, get_test_case
from coop_navigation_sds.TransportNetwork.constraints import acceptable_duration_limit
from coop_navigation_sds.TransportNetwork.routes import candidate_time_routes, route_text_from_steps


def fast_text_transport():
    return SpeechTransport(config=SpeechPipelineConfig(
        tts_engine="file",
        asr_engine="file",
        audio_dir=tempfile.mkdtemp(),
        realtime_enabled=False,
        playback_enabled=False,
    ))


class DialogManagerMonitoringTests(unittest.TestCase):
    def test_three_agent_b_models_complete_the_same_speech_pipeline(self):
        first_replies = {}
        for plugin_key in ("pareto", "robust", "diverse"):
            with self.subTest(plugin=plugin_key), tempfile.TemporaryDirectory() as tmpdir:
                manager = DialogManager(
                    get_test_case(DEFAULT_TEST_CASE),
                    create_agent_b_plugin(plugin_key, None),
                    num_turns=4,
                    speech_transport=SpeechTransport(config=SpeechPipelineConfig(
                        tts_engine="file",
                        asr_engine="file",
                        audio_dir=tmpdir,
                        realtime_enabled=False,
                        playback_enabled=False,
                    )),
                    agent_a_responder=TemplateAgentAResponder(),
                )

                result = manager.run(NullEventQueue())
                first_replies[plugin_key] = next(
                    text for speaker, text in result.conversation if speaker == "Agent B"
                )

                self.assertTrue(result.route_valid)
                self.assertTrue(result.route_reaches_goal)
                self.assertTrue(all(turn["pipeline_ok"] for turn in result.extra["speech_turns"]))
                self.assertTrue(any(
                    event["event_type"] == "stage_entered"
                    for event in result.extra["runtime_events"]
                ))

        self.assertEqual(len(set(first_replies.values())), 3)

    def test_controller_smoke_run_emits_conversation_step_logs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ui_queue = queue.Queue()
            logger = SessionLogger("dialog", tmpdir)
            event_queue = MonitoringEventQueue(ui_queue, logger)
            manager = DialogManager(
                get_test_case(DEFAULT_TEST_CASE),
                SimplePlannerAgentBPlugin(),
                num_turns=2,
                speech_transport=fast_text_transport(),
                agent_a_responder=TemplateAgentAResponder(),
                monitor=event_queue,
            )

            try:
                with event_queue.segment("dialog.run", test_case=DEFAULT_TEST_CASE, turns=1):
                    result = manager.run(event_queue)
            finally:
                event_queue.close()

            self.assertGreaterEqual(len(result.conversation), 2)
            self.assertTrue(result.metrics_text)
            self.assertIn("[Run]", result.metrics_text)
            self.assertIn("[Task]", result.metrics_text)
            self.assertIn("[Comparison]", result.metrics_text)
            self.assertIn("[Execution]", result.metrics_text)
            self.assertLessEqual(len(result.metrics_text.splitlines()), 30)
            self.assertTrue(all(" | " not in line for line in result.metrics_text.splitlines()))
            self.assertIsNotNone(result.extra["constraint_duration_min"])
            self.assertIn("constraint_duration_gap_min", result.extra)
            self.assertTrue(result.extra["runtime_events"])
            self.assertTrue(result.extra["preflight_viability"]["constraint_route_available"])
            self.assertTrue(any(
                event["event_type"] == "route_calculations_deferred"
                for event in result.extra["runtime_events"]
            ))
            self.assertFalse(any(
                event["event_type"] == "viability_check"
                and event["phase"] == "preflight"
                for event in result.extra["runtime_events"]
            ))
            self.assertEqual(result.extra["speech_turns"][0]["mode"], "speech")
            self.assertTrue(result.extra["speech_turns"][0]["pipeline_ok"])

            jsonl_files = list(Path(tmpdir).glob("dialog-*.jsonl"))
            self.assertEqual(len(jsonl_files), 1)
            rows = [json.loads(line) for line in jsonl_files[0].read_text(encoding="utf-8").splitlines()]
            self.assertTrue(any(row["kind"] == "conversation.step" for row in rows))
            self.assertTrue(any(row["kind"] == "program.segment" for row in rows))

    def test_controller_emits_readable_pipeline_phase_order(self):
        ui_queue = queue.Queue()
        manager = DialogManager(
            get_test_case(DEFAULT_TEST_CASE),
            SimplePlannerAgentBPlugin(),
            num_turns=2,
            speech_transport=fast_text_transport(),
            agent_a_responder=TemplateAgentAResponder(),
        )

        manager.run(ui_queue)

        events = []
        while not ui_queue.empty():
            events.append(ui_queue.get_nowait())
        agent_b_phases = [
            event[1]["phase"]
            for event in events
            if event[0] == "phase"
            and event[1].get("speaker") == "Agent B"
        ]
        self.assertEqual(agent_b_phases[:4], ["NLG", "TTS", "ASR", "NLU"])

    def test_agent_a_elicits_multiple_compared_route_candidates(self):
        manager = DialogManager(
            get_test_case(DEFAULT_TEST_CASE).with_persona("distracted_multitasker"),
            SimplePlannerAgentBPlugin(),
            num_turns=6,
            speech_transport=fast_text_transport(),
            agent_a_responder=TemplateAgentAResponder(),
        )

        result = manager.run(NullEventQueue())
        agent_a_replies = [
            text
            for speaker, text in result.conversation[1:]
            if speaker == "Agent A"
        ]

        self.assertGreaterEqual(result.extra["candidate_routes"], 2)
        self.assertIn("route_revisions", result.extra)
        self.assertIsNotNone(result.extra["constraint_duration_min"])
        self.assertIsNotNone(result.extra["constraint_duration_gap_min"])
        self.assertGreaterEqual(len(result.extra["stated_constraints"]), 1)
        self.assertTrue(any("Can you make it" in text for text in agent_a_replies))

    def test_agent_b_defers_secondary_constraints_until_agent_a_asks(self):
        manager = DialogManager(
            get_test_case(DEFAULT_TEST_CASE).with_persona("distracted_multitasker"),
            SimplePlannerAgentBPlugin(),
            num_turns=6,
            speech_transport=fast_text_transport(),
            agent_a_responder=TemplateAgentAResponder(),
        )

        result = manager.run(NullEventQueue())
        agent_b_replies = [text for speaker, text in result.conversation if speaker == "Agent B"]
        agent_a_replies = [text for speaker, text in result.conversation if speaker == "Agent A"]

        self.assertGreaterEqual(len(agent_b_replies), 3)
        self.assertNotIn("delay risk", agent_b_replies[0])
        self.assertNotIn("transfer miss risk", agent_b_replies[0])
        self.assertNotIn("delay risk", agent_b_replies[1])
        self.assertTrue(any("capacity" in text for text in agent_a_replies))
        self.assertIn("capacity", agent_b_replies[2])
        self.assertNotIn("delay risk", agent_b_replies[2])

    def test_controller_blocks_phase_progress_when_current_objective_failed(self):
        class InvalidRoutePlugin:
            name = "invalid-route-plugin"

            def run_agent_b(self, _state):
                return "Take Bravo to Alpha."

        class PrematureConstraintResponder:
            name = "premature-constraint-agent-a"

            def reply(self, *_args):
                return "Can you make it not near capacity?"

        manager = DialogManager(
            get_test_case(DEFAULT_TEST_CASE),
            InvalidRoutePlugin(),
            num_turns=5,
            speech_transport=fast_text_transport(),
            agent_a_responder=PrematureConstraintResponder(),
        )

        result = manager.run(NullEventQueue())

        self.assertEqual(result.extra["stated_constraints"], [])
        self.assertIn("current objective", result.conversation[-1][1])

    def test_controller_blocks_closure_before_constraint_phases_complete(self):
        class ValidRoutePlugin:
            name = "valid-route-plugin"

            def run_agent_b(self, _state):
                return "Take metro line M1 from Bravo to Harbor. It takes 12 minutes, with no changes."

        class PrematureClosureResponder:
            name = "premature-closure-agent-a"

            def reply(self, *_args):
                return "Thanks, that works. I'll take it."

        manager = DialogManager(
            get_test_case(DEFAULT_TEST_CASE),
            ValidRoutePlugin(),
            num_turns=5,
            speech_transport=fast_text_transport(),
            agent_a_responder=PrematureClosureResponder(),
        )

        result = manager.run(NullEventQueue())

        self.assertIn("not ready", result.conversation[2][1].lower())
        self.assertNotIn("I'll take it", result.conversation[2][1])

    def test_transfer_constraint_miss_uses_configured_slack(self):
        self.assertFalse(constraint_gap_missed({"line_change_gap": 1}, transfer_tolerance=1))
        self.assertFalse(constraint_gap_missed({"line_change_gap": 2}, transfer_tolerance=2))
        self.assertTrue(constraint_gap_missed({"line_change_gap": 2}, transfer_tolerance=1))
        self.assertTrue(constraint_gap_missed({"near_capacity_gap": 1}, transfer_tolerance=2))

    def test_agent_b_state_uses_last_pipeline_transcript_from_agent_a(self):
        class TestTextToSpeech:
            name = "test-tts"

            def synthesize(self, speaker, text):
                return SpeechSignal(speaker=speaker, text=text, audio={"path": "test.wav"})

        class TestSpeechToText:
            name = "test-asr"

            def transcribe(self, signal):
                if signal.speaker == "Agent A":
                    return "At 08:07, start Alpha and destination Echo."
                return signal.text

        class CapturingAgentB:
            name = "capturing-agent-b"

            def __init__(self):
                self.last_agent_a_text = None

            def run_agent_b(self, state):
                self.last_agent_a_text = state.conversation[-1][1]
                return "Take Alpha to Golf to Foxtrot to Echo."

        agent_b = CapturingAgentB()
        manager = DialogManager(
            get_test_case("midday_transfer"),
            agent_b,
            num_turns=2,
            speech_transport=SpeechTransport(
                tts_engine=TestTextToSpeech(),
                asr_engine=TestSpeechToText(),
                config=SpeechPipelineConfig(
                    realtime_enabled=False,
                    playback_enabled=False,
                ),
            ),
            agent_a_responder=TemplateAgentAResponder(),
        )

        result = manager.run(NullEventQueue())

        self.assertEqual(agent_b.last_agent_a_text, "At 08:07, start Alpha and destination Echo.")
        self.assertEqual(result.conversation[0][1], "At 08:07, start Alpha and destination Echo.")

    def test_agent_a_stops_early_after_repeated_invalid_routes(self):
        class InvalidRoutePlugin:
            name = "invalid-route-plugin"

            def run_agent_b(self, _state):
                return "Take Alpha to Alpha."

        manager = DialogManager(
            get_test_case(DEFAULT_TEST_CASE),
            InvalidRoutePlugin(),
            num_turns=8,
            speech_transport=fast_text_transport(),
            agent_a_responder=TemplateAgentAResponder(),
            invalid_route_limit=2,
        )

        result = manager.run(NullEventQueue())

        self.assertEqual(result.extra["early_stop_reason"], "invalid_route_limit")
        self.assertEqual(result.extra["invalid_route_count"], 2)
        self.assertLess(len(result.conversation), 8)
        self.assertEqual(result.conversation[-1][0], "Agent A")
        self.assertIn("stop here", result.conversation[-1][1])

    def test_agent_a_stops_early_when_time_frame_keep_being_missed(self):
        test_case = get_test_case(DEFAULT_TEST_CASE).with_persona("distracted_multitasker")
        scenario = test_case.scenario
        duration_limit = acceptable_duration_limit(scenario, test_case.persona)
        slow_replies = [
            route_text_from_steps(steps)
            for duration, _route, steps in candidate_time_routes(
                scenario["start_station"],
                scenario["destination_station"],
                scenario["start_time_min"],
                scenario["transfer_time_min"],
                limit=80,
                max_extra_stops=8,
                max_paths=50000,
                allowed_modes=("metro", "tram", "bus", "walking"),
            )
            if duration > duration_limit
        ][:3]
        self.assertGreaterEqual(len(slow_replies), 2)

        class TimeFrameMissPlugin:
            name = "time-frame-miss-plugin"

            def __init__(self):
                self.replies = slow_replies

            def run_agent_b(self, state):
                return self.replies[min(state.turn, len(self.replies) - 1)]

        manager = DialogManager(
            test_case,
            TimeFrameMissPlugin(),
            num_turns=8,
            speech_transport=fast_text_transport(),
            agent_a_responder=TemplateAgentAResponder(),
            constraint_miss_limit=1,
        )

        result = manager.run(NullEventQueue())

        self.assertEqual(result.extra["early_stop_reason"], "time_frame_miss_limit")
        self.assertEqual(result.extra["time_frame_miss_count"], 2)
        self.assertEqual(result.conversation[-1][0], "Agent A")
        self.assertIn("too slow", result.conversation[-1][1])

    def test_dialog_ends_when_agent_a_closes_call(self):
        manager = DialogManager(
            get_test_case(DEFAULT_TEST_CASE),
            SimplePlannerAgentBPlugin(),
            num_turns=9,
            speech_transport=fast_text_transport(),
            agent_a_responder=TemplateAgentAResponder(),
        )

        result = manager.run(NullEventQueue())

        self.assertEqual(result.extra["early_stop_reason"], "agent_a_closed")
        self.assertEqual(result.conversation[-1][0], "Agent A")
        self.assertIn("Thanks", result.conversation[-1][1])

    def test_dialog_manager_runs_file_backed_speech_for_both_agents(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DialogManager(
                get_test_case("airport_connection"),
                SimplePlannerAgentBPlugin(),
                num_turns=2,
                speech_transport=SpeechTransport(
                    config=SpeechPipelineConfig(
                        tts_engine="file",
                        asr_engine="file",
                        audio_dir=tmpdir,
                        playback_enabled=False,
                        realtime_enabled=False,
                    )
                ),
                agent_a_responder=TemplateAgentAResponder(),
            )

            result = manager.run(NullEventQueue())

            speakers = {turn["speaker"] for turn in result.extra["speech_turns"]}
            audio_paths = [
                Path(turn["audio"]["path"])
                for turn in result.extra["speech_turns"]
                if isinstance(turn.get("audio"), dict)
            ]

            self.assertIn("Agent A", speakers)
            self.assertIn("Agent B", speakers)
            self.assertTrue(audio_paths)
            self.assertTrue(all(path.exists() for path in audio_paths))
            self.assertGreater(result.extra["constraint_delay_probability"], 0.0)

    def test_simple_dialog_turns_stay_natural_length(self):
        manager = DialogManager(
            get_test_case("airport_connection"),
            SimplePlannerAgentBPlugin(),
            num_turns=3,
            speech_transport=fast_text_transport(),
            agent_a_responder=TemplateAgentAResponder(),
        )

        result = manager.run(NullEventQueue())
        word_counts = [len(text.split()) for _, text in result.conversation]

        self.assertLessEqual(max(word_counts), 24)
        self.assertLessEqual(len(result.conversation), 6)

    def test_turn_timing_is_capped_at_configured_budget(self):
        class SlowTransport:
            description = "slow-test-transport"
            asr_engine = type("AsrEngine", (), {"name": "test-asr", "pattern_key": "slow"})()

            def transmit_trace(self, speaker, text):
                return SpeechPipelineTrace(
                    speaker=speaker,
                    generated_text=text,
                    outgoing_text=text,
                    incoming_transcript=text,
                    outgoing_enabled=True,
                    incoming_enabled=True,
                    tts_engine="test",
                    asr_engine="test",
                    pattern_key="slow",
                    simulated_duration_sec=25.0,
                    mode="speech",
                )

        manager = DialogManager(
            get_test_case("airport_connection"),
            SimplePlannerAgentBPlugin(),
            num_turns=2,
            speech_transport=SlowTransport(),
            agent_a_responder=TemplateAgentAResponder(),
            max_turn_elapsed_sec=20.0,
        )

        result = manager.run(NullEventQueue())

        elapsed_values = [turn["turn_elapsed_sec"] for turn in result.extra["timing_turns"]]
        raw_values = [turn["raw_turn_elapsed_sec"] for turn in result.extra["timing_turns"]]
        self.assertTrue(elapsed_values)
        self.assertLessEqual(max(elapsed_values), 20.0)
        self.assertGreater(max(raw_values), 20.0)
        self.assertGreater(result.extra["turn_over_budget_count"], 0)
        self.assertEqual(len(result.extra["phase_timings"]), len(result.conversation))
        first_breakdown = result.extra["phase_timings"][0]
        self.assertIn("audio_duration_sec", first_breakdown)
        self.assertIn("speech_pipeline_wall_sec", first_breakdown)
        self.assertIn("observed_turn_sec", first_breakdown)

    def test_conversation_timer_starts_when_agents_are_ready(self):
        class CapturingQueue:
            def __init__(self):
                self.events = []

            def put(self, event):
                self.events.append(event)

        event_queue = CapturingQueue()
        manager = DialogManager(
            get_test_case("airport_connection"),
            SimplePlannerAgentBPlugin(),
            num_turns=2,
            speech_transport=fast_text_transport(),
            agent_a_responder=TemplateAgentAResponder(),
        )

        result = manager.run(event_queue)

        kinds = [event[0] for event in event_queue.events]
        self.assertIn("timer_start", kinds)
        self.assertLess(kinds.index("timer_start"), kinds.index("message"))
        self.assertTrue(result.extra["runtime_events"])
        self.assertGreaterEqual(result.runtime_sec, result.extra["runtime_events"][0]["elapsed_sec"])
