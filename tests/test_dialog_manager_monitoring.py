import json
import queue
import tempfile
import unittest
from pathlib import Path

from minillama.caller.responder import TemplateAgentAResponder
from minillama.assistant.plugin_registry import SimplePlannerAgentBPlugin
from minillama.speech.io import SpeechPipelineTrace, SpeechSignal, SpeechTransport
from minillama.speech.io import SpeechPipelineConfig
from minillama.orchestration.dialog_manager import DialogManager, constraint_gap_missed
from minillama.orchestration.dialog_result import NullEventQueue
from minillama.orchestration.session_logging import MonitoringEventQueue, SessionLogger
from minillama.scenarios import DEFAULT_TEST_CASE, get_test_case


def fast_text_transport():
    return SpeechTransport(config=SpeechPipelineConfig(
        mode="pure_text",
        incoming_enabled=False,
        outgoing_enabled=False,
        scope="none",
        realtime_enabled=False,
        playback_enabled=False,
    ))


class DialogManagerMonitoringTests(unittest.TestCase):
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
            self.assertIn("Displayed line sequence:", result.metrics_text)
            self.assertIn("Reference line sequence:", result.metrics_text)
            self.assertIn("Constraint route:", result.metrics_text)
            self.assertIsNotNone(result.extra["constraint_duration_min"])
            self.assertIn("constraint_duration_gap_min", result.extra)
            self.assertTrue(result.extra["runtime_events"])
            self.assertTrue(result.extra["preflight_viability"]["constraint_route_available"])
            self.assertEqual(result.extra["speech_turns"][0]["mode"], "pure_text")
            self.assertTrue(result.extra["speech_turns"][0]["pipeline_ok"])

            jsonl_files = list(Path(tmpdir).glob("dialog-*.jsonl"))
            self.assertEqual(len(jsonl_files), 1)
            rows = [json.loads(line) for line in jsonl_files[0].read_text(encoding="utf-8").splitlines()]
            self.assertTrue(any(row["kind"] == "conversation.step" for row in rows))
            self.assertTrue(any(row["kind"] == "program.segment" for row in rows))

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
        self.assertGreaterEqual(result.extra["route_revisions"], 1)
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

        self.assertGreaterEqual(len(agent_b_replies), 3)
        self.assertNotIn("delay risk", agent_b_replies[0])
        self.assertNotIn("transfer miss risk", agent_b_replies[0])
        self.assertNotIn("delay risk", agent_b_replies[1])
        self.assertIn("delay risk", agent_b_replies[2])

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
                    return "heard start Alpha destination Echo"
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
                    mode="speech",
                    incoming_enabled=True,
                    outgoing_enabled=True,
                    scope="both",
                    realtime_enabled=False,
                    playback_enabled=False,
                ),
            ),
            agent_a_responder=TemplateAgentAResponder(),
        )

        result = manager.run(NullEventQueue())

        self.assertEqual(agent_b.last_agent_a_text, "heard start Alpha destination Echo")
        self.assertEqual(result.conversation[0][1], "heard start Alpha destination Echo")

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
        class TimeFrameMissPlugin:
            name = "time-frame-miss-plugin"

            def __init__(self):
                self.replies = [
                    "Take Bravo to Alpha to Golf to November to Uniform to Birch to Ivy to Harbor.",
                    "Take Bravo to Alpha to Golf to Mike to Sierra to Yankee to Elm to Flint to Grove to Harbor.",
                    "Take Bravo to Alpha to Hotel to Oscar to Victor to Cedar to Jasper to Ivy to Harbor.",
            ]

            def run_agent_b(self, state):
                return self.replies[min(state.turn, len(self.replies) - 1)]

        manager = DialogManager(
            get_test_case(DEFAULT_TEST_CASE).with_persona("distracted_multitasker"),
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
                        mode="speech",
                        incoming_enabled=True,
                        outgoing_enabled=True,
                        scope="both",
                        engine="file",
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
