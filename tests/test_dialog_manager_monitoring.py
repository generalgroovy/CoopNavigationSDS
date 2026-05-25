import json
import queue
import tempfile
import unittest
from pathlib import Path

from minillama.agent_a.agent_a_responder import TemplateAgentAResponder
from minillama.agent_b.plugin_registry import SimplePlannerAgentBPlugin
from minillama.agent_b.speech_io import SpeechSignal, SpeechTransport
from minillama.agent_b.speech_io import SpeechPipelineConfig
from minillama.controller.dialog_manager import DialogManager
from minillama.controller.dialog_result import NullEventQueue
from minillama.controller.session_logging import MonitoringEventQueue, SessionLogger
from minillama.test_cases import DEFAULT_TEST_CASE, get_test_case


def fast_text_transport():
    return SpeechTransport(config=SpeechPipelineConfig(
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
            self.assertTrue(result.extra["metric_snapshots"])
            self.assertEqual(result.extra["speech_turns"][0]["mode"], "pure_text")
            self.assertTrue(result.extra["speech_turns"][0]["pipeline_ok"])

            jsonl_files = list(Path(tmpdir).glob("dialog-*.jsonl"))
            self.assertEqual(len(jsonl_files), 1)
            rows = [json.loads(line) for line in jsonl_files[0].read_text(encoding="utf-8").splitlines()]
            self.assertTrue(any(row["kind"] == "conversation.step" for row in rows))
            self.assertTrue(any(row["kind"] == "system" and row["name"] == "metric.snapshot" for row in rows))
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
        self.assertTrue(any("one" in text and "valid route" in text for text in agent_a_replies))
        self.assertTrue(any("less full" in text or "fewer line changes" in text for text in agent_a_replies))

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

    def test_agent_a_stops_early_when_constraints_keep_being_missed(self):
        class ConstraintMissPlugin:
            name = "constraint-miss-plugin"

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
            ConstraintMissPlugin(),
            num_turns=8,
            speech_transport=fast_text_transport(),
            agent_a_responder=TemplateAgentAResponder(),
            constraint_miss_limit=2,
        )

        result = manager.run(NullEventQueue())

        self.assertEqual(result.extra["early_stop_reason"], "constraint_miss_limit")
        self.assertEqual(result.extra["constraint_miss_count"], 2)
        self.assertEqual(result.conversation[-1][0], "Agent A")
        self.assertIn("constraints", result.conversation[-1][1])

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
