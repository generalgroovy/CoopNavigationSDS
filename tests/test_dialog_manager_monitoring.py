import json
import queue
import tempfile
import unittest
from pathlib import Path

from minillama.agent_a.agent_a_responder import TemplateAgentAResponder
from minillama.agent_b.agent_b_plugins import SimplePlannerAgentBPlugin
from minillama.agent_b.speech_io import SpeechTransport
from minillama.controller.dialog_manager import DialogManager
from minillama.controller.dialog_result import NullEventQueue
from minillama.controller.session_logging import MonitoringEventQueue, SessionLogger
from minillama.test_cases import DEFAULT_TEST_CASE, get_test_case


class DialogManagerMonitoringTests(unittest.TestCase):
    def test_controller_smoke_run_emits_conversation_step_logs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ui_queue = queue.Queue()
            logger = SessionLogger("dialog", tmpdir)
            event_queue = MonitoringEventQueue(ui_queue, logger)
            manager = DialogManager(
                get_test_case(DEFAULT_TEST_CASE),
                SimplePlannerAgentBPlugin(),
                num_turns=1,
                speech_transport=SpeechTransport(),
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

            jsonl_files = list(Path(tmpdir).glob("dialog-*.jsonl"))
            self.assertEqual(len(jsonl_files), 1)
            rows = [json.loads(line) for line in jsonl_files[0].read_text(encoding="utf-8").splitlines()]
            self.assertTrue(any(row["kind"] == "conversation.step" for row in rows))
            self.assertTrue(any(row["kind"] == "program.segment" for row in rows))

    def test_agent_a_elicits_multiple_compared_route_candidates(self):
        manager = DialogManager(
            get_test_case(DEFAULT_TEST_CASE).with_persona("distracted_multitasker"),
            SimplePlannerAgentBPlugin(),
            num_turns=3,
            speech_transport=SpeechTransport(),
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
