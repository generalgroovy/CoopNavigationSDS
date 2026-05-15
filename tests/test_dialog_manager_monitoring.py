import json
import queue
import tempfile
import unittest
from pathlib import Path

from minillama.agent_a.agent_a_responder import TemplateAgentAResponder
from minillama.agent_b.agent_b_plugins import SimplePlannerAgentBPlugin
from minillama.agent_b.speech_io import SpeechTransport
from minillama.controller.dialog_manager import DialogManager
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

            jsonl_files = list(Path(tmpdir).glob("dialog-*.jsonl"))
            self.assertEqual(len(jsonl_files), 1)
            rows = [json.loads(line) for line in jsonl_files[0].read_text(encoding="utf-8").splitlines()]
            self.assertTrue(any(row["kind"] == "conversation.step" for row in rows))
            self.assertTrue(any(row["kind"] == "program.segment" for row in rows))
