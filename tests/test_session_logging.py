import json
import queue
import tempfile
import unittest
from pathlib import Path

from minillama.controller.session_logging import MonitoringEventQueue, SessionLogger


class SessionLoggingTests(unittest.TestCase):
    def test_session_logger_writes_structured_session_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = SessionLogger("unit", tmpdir)
            with logger.segment("alpha", source="test"):
                logger.log_step(
                    turn=1,
                    speaker="Agent A",
                    utterance="hello world",
                    metrics={"route_valid": True, "route": ["A", "B"]},
                )
                logger.log_metric_snapshot({"turn": 1, "message_count": 1, "candidate_routes": 0})
            logger.close()

            jsonl_files = list(Path(tmpdir).glob("unit-*.jsonl"))
            self.assertEqual(len(jsonl_files), 1)
            summary_files = list(Path(tmpdir).glob("unit-*.log"))
            self.assertEqual(len(summary_files), 1)

            rows = [json.loads(line) for line in jsonl_files[0].read_text(encoding="utf-8").splitlines()]
            kinds = [row["kind"] for row in rows]
            self.assertIn("program.segment", kinds)
            self.assertIn("conversation.step", kinds)
            self.assertTrue(any(row["kind"] == "system" and row["name"] == "metric.snapshot" for row in rows))
            self.assertTrue(any(row["kind"] == "system" and row["name"] == "session.end" for row in rows))

            summary = json.loads((Path(tmpdir) / f"{jsonl_files[0].stem}-summary.json").read_text(encoding="utf-8"))
            self.assertGreaterEqual(summary["events"], 3)

    def test_monitoring_queue_forwards_ui_events_and_logs_structured_payloads(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ui_queue = queue.Queue()
            logger = SessionLogger("unit", tmpdir)
            event_queue = MonitoringEventQueue(ui_queue, logger)

            event_queue.put(("message", "Agent A", "hello"))
            event_queue.put(("metric_snapshot", {"turn": 1, "message_count": 1}))
            event_queue.put(("warning", "check route"))
            event_queue.put(("done",))
            event_queue.close()

            forwarded = [ui_queue.get_nowait(), ui_queue.get_nowait(), ui_queue.get_nowait(), ui_queue.get_nowait()]
            self.assertEqual(forwarded[0][0], "message")
            self.assertEqual(forwarded[1][0], "metric_snapshot")
            self.assertEqual(forwarded[2][0], "warning")
            self.assertEqual(forwarded[3][0], "done")

            jsonl_files = list(Path(tmpdir).glob("unit-*.jsonl"))
            rows = [json.loads(line) for line in jsonl_files[0].read_text(encoding="utf-8").splitlines()]
            kinds = [row["kind"] for row in rows]
            self.assertIn("conversation.step", kinds)
            self.assertIn("system", kinds)
            self.assertTrue(any(row["kind"] == "system" and row["name"] == "metric.snapshot" for row in rows))
            self.assertTrue(any(row["kind"] == "system" and row["name"] == "session.end" for row in rows))
