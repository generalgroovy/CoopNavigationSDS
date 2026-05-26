import queue
import threading
import unittest

from minillama.controller.main import conversation_worker, default_run_config, select_run_config, start_gui_thread


class MainControllerTests(unittest.TestCase):
    def test_default_config_keeps_gui_optional(self):
        config = default_run_config()

        self.assertIn("gui_enabled", config)
        self.assertIsInstance(config["gui_enabled"], bool)
        self.assertEqual(config["num_turns"], 5)
        self.assertEqual(config["run_mode"], "pure_text")
        self.assertFalse(config["speech_playback_enabled"])
        self.assertFalse(config["speech_realtime_enabled"])
        self.assertEqual(config["speech_scope"], "none")
        self.assertIn("persona_key", config)
        self.assertIn("network_data_card_enabled", config)
        self.assertFalse(config["network_data_card_enabled"])

    def test_select_run_config_skips_startup_gui_when_disabled(self):
        import minillama.controller.main as controller_main

        original = controller_main.GUI_ENABLED
        try:
            controller_main.GUI_ENABLED = False

            config = select_run_config()

        finally:
            controller_main.GUI_ENABLED = original

        self.assertFalse(config["gui_enabled"])
        self.assertEqual(config["test_case_key"], default_run_config()["test_case_key"])

    def test_conversation_worker_reports_invalid_speech_pipeline(self):
        class FakeEventQueue:
            def __init__(self):
                self.events = []
                self.closed = False

            def put(self, event):
                self.events.append(event)

            def close(self):
                self.closed = True

        config = default_run_config()
        config.update({
            "run_mode": "speech",
            "agent_b_plugin": "simple",
            "tts_engine": "loopback",
            "asr_engine": "loopback",
        })
        event_queue = FakeEventQueue()

        conversation_worker(event_queue, None, config)

        self.assertTrue(event_queue.closed)
        warnings = [event for event in event_queue.events if event[0] == "warning"]
        self.assertTrue(any("Speech pipeline failed" in event[1] for event in warnings))
        self.assertTrue(any("Troubleshooting" in event[1] for event in warnings))

    def test_start_gui_thread_runs_dialog_runner_independently(self):
        caller_thread = threading.get_ident()
        seen = {}
        done = threading.Event()

        def fake_dialog_runner(ui_queue, scenario, gui_mode, network_data_card_enabled):
            seen["thread"] = threading.get_ident()
            seen["queue"] = ui_queue
            seen["scenario"] = scenario
            seen["gui_mode"] = gui_mode
            seen["network_data_card_enabled"] = network_data_card_enabled
            done.set()

        ui_queue = queue.Queue()
        scenario = {"name": "Threaded GUI"}

        thread = start_gui_thread(
            ui_queue,
            scenario,
            gui_mode="conversation",
            network_data_card_enabled=True,
            dialog_runner=fake_dialog_runner,
        )
        thread.join(timeout=2.0)

        self.assertFalse(thread.is_alive())
        self.assertTrue(done.is_set())
        self.assertEqual(thread.name, "minillama-gui")
        self.assertNotEqual(seen["thread"], caller_thread)
        self.assertIs(seen["queue"], ui_queue)
        self.assertIs(seen["scenario"], scenario)
        self.assertEqual(seen["gui_mode"], "conversation")
        self.assertTrue(seen["network_data_card_enabled"])


if __name__ == "__main__":
    unittest.main()
