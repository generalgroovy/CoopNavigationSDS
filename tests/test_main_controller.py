import queue
import threading
import unittest

from minillama.controller.main import default_run_config, select_run_config, start_gui_thread


class MainControllerTests(unittest.TestCase):
    def test_default_config_keeps_gui_optional(self):
        config = default_run_config()

        self.assertIn("gui_enabled", config)
        self.assertIsInstance(config["gui_enabled"], bool)

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

    def test_start_gui_thread_runs_dialog_runner_independently(self):
        caller_thread = threading.get_ident()
        seen = {}
        done = threading.Event()

        def fake_dialog_runner(ui_queue, scenario, gui_mode):
            seen["thread"] = threading.get_ident()
            seen["queue"] = ui_queue
            seen["scenario"] = scenario
            seen["gui_mode"] = gui_mode
            done.set()

        ui_queue = queue.Queue()
        scenario = {"name": "Threaded GUI"}

        thread = start_gui_thread(
            ui_queue,
            scenario,
            gui_mode="conversation",
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


if __name__ == "__main__":
    unittest.main()
