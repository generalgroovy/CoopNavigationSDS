import queue
import sys
import threading
import types
import unittest
from unittest.mock import patch

from minillama.controller.main import BroadcastQueue, conversation_worker, default_run_config, normalize_run_config, select_run_config, start_gui_thread, start_gui_threads


class MainControllerTests(unittest.TestCase):
    def test_default_config_is_speech_first_and_keeps_gui_optional(self):
        config = default_run_config()

        self.assertIn("gui_enabled", config)
        self.assertIsInstance(config["gui_enabled"], bool)
        self.assertFalse(config["gui_enabled"])
        self.assertEqual(config["num_turns"], 7)
        self.assertEqual(config["run_mode"], "speech")
        self.assertTrue(config["speech_incoming_enabled"])
        self.assertTrue(config["speech_outgoing_enabled"])
        self.assertTrue(config["speech_playback_enabled"])
        self.assertTrue(config["speech_realtime_enabled"])
        self.assertEqual(config["speech_scope"], "both")
        self.assertEqual(config["tts_engine"], "sapi")
        self.assertEqual(config["asr_engine"], "sapi")
        self.assertIn("persona_key", config)
        self.assertIn("gui_refresh_ms", config)
        self.assertGreaterEqual(config["gui_refresh_ms"], 50)
        self.assertEqual(config["max_turn_elapsed_sec"], 3.0)
        self.assertEqual(config["calculation_max_time_sec"], 5.0)
        self.assertEqual(config["agent_a_words_per_minute"], 185)
        self.assertEqual(config["agent_b_words_per_minute"], 195)
        self.assertEqual(config["min_utterance_sec"], 0.35)
        self.assertEqual(config["max_utterance_sec"], 1.6)
        self.assertIn("network_data_card_enabled", config)
        self.assertFalse(config["network_data_card_enabled"])
        self.assertIn("metric_config", config)
        self.assertTrue(config["metric_config"]["asr_wer"])

    def test_metric_switches_are_normalized(self):
        config = default_run_config()
        config["metric_config"] = {"asr_wer": False}

        normalized = normalize_run_config(config)

        self.assertFalse(normalized["metric_config"]["asr_wer"])
        self.assertTrue(normalized["metric_config"]["audio_turn_latency"])

    def test_runtime_lengths_are_configurable_and_clamped(self):
        config = default_run_config()
        config["max_turn_elapsed_sec"] = 12.0
        config["calculation_max_time_sec"] = 9.0

        normalized = normalize_run_config(config)

        self.assertEqual(normalized["max_turn_elapsed_sec"], 12.0)
        self.assertEqual(normalized["calculation_max_time_sec"], 9.0)

        config["max_turn_elapsed_sec"] = 99.0
        config["calculation_max_time_sec"] = 99.0
        normalized = normalize_run_config(config)

        self.assertEqual(normalized["max_turn_elapsed_sec"], 20.0)
        self.assertEqual(normalized["calculation_max_time_sec"], 20.0)

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

    def test_main_falls_back_to_simple_agent_b_when_weights_are_missing(self):
        import minillama.controller.main as controller_main

        captured = {}
        fake_model_runtime = types.ModuleType("minillama.model.model_runtime")
        fake_model_runtime.create_model_adapter = lambda: (_ for _ in ()).throw(RuntimeError("missing weights"))
        original_gui_enabled = controller_main.GUI_ENABLED
        original_session_profile = controller_main.SESSION_LOG_PROFILE
        try:
            controller_main.GUI_ENABLED = False
            controller_main.SESSION_LOG_PROFILE = "off"

            def capture_worker(_event_queue, model_adapter, run_config):
                captured["model_adapter"] = model_adapter
                captured["agent_b_plugin"] = run_config["agent_b_plugin"]
                captured["llm_agent_a"] = run_config["llm_agent_a"]

            with patch.dict(sys.modules, {"minillama.model.model_runtime": fake_model_runtime}):
                with patch.object(controller_main, "write_network_research_artifacts"):
                    with patch.object(controller_main, "conversation_worker", side_effect=capture_worker):
                        controller_main.main()
        finally:
            controller_main.GUI_ENABLED = original_gui_enabled
            controller_main.SESSION_LOG_PROFILE = original_session_profile

        self.assertIsNone(captured["model_adapter"])
        self.assertEqual(captured["agent_b_plugin"], "simple")
        self.assertFalse(captured["llm_agent_a"])

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

    def test_start_gui_threads_splits_conversation_metrics_and_network(self):
        seen = []
        lock = threading.Lock()

        def fake_dialog_runner(
            ui_queue,
            scenario,
            gui_mode,
            network_data_card_enabled,
            gui_refresh_ms,
            window_layout_index,
            window_layout_count,
        ):
            with lock:
                seen.append((
                    ui_queue,
                    scenario,
                    gui_mode,
                    network_data_card_enabled,
                    gui_refresh_ms,
                    window_layout_index,
                    window_layout_count,
                    threading.get_ident(),
                ))

        scenario = {"name": "Threaded GUI"}

        sink, threads = start_gui_threads(
            scenario,
            gui_mode="conversation",
            network_data_card_enabled=True,
            gui_refresh_ms=250,
            dialog_runner=fake_dialog_runner,
        )
        for thread in threads:
            thread.join(timeout=2.0)

        self.assertIsInstance(sink, BroadcastQueue)
        self.assertEqual(len(threads), 3)
        self.assertEqual({entry[2] for entry in seen}, {"conversation", "metrics", "network"})
        self.assertTrue(any(entry[3] for entry in seen if entry[2] == "network"))
        self.assertTrue(all(entry[1] is scenario for entry in seen))
        self.assertEqual({entry[4] for entry in seen}, {250})
        self.assertEqual({entry[6] for entry in seen}, {3})
        self.assertEqual({entry[5] for entry in seen}, {0, 1, 2})

        sink.put(("message", "Agent A", "hello"))
        self.assertTrue(all(not entry[0].empty() for entry in seen))


if __name__ == "__main__":
    unittest.main()
