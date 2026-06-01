"""Run MiniLlama from explicit values in this script, without a startup dialog."""
from pathlib import Path
import queue
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minillama.controller.main import conversation_worker, default_run_config
from minillama.controller.session_logging import MonitoringEventQueue


SCRIPT_CONFIG = {
    "run_mode": "speech",
    "agent_b_plugin": "simple",
    "test_case_key": "morning_peak_cross_city",
    "persona_key": "focused_commuter",
    "num_turns": 5,
    "speech_pattern_key": "mostly_clean",
    "speech_engine": "file",
    "tts_engine": "file",
    "asr_engine": "file",
    "speech_playback_enabled": False,
    "speech_realtime_enabled": False,
    "gui_enabled": False,
}


def main():
    run_config = default_run_config()
    run_config.update(SCRIPT_CONFIG)
    ui_queue = queue.Queue()
    event_queue = MonitoringEventQueue(ui_queue, logger=None)
    conversation_worker(event_queue, model_adapter=None, run_config=run_config)

    while not ui_queue.empty():
        event = ui_queue.get()
        if event and event[0] in {"message", "system", "warning", "done"}:
            print(event)


if __name__ == "__main__":
    main()
