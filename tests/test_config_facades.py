from pathlib import Path
import tempfile
import unittest

from coop_navigation_sds.Configuration.runtime import NUM_TURNS, RESULTS_DIR, SESSION_LOG_DIR
from coop_navigation_sds.Configuration.schema import resolve_result_group, resolve_results_root
from coop_navigation_sds.Configuration.speech import (
    AGENT_B_PLUGIN,
    SPEECH_AUDIO_DIR,
    SPEECH_PATTERNS,
    SPEECH_PLAYBACK_ENABLED,
    SPEECH_REALTIME_ENABLED,
)
from coop_navigation_sds.Configuration.travel import DEVICE, MODEL
from coop_navigation_sds.NaturalLanguageGeneration.caller.config import LLM_AGENT_A
from coop_navigation_sds.DialogManagement.speech_pipeline import SpeechPipelineConfig, SpeechTransport


class ConfigModuleTests(unittest.TestCase):
    def test_configuration_exports_core_values(self):
        self.assertTrue(MODEL)
        self.assertTrue(DEVICE)
        self.assertTrue(SPEECH_PATTERNS)
        self.assertTrue(NUM_TURNS)
        self.assertTrue(AGENT_B_PLUGIN)
        self.assertEqual(Path(RESULTS_DIR), Path(__file__).resolve().parents[1] / "results")
        self.assertEqual(Path(resolve_results_root("results")), Path(RESULTS_DIR))
        self.assertTrue(SESSION_LOG_DIR)
        self.assertFalse(LLM_AGENT_A)

    def test_default_speech_settings_play_and_wait(self):
        self.assertTrue(SPEECH_PLAYBACK_ENABLED)
        self.assertTrue(SPEECH_REALTIME_ENABLED)
        self.assertTrue(SPEECH_AUDIO_DIR)

    def test_result_groups_remain_beneath_the_single_results_root(self):
        root = Path(resolve_results_root("results"))
        grouped = Path(resolve_result_group(root, "agent_b/primary/01-small/userlm"))
        self.assertEqual(grouped, root / "01-small" / "userlm")
        for invalid in ("../outside", "/absolute", "agent_b/../outside"):
            with self.assertRaises(ValueError):
                resolve_result_group(root, invalid)

    def test_file_pipeline_produces_audio_and_transcript(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            transport = SpeechTransport(config=SpeechPipelineConfig(
                tts_engine="file",
                asr_engine="file",
                audio_dir=tmpdir,
                playback_enabled=False,
                realtime_enabled=False,
            ))
            trace = transport.transmit_trace("Agent A", "Need Alpha to Echo.")
            self.assertTrue(Path(trace.audio["path"]).exists())

        self.assertEqual(trace.incoming_transcript, "Need Alpha to Echo.")
        self.assertTrue(trace.incoming_enabled)
        self.assertTrue(trace.outgoing_enabled)
        self.assertIn("tts=file:asr=file", transport.description)


if __name__ == "__main__":
    unittest.main()
