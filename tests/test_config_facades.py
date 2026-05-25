from pathlib import Path
import tempfile
import unittest

from minillama.agent_b.config import (
    AGENT_B_PLUGIN,
    SPEECH_AUDIO_DIR,
    SPEECH_ENGINE,
    SPEECH_INCOMING_ENABLED,
    SPEECH_OUTGOING_ENABLED,
    SPEECH_PATTERNS,
    SPEECH_PLAYBACK_ENABLED,
    SPEECH_REALTIME_ENABLED,
    SPEECH_SCOPE,
)
from minillama.agent_b.speech_io import SpeechPipelineConfig, SpeechTransport
from minillama.agent_b.plugin_registry import AgentBPluginConfig, available_agent_b_plugin_keys
from minillama.controller.config import NUM_TURNS, SESSION_LOG_DIR
from minillama.model.config import DEVICE, MODEL
from minillama.model.network_overview import build_network_overview
from minillama.view.config import GUI_COLORS, GUI_WIDTH, MAP_ROUTE_LINE_WIDTH
from minillama.view.gui import DialogWindow


class ConfigModuleTests(unittest.TestCase):
    def test_model_config_exports_model_values(self):
        self.assertTrue(MODEL)
        self.assertTrue(DEVICE)
        self.assertTrue(SPEECH_PATTERNS)

    def test_view_config_exports_gui_values(self):
        self.assertTrue(GUI_WIDTH)
        self.assertTrue(GUI_COLORS)
        self.assertTrue(MAP_ROUTE_LINE_WIDTH)

    def test_controller_config_exports_controller_values(self):
        self.assertTrue(NUM_TURNS)
        self.assertTrue(AGENT_B_PLUGIN)
        self.assertTrue(SESSION_LOG_DIR)

    def test_default_speech_pipeline_plays_and_listens(self):
        self.assertTrue(SPEECH_INCOMING_ENABLED)
        self.assertTrue(SPEECH_OUTGOING_ENABLED)
        self.assertTrue(SPEECH_PLAYBACK_ENABLED)
        self.assertTrue(SPEECH_REALTIME_ENABLED)
        self.assertEqual(SPEECH_SCOPE, "both")
        self.assertEqual(SPEECH_ENGINE, "file")
        self.assertTrue(SPEECH_AUDIO_DIR)

        with tempfile.TemporaryDirectory() as tmpdir:
            transport = SpeechTransport(config=SpeechPipelineConfig(
                audio_dir=tmpdir,
                playback_enabled=False,
                realtime_enabled=False,
            ))
            trace = transport.transmit_trace("Agent A", "Need Alpha to Echo.")
            self.assertTrue(Path(trace.audio["path"]).exists())

        self.assertEqual(trace.incoming_transcript, "Need Alpha to Echo.")
        self.assertEqual(trace.outgoing_text, "Need Alpha to Echo.")
        self.assertTrue(trace.incoming_enabled)
        self.assertTrue(trace.outgoing_enabled)
        self.assertEqual(trace.tts_engine, "wavefile-tts")
        self.assertEqual(trace.asr_engine, "wavefile-asr")
        self.assertIn("outgoing+incoming:clean:both", transport.description)

    def test_gui_only_expands_speech_trace_when_pipeline_changes_message(self):
        self.assertFalse(DialogWindow.should_show_speech_trace(None))
        self.assertFalse(DialogWindow.should_show_speech_trace({
            "outgoing_enabled": False,
            "incoming_enabled": False,
            "outgoing_text": "Need Alpha to Echo.",
            "incoming_transcript": "Need Alpha to Echo.",
        }))
        self.assertTrue(DialogWindow.should_show_speech_trace({
            "outgoing_enabled": True,
            "incoming_enabled": True,
            "outgoing_text": "Take Red to Echo.",
            "incoming_transcript": "Take red to Echo.",
        }))

    def test_agent_b_plugin_config_exposes_registry(self):
        self.assertIn("minillama", available_agent_b_plugin_keys())
        self.assertIn("simple", available_agent_b_plugin_keys())
        self.assertTrue(AgentBPluginConfig("minillama").needs_model)
        self.assertTrue(AgentBPluginConfig("llm").needs_model)
        self.assertFalse(AgentBPluginConfig("simple").needs_model)

    def test_network_overview_exposes_complete_tables(self):
        overview = build_network_overview(480)

        self.assertGreater(overview.line_count, 0)
        self.assertGreater(overview.station_count, 0)
        self.assertEqual(overview.line_count, len(overview.lines))
        self.assertEqual(overview.station_count, len(overview.stations))
        self.assertTrue(overview.lines[0].route)
        self.assertTrue(overview.stations[0].neighbors)
