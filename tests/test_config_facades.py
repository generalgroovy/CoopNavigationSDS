import unittest

from minillama.agent_b.config import AGENT_B_PLUGIN, SPEECH_PATTERNS
from minillama.controller.config import NUM_TURNS, SESSION_LOG_DIR
from minillama.model.config import DEVICE, MODEL
from minillama.view.config import GUI_COLORS, GUI_WIDTH, MAP_ROUTE_LINE_WIDTH


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
