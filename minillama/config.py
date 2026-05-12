"""Central editable configuration for model runtime, experiment defaults, transit network generation, scoring, speech simulation, GUI layout, and map rendering.
"""
import os
import torch

# ---------------- MODEL ----------------

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TOKEN = os.environ.get("HF_TOKEN")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# MODEL_PROVIDER:
# - "transformers": local Hugging Face causal model
# - "openai": OpenAI/ChatGPT-compatible chat completions API
MODEL_PROVIDER = os.environ.get("MINILLAMA_MODEL_PROVIDER", "transformers").lower()
CHAT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
CHAT_API_KEY = os.environ.get("OPENAI_API_KEY")
CHAT_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
CHAT_TIMEOUT_SEC = float(os.environ.get("MINILLAMA_CHAT_TIMEOUT_SEC", "60"))

# ---------------- GENERATION SPEED / LENGTH ----------------

NUM_TURNS = 20

# Enough for full natural sentences, but still manageable on CPU.
MAX_NEW_TOKENS = 99
GENERATION_MAX_TIME_SEC = 25

MAX_INPUT_TOKENS = 9999
HISTORY_MESSAGES = 10

# False = Agent A uses fast persona templates.
# True = Agent A also uses the LLM, but runtime roughly doubles.
LLM_AGENT_A = True

# Agent B plugin:
# - "llm": default LLM-backed VerbalTransformationPipeline
# - "simple": deterministic planner-backed Agent B
# - "package.module:factory_or_class": custom plugin import path
AGENT_B_PLUGIN = os.environ.get("MINILLAMA_AGENT_B_PLUGIN", "llm")

# ---------------- NETWORK GENERATION ----------------

NUM_STATIONS = 20

# Keep at least 5 for a useful layout:
# ring + east-west + south-north + diagonal + additional connector.
NUM_LINES = 6

# Optional. Use None for automatic rectangular grid.
# Example for 20 stations: 5 columns x 4 rows.
LAYOUT_COLUMNS = None

DEFAULT_HEADWAY_MIN = 5
DEFAULT_TRAVEL_TIME_MIN = 4

MIN_TRAVEL_TIME_MIN = 2
MAX_TRAVEL_TIME_MIN = 6

NETWORK_SEED = 42
LINE_FULLNESS_SEED = 84
MIN_LINE_FULLNESS_PERCENT = 15
MAX_LINE_FULLNESS_PERCENT = 95

# Default: every station pair should be reachable.
# Set False only when explicitly testing disconnected networks.
FORCE_CONNECTED_NETWORK = True

# Line colors and generated schematic station placement.
LINE_COLORS = [
    "#d13f31",
    "#3a8f3a",
    "#4169e1",
    "#f2c230",
    "#9b59b6",
    "#111111",
    "#e67e22",
    "#16a085",
    "#2c3e50",
    "#c0392b",
]
STATION_X_GAP = 120
STATION_Y_GAP = 90
STATION_X_OFFSET = 80
STATION_Y_OFFSET = 70
STATION_ROW_STAGGER = 25
MIN_RING_STATIONS = 4
MIN_LINE_STATIONS = 2
HEADWAY_VARIATION_MOD = 3
LINE_COVERAGE_SCORE_WEIGHT = 10
LINE_KIND_BONUS = 2
LINE_LENGTH_SCORE_DIVISOR = 10

# Optional per-line stop overrides for thesis scenarios.
LINE_STOP_OVERRIDES = {
    "East-West-4": ["Charlie", "Hotel", "Mike", "Romeo"],
}

# ---------------- ROUTE INTERPRETATION / METRICS ----------------

ROUTE_INTERPRETER_SCORING = {
    "marker_bonus": 30,
    "fragment_index_weight": 10,
    "route_length_weight": 1,
    "starts_correctly_bonus": 100,
    "reaches_goal_bonus": 200,
    "arrival_penalty_divisor": 10000,
}

METRIC_QUALITY_BASE_WEIGHT = 0.70
METRIC_QUALITY_DURATION_WEIGHT = 0.30

# ---------------- SPEECH SIMULATION ----------------

SPEECH_PATTERNS = {
    "hesitation_probability": 0.45,
    "hesitation_tokens": ("um", "let me see", "okay"),
    "noisy_station_drop_probability": 0.08,
}

# ---------------- SCENARIO DEFAULTS ----------------

TRANSFER_TIME_MIN = 1
START_TIME_MIN = 8 * 60 + 7

# ---------------- GUI ----------------

GUI_WIDTH = 1220
GUI_HEIGHT = 720
GUI_DIALOG_MIN_WIDTH = 390
GUI_MAP_MIN_WIDTH = 680
GUI_MIN_WIDTH = 980
GUI_MIN_HEIGHT = 600
GUI_EQUAL_PANE_MIN_WIDTH = 560
GUI_REFRESH_MS = 100

# ---------------- GUI THEME / LAYOUT ----------------

GUI_FONT_FAMILY = "Segoe UI"
GUI_MONO_FONT_FAMILY = "Consolas"
GUI_FONT_SMALL = 10
GUI_FONT_NORMAL = 11
GUI_FONT_SECTION = 13
GUI_TABLE_FONT_SIZE = 10
GUI_TABLE_HEADER_FONT_SIZE = 10
GUI_TABLE_ROW_HEIGHT = 20
GUI_SECTION_CORNER_RADIUS = 8
GUI_SECTION_BORDER_WIDTH = 1
GUI_MAIN_PAD = 2
GUI_SECTION_PAD_Y = 2
GUI_SECTION_HEADER_PAD_X = 4
GUI_SECTION_HEADER_PAD_Y = (2, 0)
GUI_SECTION_CONTENT_PAD_Y = (0, 1)
GUI_TOGGLE_WIDTH = 76
GUI_SELECTOR_HEIGHT = 26
GUI_LEGACY_SELECTOR_WIDTH = 52
GUI_LEGACY_SELECTOR_HEIGHT = 108
GUI_LEGACY_SELECTOR_BUTTON_WIDTH = 38
GUI_LEGACY_SELECTOR_BUTTON_HEIGHT = 24
GUI_LEGACY_SELECTOR_BUTTON_RADIUS = 6
GUI_TEXTBOX_HEIGHT = 8
GUI_STATION_TABS_HEIGHT = 96
GUI_DEFAULT_TABS_HEIGHT = 96

GUI_COLORS = {
    "app_bg": "#edf2f7",
    "panel_bg": "#ffffff",
    "table_bg": "#ffffff",
    "table_heading_bg": "#edf2f7",
    "table_selected": "#dbeafe",
    "table_border": "#d1d5db",
    "text": "#111827",
    "muted_text": "#4b5563",
    "subtle_text": "#6b7280",
    "agent_a": "#174ea6",
    "agent_b": "#137333",
    "warning": "#b3261e",
    "map_bg": "#fbfaf7",
    "map_border": "#d0d5dd",
    "map_outline": "#202124",
    "start_station": "#cfe8ff",
    "destination_station": "#ffe2a8",
    "route_station": "#ffd7d2",
    "route_line": "#ff1f1f",
    "route_label_bg": "#fff1f0",
    "tab_bg": "#f9fafb",
    "tab_button_bg": "#e5e7eb",
    "tab_selected": "#2563eb",
    "tab_selected_hover": "#1d4ed8",
    "tab_unselected_hover": "#d1d5db",
}

GUI_ROUTE_TABLE_COLUMNS = [
    ("from", "From", 42, "center", False),
    ("to", "To", 42, "center", False),
    ("line", "Line", 50, "center", False),
    ("fullness", "Full", 46, "center", False),
    ("depart", "Leave", 58, "center", False),
    ("arrive", "Arrive", 58, "center", False),
    ("ride", "Ride", 46, "center", False),
    ("wait", "Wait", 46, "center", False),
    ("transfer", "Change", 58, "center", False),
]
GUI_LINE_TABLE_COLUMNS = [
    ("order", "#", 34, "center", False),
    ("station", "Stop", 48, "center", False),
    ("fullness", "Full", 46, "center", False),
    ("previous", "From", 48, "center", False),
    ("next", "Toward", 62, "center", False),
    ("ride", "Ride", 82, "center", False),
    ("elapsed", "From start", 98, "w", False),
]
GUI_STATION_LINE_TABLE_COLUMNS = [
    ("line", "Line", 118, "w", False),
    ("fullness", "Full", 46, "center", False),
    ("route", "Route", 96, "w", False),
    ("stop", "Stop", 54, "center", False),
    ("neighbors", "Neighbors", 120, "w", False),
    ("travel", "Ride times", 140, "w", True),
]
GUI_STATION_TIME_TABLE_COLUMNS = [
    ("line", "Line", 118, "w", False),
    ("movement", "Movement", 140, "w", False),
    ("times", "Arrival = departure", 250, "w", True),
]
GUI_ROUTE_TABLE_HEIGHT = 3
GUI_LINE_TABLE_HEIGHT = 5
GUI_STATION_TABLE_HEIGHT = 3

# ---------------- GRAPH DISPLAY ----------------

MAP_MIN_WIDTH = 420
MAP_MIN_HEIGHT = 300
MAP_PADDING_X = 34
MAP_PADDING_Y = 28
MAP_MIN_SCALE = 0.3
MAP_PARALLEL_LINE_SPACING = 15
MAP_PARALLEL_LINE_MIN_SPACING = 12
MAP_LINE_CASING_WIDTH = 11
MAP_LINE_CASING_MIN_WIDTH = 8
MAP_LINE_WIDTH = 6
MAP_LINE_MIN_WIDTH = 4
MAP_ROUTE_LINE_WIDTH = 13
MAP_ROUTE_LINE_MIN_WIDTH = 11
MAP_START_DEST_RADIUS = 10
MAP_ROUTE_RADIUS = 9
MAP_STATION_RADIUS = 7
MAP_MIN_STATION_RADIUS = 5
MAP_STATION_FONT_SCALE = 9
MAP_STATION_MIN_FONT = 7
MAP_ROLE_FONT_SCALE = 7
MAP_ROLE_MIN_FONT = 6
MAP_EDGE_LABEL_MIN_SCALE = 0.65
MAP_EDGE_LABEL_OFFSET = 11
MAP_EDGE_LABEL_MIN_OFFSET = 9
MAP_EDGE_LABEL_FONT_SCALE = 7
MAP_ROUTE_LABEL_FONT_SCALE = 8
MAP_EDGE_LABEL_MIN_FONT = 6
MAP_LABEL_PAD = 2
MAP_LEGEND_RIGHT_OFFSET = 88
MAP_LEGEND_TOP = 18
MAP_LEGEND_ROW_GAP = 20
