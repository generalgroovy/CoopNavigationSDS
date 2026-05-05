import os
import torch

# ---------------- MODEL ----------------

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TOKEN = os.environ.get("HF_TOKEN")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- GENERATION SPEED / LENGTH ----------------

NUM_TURNS = 20

# Enough for full natural sentences, but still manageable on CPU.
MAX_NEW_TOKENS = 48

MAX_INPUT_TOKENS = 9999
HISTORY_MESSAGES = 10

# False = Agent A uses fast persona templates.
# True = Agent A also uses the LLM, but runtime roughly doubles.
LLM_AGENT_A = True

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

# Default: every station pair should be reachable.
# Set False only when explicitly testing disconnected networks.
FORCE_CONNECTED_NETWORK = True

# ---------------- SCENARIO DEFAULTS ----------------

TRANSFER_TIME_MIN = 2
START_TIME_MIN = 8 * 60 + 7

# ---------------- GUI ----------------

GUI_WIDTH = 1220
GUI_HEIGHT = 720
GUI_DIALOG_MIN_WIDTH = 390
GUI_MAP_MIN_WIDTH = 680
