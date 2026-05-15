"""Controller-layer configuration."""
import os

NUM_TURNS = 20
SESSION_LOG_DIR = "logs"
SESSION_NAME = "minillama"
DEFAULT_MODEL_PARAM_KEY = "greedy"
SESSION_LOG_PROFILE = os.environ.get("MINILLAMA_SESSION_LOG_PROFILE", "off").lower()
