"""Controller-layer configuration."""
import os

NUM_TURNS = 20
SESSION_LOG_DIR = "logs"
SESSION_NAME = "minillama"
DEFAULT_MODEL_PARAM_KEY = "greedy"
SESSION_LOG_PROFILE = os.environ.get("MINILLAMA_SESSION_LOG_PROFILE", "runtime").lower()
GUI_ENABLED = os.environ.get("MINILLAMA_GUI_ENABLED", "true").lower() not in {"0", "false", "off", "no"}
GUI_MODE = os.environ.get("MINILLAMA_GUI_MODE", "conversation").lower()
ARTIFACT_DIR = os.environ.get("MINILLAMA_ARTIFACT_DIR", "artifacts")
NETWORK_PICTURE_DIR = os.environ.get(
    "MINILLAMA_NETWORK_PICTURE_DIR",
    os.path.join(ARTIFACT_DIR, "network_graphs"),
)
RESEARCH_LOG_DIR = os.environ.get("MINILLAMA_RESEARCH_LOG_DIR", os.path.join(ARTIFACT_DIR, "research_logs"))
