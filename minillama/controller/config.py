"""Controller-layer configuration."""
import os

NUM_TURNS = int(os.environ.get("MINILLAMA_NUM_TURNS", "7"))
INVALID_ROUTE_LIMIT = int(os.environ.get("MINILLAMA_INVALID_ROUTE_LIMIT", "2"))
CONSTRAINT_MISS_LIMIT = int(os.environ.get("MINILLAMA_CONSTRAINT_MISS_LIMIT", "2"))
AGENT_A_TRANSFER_TOLERANCE = int(os.environ.get("MINILLAMA_AGENT_A_TRANSFER_TOLERANCE", "1"))
METRIC_SNAPSHOT_INTERVAL = int(os.environ.get("MINILLAMA_METRIC_SNAPSHOT_INTERVAL", "1"))
RESULTS_DIR = os.environ.get("MINILLAMA_RESULTS_DIR", os.environ.get("MINILLAMA_ARTIFACT_DIR", "results"))
SESSION_LOG_DIR = os.environ.get("MINILLAMA_SESSION_LOG_DIR", "session_logs")
SESSION_NAME = "minillama"
DEFAULT_MODEL_PARAM_KEY = "greedy"
SESSION_LOG_PROFILE = os.environ.get("MINILLAMA_SESSION_LOG_PROFILE", "runtime").lower()
GUI_ENABLED = os.environ.get("MINILLAMA_GUI_ENABLED", "false").lower() not in {"0", "false", "off", "no"}
GUI_MODE = os.environ.get("MINILLAMA_GUI_MODE", "conversation").lower()
NETWORK_DATA_CARD_ENABLED = os.environ.get("MINILLAMA_NETWORK_DATA_CARD_ENABLED", "false").lower() in {"1", "true", "on", "yes"}
ARTIFACT_DIR = RESULTS_DIR
NETWORK_PICTURE_DIR = os.environ.get(
    "MINILLAMA_NETWORK_PICTURE_DIR",
    os.path.join(RESULTS_DIR, "network_graphs"),
)
RESEARCH_LOG_DIR = os.environ.get("MINILLAMA_RESEARCH_LOG_DIR", RESULTS_DIR)
PROTOCOL_LOG_DIR = os.environ.get("MINILLAMA_PROTOCOL_LOG_DIR", RESULTS_DIR)
