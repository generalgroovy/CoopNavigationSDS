"""Run MiniLlama with the startup configuration dialog enabled."""
from pathlib import Path
import os
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MINILLAMA_GUI_ENABLED", "true")

from minillama.controller.main import main


if __name__ == "__main__":
    main()
