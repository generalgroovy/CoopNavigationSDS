"""Run CoopNavigationSDS with the startup configuration dialog enabled."""
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coop_navigation_sds.app import main


if __name__ == "__main__":
    main()
