"""Rebuild the results-root experiment coverage registry."""
import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coop_navigation_sds.ResultsAndArtifacts.coverage import update_experiment_coverage  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Rebuild planned/completed experiment coverage files.")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    for name, path in update_experiment_coverage(args.results_dir).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
