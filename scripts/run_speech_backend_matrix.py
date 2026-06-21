"""Run every registered text-to-speech and recognition combination."""
from argparse import ArgumentParser
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coop_navigation_sds.app import default_run_config
from coop_navigation_sds.Configuration.runtime import RESULTS_DIR
from coop_navigation_sds.Configuration.settings import default_settings_path, load_run_settings
from coop_navigation_sds.ResultsAndArtifacts.artifacts import create_execution_run_dir
from coop_navigation_sds.ResultsAndArtifacts.speech_matrix import run_speech_backend_matrix


def main():
    parser = ArgumentParser()
    parser.add_argument("--settings-file", default=str(default_settings_path()))
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()

    config = load_run_settings(default_run_config(), args.settings_file)
    run_dir = create_execution_run_dir(args.results_dir, "speech_backend_matrix")
    protocol, paths = run_speech_backend_matrix(
        run_dir,
        base_config=config,
        run_live=args.live,
    )
    summary = protocol["summary"]
    print(
        "Speech backend matrix: "
        f"{summary['contract_passed']}/{summary['combination_count']} contract combinations passed; "
        f"{summary['live_ready_combinations']} live-ready."
    )
    if args.live:
        print(
            f"Live results: {summary['live_passed']} passed, "
            f"{summary['live_failed']} failed, {summary['live_skipped']} skipped."
        )
    print(f"Protocol JSON: {paths['json']}")
    print(f"Case CSV: {paths['csv']}")
    print(f"Report: {paths['markdown']}")


if __name__ == "__main__":
    main()
