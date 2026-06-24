"""Run an interactive experiment or a deterministic smoke experiment."""

import argparse

from coop_navigation_sds.app import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="python -m coop_navigation_sds")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run a fast deterministic pipeline without opening the configuration GUI",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="top-level result directory used by --smoke",
    )
    arguments = parser.parse_args()
    if arguments.smoke:
        from coop_navigation_sds.smoke import run_smoke

        result, paths = run_smoke(arguments.results_dir)
        print(f"Smoke result: {result.extra.get('conversation_outcome', 'unknown')}")
        print(f"Run folder: {paths['run_dir']}")
    else:
        main()
