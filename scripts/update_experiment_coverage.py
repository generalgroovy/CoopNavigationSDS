"""Rebuild the results-root experiment coverage registry."""
import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coop_navigation_sds.ResultsAndArtifacts.coverage import update_experiment_coverage  # noqa: E402
from coop_navigation_sds.ResultsAndArtifacts.artifacts import compact_existing_result_tree  # noqa: E402
from coop_navigation_sds.ResultsAndArtifacts.comparison import (  # noqa: E402
    compare_runs,
    discover_evidence_run_directories,
    discover_run_directories,
    write_evidence_comparison,
)


def main():
    parser = argparse.ArgumentParser(description="Rebuild planned/completed experiment coverage files.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--compact-results",
        action="store_true",
        help="Consolidate finalized artifacts and recover readable combined files for interrupted runs.",
    )
    args = parser.parse_args()
    if args.compact_results:
        for report in compact_existing_result_tree(args.results_dir):
            print(f"compaction: {report}")
    for name, path in update_experiment_coverage(args.results_dir).items():
        print(f"{name}: {path}")
    if discover_run_directories([args.results_dir]):
        for name, path in compare_runs(
            [args.results_dir],
            Path(args.results_dir).resolve() / "comparison",
        ).items():
            print(f"comparison_{name}: {path}")
    if discover_evidence_run_directories([args.results_dir]):
        for name, path in write_evidence_comparison(
            [args.results_dir],
            Path(args.results_dir).resolve() / "general",
        ).items():
            print(f"evidence_{name}: {path}")


if __name__ == "__main__":
    main()
