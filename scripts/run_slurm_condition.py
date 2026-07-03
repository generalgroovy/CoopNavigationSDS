"""Preview or execute one condition from a scheduler-neutral experiment grid."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coop_navigation_sds.Configuration.slurm_grid import (  # noqa: E402
    SlurmConditionGrid,
    export_condition_json,
    reserve_condition_directory,
)


def condition_index(explicit_index=None, environment=None):
    """Resolve an explicit local index or Slurm's array-task index."""
    if explicit_index is not None:
        return int(explicit_index)
    value = (environment or os.environ).get("SLURM_ARRAY_TASK_ID")
    if value is None:
        raise ValueError("Pass --index locally or set SLURM_ARRAY_TASK_ID.")
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError("SLURM_ARRAY_TASK_ID must be an integer.") from exc


def preview(grid, limit=20, count_only=False):
    print(f"Grid: {grid.name}")
    print(f"Condition count: {len(grid.conditions)}")
    print(f"Valid array range: 0-{len(grid.conditions) - 1}")
    if count_only:
        return
    for condition in grid.conditions[:max(0, int(limit))]:
        row = condition.as_dict()
        print(json.dumps({
            "index": condition.index,
            "condition_id": condition.condition_id,
            "agent_b": condition.backend.key,
            "persona": condition.persona_key,
            "test_case": condition.test_case_key,
            "run_mode": condition.run_mode,
            "speech_pattern": row.get("speech_pattern_key"),
            "seed": condition.seed,
            "repetition": condition.repetition,
        }, sort_keys=True))


def execute(grid, index, results_dir, model_device=None):
    condition = grid.condition(index)
    overrides = {"model_device": model_device} if model_device else {}
    task_dir = reserve_condition_directory(results_dir, condition, overrides)
    condition_path = export_condition_json(
        condition,
        task_dir / "condition.json",
        runtime_overrides=overrides,
    )
    command = [
        sys.executable,
        "-m", "minillama.orchestration.run_experiments",
        "--condition-file", str(condition_path),
        "--results-dir", str(task_dir),
    ]
    print(f"Condition {condition.index}/{len(grid.conditions) - 1}: {condition.condition_id}")
    print(f"Task directory: {task_dir}")
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)
    return task_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preview_parser = subparsers.add_parser("preview", help="Print grid size and selected rows.")
    preview_parser.add_argument("--grid", required=True)
    preview_parser.add_argument("--limit", type=int, default=20)
    preview_parser.add_argument("--count-only", action="store_true")
    run_parser = subparsers.add_parser("run", help="Execute one local or Slurm array condition.")
    run_parser.add_argument("--grid", required=True)
    run_parser.add_argument("--index", type=int)
    run_parser.add_argument("--results-dir", default="results/slurm")
    run_parser.add_argument("--model-device", choices=("cpu", "cuda", "auto"))
    args = parser.parse_args(argv)
    grid = SlurmConditionGrid.from_file(args.grid)
    if args.command == "preview":
        preview(grid, args.limit, args.count_only)
        return
    try:
        index = condition_index(args.index)
        execute(grid, index, args.results_dir, args.model_device)
    except (IndexError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
