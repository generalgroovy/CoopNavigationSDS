"""Submit only uncovered selected thesis conditions.

The script reads ``results/experiment_coverage_conditions.csv`` and maps
unfinished condition IDs back to each active Agent B ``.job`` file. It then
prints or submits Slurm arrays for the exact missing condition IDs. The index
passed to Slurm is the full-grid index because the coverage CSV is already the
source of truth for which designs are valid thesis conditions. Raw results are
never deleted or modified. The file name is kept for compatibility with older
cluster notes; the active selection is defined by ``SELECTED_JOB_FILES``.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coop_navigation_sds.Configuration.jobs import load_experiment_job  # noqa: E402
from scripts.submit_agent_b_model_jobs import (  # noqa: E402
    condition_list,
)


SELECTED_JOB_FILES = (
    "jobs/agent_b_llm/userlm_transformers_speech_grid/small/01-tinyllama-1.1b.job",
    "jobs/agent_b_llm/userlm_transformers_speech_grid/small/02-qwen2.5-0.5b.job",
    "jobs/agent_b_llm/userlm_transformers_speech_grid/medium/01-qwen2.5-1.5b.job",
    "jobs/agent_b_llm/userlm_transformers_speech_grid/medium/02-phi3-mini.job",
    "jobs/agent_b_llm/userlm_transformers_speech_grid/large/01-qwen2.5-7b.job",
    "jobs/agent_b_llm/userlm_pressure_grid/small/01-tinyllama-1.1b.job",
    "jobs/agent_b_llm/userlm_pressure_grid/small/02-qwen2.5-0.5b.job",
    "jobs/agent_b_llm/userlm_pressure_grid/medium/01-qwen2.5-1.5b.job",
    "jobs/agent_b_llm/userlm_pressure_grid/medium/02-phi3-mini.job",
    "jobs/agent_b_llm/userlm_pressure_grid/large/01-qwen2.5-7b.job",
)

RESOURCE_BY_TIER = {
    "small": {"cpus": 4, "mem": "36G", "time": "03:59:00"},
    "medium": {"cpus": 6, "mem": "48G", "time": "04:59:00"},
    "large": {"cpus": 6, "mem": "60G", "time": "05:59:00"},
}


@dataclass(frozen=True)
class Submission:
    job_file: Path
    tier: str
    model_label: str
    indexes: list[int]


def read_coverage(coverage_file: Path) -> tuple[set[str], set[str]]:
    planned = set()
    completed = set()
    if not coverage_file.is_file():
        raise SystemExit(f"Coverage file does not exist: {coverage_file}")
    with coverage_file.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            condition_id = str(row.get("condition_id") or "")
            if not condition_id:
                continue
            if str(row.get("planned") or "").strip().casefold() == "true":
                planned.add(condition_id)
            try:
                count = int(float(row.get("completed_count") or 0))
            except ValueError:
                count = 0
            if count > 0:
                completed.add(condition_id)
    return planned, completed


def compact_ranges(indexes: list[int]) -> str:
    if not indexes:
        return ""
    parts = []
    start = previous = indexes[0]
    for value in indexes[1:]:
        if value == previous + 1:
            previous = value
            continue
        parts.append(f"{start}-{previous}" if start != previous else str(start))
        start = previous = value
    parts.append(f"{start}-{previous}" if start != previous else str(start))
    return ",".join(parts)


def chunks(values: list[int], width: int) -> list[list[int]]:
    width = max(1, int(width))
    return [values[index:index + width] for index in range(0, len(values), width)]


def safe_name(value: str, limit: int = 42) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in value)
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    return cleaned[:limit].rstrip("-") or "missing"


def find_missing(coverage_file: Path) -> list[Submission]:
    planned, completed = read_coverage(coverage_file)
    submissions = []
    for relative in SELECTED_JOB_FILES:
        path = ROOT / relative
        if not path.is_file():
            raise SystemExit(f"Missing job file: {path}")
        job = load_experiment_job(path)
        config = job.get("config", {})
        if str(config.get("model_profile", "")).casefold().startswith("mistral"):
            continue
        conditions = condition_list(job)
        missing_indexes = [
            index for index, condition in enumerate(conditions)
            if condition.condition_id in planned and condition.condition_id not in completed
        ]
        tier = path.parent.name
        model_label = str(config.get("model_profile") or config.get("model_name") or path.stem)
        submissions.append(Submission(path, tier, model_label, missing_indexes))
    return submissions


def submit_command(args, submission: Submission, index_chunk: list[int]) -> list[str]:
    resource = RESOURCE_BY_TIER[submission.tier]
    array_spec = compact_ranges(index_chunk) + f"%{args.array_concurrency}"
    job_name = safe_name(f"rem-{submission.tier}-{submission.model_label}")
    export_values = {
        "PROJECT_ROOT": str(ROOT),
        "PYTHON_BIN": args.python_bin,
        "RESULTS_ROOT": args.results_dir,
        "JOB_FILE": str(submission.job_file),
        "START_OLLAMA": "0",
        "VALID_CONDITIONS_ONLY": "0",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "ONNXRUNTIME_THREAD_POOL_SIZE": "1",
        "ORT_LOG_SEVERITY_LEVEL": "3",
    }
    return [
        "sbatch",
        "-p", args.partition,
        f"--array={array_spec}",
        f"--job-name={job_name}",
        f"--cpus-per-task={resource['cpus']}",
        f"--mem={resource['mem']}",
        f"--time={resource['time']}",
        "--export=ALL," + ",".join(f"{key}={value}" for key, value in export_values.items()),
        str(ROOT / "slurm" / "agent_b_model_cpu_array.sbatch"),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage-file", default=str(ROOT / "results" / "experiment_coverage_conditions.csv"))
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    parser.add_argument("--python-bin", default=str(ROOT / ".venv-linux" / "bin" / "python"))
    parser.add_argument("--partition", default="standard")
    parser.add_argument("--max-conditions-per-array", type=int, default=18)
    parser.add_argument("--array-concurrency", type=int, default=1)
    parser.add_argument("--submit", action="store_true", help="Actually call sbatch. Default only prints commands.")
    args = parser.parse_args(argv)

    submissions = find_missing(Path(args.coverage_file))
    total_missing = sum(len(item.indexes) for item in submissions)
    print(f"selected_active_models={len(submissions)}", flush=True)
    print(f"missing_condition_tasks={total_missing}", flush=True)
    for item in submissions:
        print(
            f"{item.tier:6} {item.model_label:34} missing={len(item.indexes):3} "
            f"indices={compact_ranges(item.indexes) or '-'}",
            flush=True,
        )
        for index_chunk in chunks(item.indexes, args.max_conditions_per_array):
            command = submit_command(args, item, index_chunk)
            print("  " + " ".join(command), flush=True)
            if args.submit:
                subprocess.run(command, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
