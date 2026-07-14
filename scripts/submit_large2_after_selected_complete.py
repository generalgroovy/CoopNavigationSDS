"""Submit large2 / Mistral only after selected non-large2 coverage is complete.

This helper is intentionally separate from the normal remaining-coverage
submitter. It prevents accidental Mistral submission while small1, small2,
medium1, medium2, or large1 still have planned selected conditions missing.

The script reads results/experiment_coverage_conditions.csv, maps missing
large2 condition IDs back to their source job-file indices, and prints or
submits Slurm arrays. It never deletes results and never recomputes analysis.
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
from coop_navigation_sds.Configuration.travel import NETWORK_SEED  # noqa: E402
from coop_navigation_sds.ResultsAndArtifacts.coverage import _coverage_key  # noqa: E402
from scripts.submit_agent_b_model_jobs import condition_list  # noqa: E402


NON_LARGE2_SELECTED_JOB_FILES = (
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
    "jobs/agent_b_llm/transformers_speech_grid/small/01-tinyllama-1.1b.job",
    "jobs/agent_b_llm/transformers_speech_grid/small/02-qwen2.5-0.5b.job",
    "jobs/agent_b_llm/transformers_speech_grid/medium/01-qwen2.5-1.5b.job",
    "jobs/agent_b_llm/transformers_speech_grid/medium/02-phi3-mini.job",
    "jobs/agent_b_llm/transformers_speech_grid/large/01-qwen2.5-7b.job",
)


LARGE2_JOB_FILES = (
    "jobs/agent_b_llm/userlm_transformers_speech_grid/large/02-mistral-7b.job",
    "jobs/agent_b_llm/userlm_pressure_grid/large/02-mistral-7b.job",
    "jobs/agent_b_llm/transformers_speech_grid/large/02-mistral-7b.job",
)


@dataclass(frozen=True)
class Submission:
    job_file: Path
    model_label: str
    indexes: list[int]
    cpus: int
    mem: str
    time: str
    chunk_size: int


def read_coverage(coverage_file: Path) -> tuple[set[str], set[str]]:
    planned: set[str] = set()
    completed: set[str] = set()
    if not coverage_file.is_file():
        raise SystemExit(f"Coverage file does not exist: {coverage_file}")
    with coverage_file.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            coverage_key = str(row.get("coverage_key") or "")
            if not coverage_key:
                continue
            if str(row.get("planned") or "").strip().casefold() == "true":
                planned.add(coverage_key)
            try:
                completed_count = int(float(row.get("completed_count") or 0))
            except ValueError:
                completed_count = 0
            if completed_count > 0:
                completed.add(coverage_key)
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
    return cleaned[:limit].rstrip("-") or "large2"


def condition_coverage_key(job: dict, condition) -> str:
    config = job["config"]
    parameters = dict(condition.parameter_values)
    return _coverage_key({
        "condition_id": condition.condition_id,
        "experiment_platform": parameters.get("experiment_platform", "unspecified"),
        "matrix_family": parameters.get("matrix_family", job["name"]),
        "test_case_key": str(condition.test_case_key).split(":", 1)[0],
        "scenario_key": condition.scenario_key,
        "persona_key": condition.persona_key,
        "agent_a_type": config.get("agent_a_type", "staged"),
        "agent_a_audio_persona": condition.agent_a_audio_persona,
        "agent_b_model": condition.agent_b_model,
        "agent_b_llm_size": parameters.get("agent_b_llm_size"),
        "agent_b_model_slot": parameters.get("agent_b_model_slot"),
        "agent_b_model_role": parameters.get("agent_b_model_role"),
        "agent_b_audio_persona": condition.agent_b_audio_persona,
        "configured_tts_engine": condition.tts_engine,
        "configured_asr_engine": condition.asr_engine,
        "asr_search_width": parameters.get("asr_beam_size", config.get("asr_beam_size", "default")),
        "speech_pattern_key": condition.speech_pattern_key,
        "model_param_key": condition.model_param_key,
        "objective_mode": condition.objective_mode,
        "network_seed": parameters.get("network_seed", NETWORK_SEED),
        "iteration": condition.iteration,
        "run_type": condition.run_type,
    })


def condition_indexes(relative_job_file: str, planned: set[str], completed: set[str]) -> list[int]:
    path = ROOT / relative_job_file
    if not path.is_file():
        raise SystemExit(f"Missing job file: {path}")
    job = load_experiment_job(path)
    return [
        index for index, condition in enumerate(condition_list(job))
        if (key := condition_coverage_key(job, condition)) in planned and key not in completed
    ]


def verify_non_large2_complete(planned: set[str], completed: set[str]) -> list[tuple[str, int, str]]:
    missing = []
    for relative in NON_LARGE2_SELECTED_JOB_FILES:
        indexes = condition_indexes(relative, planned, completed)
        if indexes:
            missing.append((relative, len(indexes), compact_ranges(indexes)))
    return missing


def resource_for(relative_job_file: str, args) -> tuple[int, str, str, int]:
    if "userlm_" in relative_job_file or "/userlm_" in relative_job_file:
        return args.userlm_cpus, args.userlm_mem, args.userlm_time, args.userlm_chunk_size
    return args.tinyllama_cpus, args.tinyllama_mem, args.tinyllama_time, args.tinyllama_chunk_size


def find_large2_submissions(args) -> list[Submission]:
    planned, completed = read_coverage(Path(args.coverage_file))
    if not args.allow_before_non_large2_complete:
        missing_non_large2 = verify_non_large2_complete(planned, completed)
        if missing_non_large2:
            print("Non-large2 selected coverage is not complete. Large2 submission is blocked.", flush=True)
            for relative, count, indexes in missing_non_large2:
                print(f"missing_non_large2 | {relative} | count={count} | indices={indexes}", flush=True)
            raise SystemExit(3)

    submissions = []
    for relative in LARGE2_JOB_FILES:
        path = ROOT / relative
        job = load_experiment_job(path)
        config = job.get("config", {})
        indexes = condition_indexes(relative, planned, completed)
        cpus, mem, time_limit, chunk_size = resource_for(relative, args)
        submissions.append(Submission(
            path,
            str(config.get("model_profile") or config.get("model_name") or path.stem),
            indexes,
            cpus,
            mem,
            time_limit,
            chunk_size,
        ))
    return submissions


def submit_command(args, submission: Submission, index_chunk: list[int]) -> list[str]:
    array_spec = compact_ranges(index_chunk) + f"%{args.array_concurrency}"
    job_name = safe_name(f"large2-{submission.job_file.parent.parent.name}-{submission.job_file.stem}")
    export_values = {
        "PROJECT_ROOT": str(ROOT),
        "PYTHON_BIN": args.python_bin,
        "RESULTS_ROOT": args.results_dir,
        "JOB_FILE": str(submission.job_file),
        "START_OLLAMA": "0",
        "VALID_CONDITIONS_ONLY": "0",
        "CUDA_VISIBLE_DEVICES": "",
        "NVIDIA_VISIBLE_DEVICES": "void",
        "ACCELERATE_USE_CPU": "true",
        "PYTORCH_NVML_BASED_CUDA_CHECK": "0",
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
        f"--cpus-per-task={submission.cpus}",
        f"--mem={submission.mem}",
        f"--time={submission.time}",
        "--export=ALL," + ",".join(f"{key}={value}" for key, value in export_values.items()),
        str(ROOT / "slurm" / "agent_b_model_cpu_array.sbatch"),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage-file", default=str(ROOT / "results" / "experiment_coverage_conditions.csv"))
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    parser.add_argument("--python-bin", default=str(ROOT / ".venv-linux" / "bin" / "python"))
    parser.add_argument("--partition", default="standard")
    parser.add_argument("--array-concurrency", type=int, default=1)
    parser.add_argument("--tinyllama-cpus", type=int, default=6)
    parser.add_argument("--tinyllama-mem", default="72G")
    parser.add_argument("--tinyllama-time", default="07:59:00")
    parser.add_argument("--tinyllama-chunk-size", type=int, default=6)
    parser.add_argument("--userlm-cpus", type=int, default=8)
    parser.add_argument("--userlm-mem", default="112G")
    parser.add_argument("--userlm-time", default="09:59:00")
    parser.add_argument("--userlm-chunk-size", type=int, default=4)
    parser.add_argument(
        "--allow-before-non-large2-complete",
        action="store_true",
        help="Preview or submit large2 even when selected non-large2 conditions are missing.",
    )
    parser.add_argument("--submit", action="store_true", help="Actually call sbatch. Default only prints commands.")
    args = parser.parse_args(argv)

    submissions = find_large2_submissions(args)
    total_missing = sum(len(item.indexes) for item in submissions)
    print(f"large2_job_files={len(submissions)}", flush=True)
    print(f"large2_missing_condition_tasks={total_missing}", flush=True)
    for item in submissions:
        print(
            f"{item.model_label:30} {item.job_file.relative_to(ROOT)} "
            f"missing={len(item.indexes):3} indices={compact_ranges(item.indexes) or '-'} "
            f"resources=cpus:{item.cpus} mem:{item.mem} time:{item.time} chunk:{item.chunk_size}",
            flush=True,
        )
        for index_chunk in chunks(item.indexes, item.chunk_size):
            command = submit_command(args, item, index_chunk)
            print("  " + " ".join(command), flush=True)
            if args.submit:
                subprocess.run(command, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
