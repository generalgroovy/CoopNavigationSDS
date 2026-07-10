"""Submit one independent Slurm array per Agent B model job.

The submitter discovers registered Agent B ``.job`` files, computes each job's
condition count, and submits a separate Slurm array for each model. A failed,
delayed, or cancelled model therefore does not block other model treatments.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coop_navigation_sds.Configuration.jobs import (  # noqa: E402
    job_linked_profiles,
    job_parameter_grid,
    job_parameter_profiles,
    load_experiment_job,
)
from coop_navigation_sds.experiments import build_condition_grid  # noqa: E402
from coop_navigation_sds.NaturalLanguageGeneration.models import model_memory_requirement_gb  # noqa: E402


DEFAULT_ROOTS = {
    "tinyllama": (ROOT / "jobs" / "agent_b_llm" / "transformers_speech_grid",),
    "userlm": (
        ROOT / "jobs" / "agent_b_llm" / "userlm_speech_grid",
        ROOT / "jobs" / "agent_b_llm" / "userlm_transformers_speech_grid",
    ),
}

RESOURCE_TABLE = {
    ("tinyllama", "small"): (4, "18G", "01:29:00"),
    ("tinyllama", "medium"): (6, "32G", "02:29:00"),
    ("tinyllama", "large"): (8, "64G", "03:59:00"),
    ("userlm", "small"): (6, "44G", "03:59:00"),
    ("userlm", "medium"): (8, "56G", "03:59:00"),
    ("userlm", "large"): (8, "72G", "03:59:00"),
}
TIER_ORDER = {"small": 0, "medium": 1, "large": 2}


@dataclass(frozen=True)
class ModelJob:
    path: Path
    family: str
    tier: str
    name: str
    agent_a_type: str
    provider: str
    model_profile: str
    model_name: str
    condition_count: int

    @property
    def agent_b_memory_gb(self) -> float:
        return float(model_memory_requirement_gb(self.model_name, self.provider) or 0.0)

    @property
    def starts_ollama(self) -> bool:
        return self.provider == "ollama"

    @property
    def job_name(self) -> str:
        suffix = hashlib.sha1(str(self.path).encode("utf-8")).hexdigest()[:6]
        return safe_slurm_name(f"{self.agent_a_type}-{self.tier}-{self.path.stem}-{suffix}")


def safe_slurm_name(value: str, limit: int = 48) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in value)
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    if not cleaned:
        return "agent-b-model"
    parts = cleaned.split("-")
    suffix = parts[-1] if len(parts[-1]) == 6 else ""
    if suffix and len(cleaned) > limit:
        prefix = "-".join(parts[:-1])[: max(1, limit - len(suffix) - 1)].rstrip("-")
        return f"{prefix}-{suffix}"
    return cleaned[:limit]


def condition_count(job: dict) -> int:
    grid = job["grid"]
    return len(list(build_condition_grid(
        test_case_keys=grid.get("test_cases"),
        persona_keys=grid.get("personas"),
        speech_pattern_keys=grid.get("speech_patterns"),
        model_param_keys=grid.get("model_params"),
        objective_modes=grid.get("objective_modes"),
        agent_a_audio_persona_keys=grid.get("agent_a_audio_personas"),
        agent_b_audio_persona_keys=grid.get("agent_b_audio_personas"),
        tts_engine_keys=grid.get("tts_engines"),
        asr_engine_keys=grid.get("asr_engines"),
        agent_b_model_keys=grid.get("agent_b_models"),
        iterations=job["iterations"],
        parameter_grid=job_parameter_grid(job),
        parameter_profiles=job_parameter_profiles(job),
        linked_profiles=job_linked_profiles(job),
        coverage_strategy=job["coverage_strategy"],
        pair_audio_with_text=bool(job["config"].get("paired_audio_text_runs", False)),
    )))


def resolve_roots(args) -> list[tuple[str, Path]]:
    if args.root:
        return [
            (infer_family_from_root(Path(value).expanduser().resolve()), Path(value).expanduser().resolve())
            for value in args.root
        ]
    if args.family == "all":
        return [
            (family, path)
            for family, paths in DEFAULT_ROOTS.items()
            for path in paths
        ]
    return [(args.family, path) for path in DEFAULT_ROOTS[args.family]]


def infer_family_from_root(path: Path) -> str:
    """Infer the caller family from a custom job root for readable output and resources."""
    normalized = path.as_posix().casefold()
    if "userlm" in normalized:
        return "userlm"
    if "tinyllama" in normalized or "transformers_speech_grid" in normalized:
        return "tinyllama"
    return "custom"


def discover_jobs(args) -> list[ModelJob]:
    tiers = set(args.tier)
    providers = set(args.provider)
    profiles = {
        str(profile).strip()
        for profile in (getattr(args, "profile", None) or ())
        if str(profile).strip()
    }
    discovered: list[ModelJob] = []
    for family, root in resolve_roots(args):
        if not root.is_dir():
            continue
        for path in sorted(root.glob("*/*.job")):
            tier = path.parent.name.lower()
            if tier not in tiers:
                continue
            job = load_experiment_job(path)
            config = job["config"]
            provider = str(config.get("model_provider") or "").strip().lower()
            if "all" not in providers and provider not in providers:
                continue
            model_profile = str(config.get("model_profile") or "").strip()
            if profiles and model_profile not in profiles:
                continue
            discovered.append(ModelJob(
                path=path,
                family=family,
                tier=tier,
                name=job["name"],
                agent_a_type=str(config.get("agent_a_type") or family),
                provider=provider,
                model_profile=model_profile,
                model_name=str(config.get("model_name") or "").strip(),
                condition_count=condition_count(job),
            ))
    if not discovered:
        raise SystemExit("No Agent B model jobs matched the selected roots and tiers.")
    return sorted(discovered, key=lambda job: (
        job.family,
        TIER_ORDER.get(job.tier, 99),
        job.agent_b_memory_gb,
        job.path.name,
    ))


def resources_for(job: ModelJob, args) -> tuple[int, str, str]:
    cpus, memory, time_limit = RESOURCE_TABLE.get(
        (job.family, job.tier),
        RESOURCE_TABLE.get((job.agent_a_type, job.tier), (4, "24G", "01:59:00")),
    )
    if job.agent_a_type == "userlm":
        agent_a_memory = model_memory_requirement_gb("microsoft/UserLM-8b", "transformers") or 34.0
        agent_b_memory = model_memory_requirement_gb(job.model_name, job.provider) or 8.0
        speech_and_runtime_overhead = 10.0
        if job.tier == "large":
            speech_and_runtime_overhead += 4.0
        required = agent_a_memory + agent_b_memory + speech_and_runtime_overhead
        memory = f"{max(24, int(math.ceil(required / 4.0) * 4))}G"
        cpus = {
            "small": 6,
            "medium": 8,
            "large": 10 if job.provider == "transformers" else 8,
        }.get(job.tier, cpus)
        time_limit = {
            "small": "03:59:00",
            "medium": "03:59:00",
            "large": "03:59:00",
        }.get(job.tier, time_limit)
    return (
        int(args.cpus_per_task or cpus),
        args.memory or memory,
        args.time_limit or time_limit,
    )


def condition_chunks(condition_count: int, chunk_count: int) -> list[tuple[int, int, int, int]]:
    """Split a condition index range into bounded Slurm array chunks."""
    total = max(0, int(condition_count))
    chunks = max(1, int(chunk_count))
    if not total:
        return [(0, 0, 1, 1)]
    width = math.ceil(total / chunks)
    ranges = []
    start = 0
    while start < total:
        end = min(total - 1, start + width - 1)
        ranges.append((start, end, len(ranges) + 1, 0))
        start = end + 1
    total_ranges = len(ranges)
    return [
        (start, end, index, total_ranges)
        for start, end, index, _ in ranges
    ]


def condition_chunks_by_size(condition_count: int, max_conditions: int) -> list[tuple[int, int, int, int]]:
    """Split condition indices so every submitted array has at most max_conditions tasks."""
    total = max(0, int(condition_count))
    maximum = max(1, int(max_conditions))
    chunks = max(1, math.ceil(total / maximum))
    return condition_chunks(total, chunks)


def sbatch_command(
    job: ModelJob,
    args,
    *,
    array_start: int | None = None,
    array_end: int | None = None,
    chunk_index: int = 1,
    chunk_total: int = 1,
) -> list[str]:
    cpus, memory, time_limit = resources_for(job, args)
    array_first = 0 if array_start is None else int(array_start)
    array_last = max(array_first, max(0, job.condition_count - 1) if array_end is None else int(array_end))
    job_name = job.job_name
    if chunk_total > 1:
        job_name = safe_slurm_name(f"{job.job_name}-c{chunk_index:02d}of{chunk_total:02d}")
    export_values = {
        "PROJECT_ROOT": str(ROOT),
        "PYTHON_BIN": args.python_bin,
        "RESULTS_ROOT": str(Path(args.results_dir).expanduser()),
        "JOB_FILE": str(job.path),
        "START_OLLAMA": "1" if job.starts_ollama else "0",
    }
    export_arg = "ALL," + ",".join(f"{key}={value}" for key, value in export_values.items())
    return [
        "sbatch",
        f"--array={array_first}-{array_last}%{args.array_concurrency}",
        f"--job-name={job_name}",
        f"--cpus-per-task={cpus}",
        f"--mem={memory}",
        f"--time={time_limit}",
        f"--export={export_arg}",
        str(ROOT / "slurm" / "agent_b_model_cpu_array.sbatch"),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("all", "tinyllama", "userlm"), default="all")
    parser.add_argument("--root", action="append", help="Custom root containing tier subfolders with .job files.")
    parser.add_argument("--tier", nargs="+", choices=("small", "medium", "large"), default=("small", "medium", "large"))
    parser.add_argument("--provider", nargs="+", choices=("all", "transformers", "ollama", "openai_compatible", "llama_cpp"), default=("all",))
    parser.add_argument("--profile", action="append", help="Restrict submission to one registered model_profile. Repeat for multiple profiles.")
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    parser.add_argument("--python-bin", default=str(ROOT / ".venv-linux" / "bin" / "python"))
    parser.add_argument("--array-concurrency", type=int, default=1)
    parser.add_argument(
        "--array-chunks",
        type=int,
        default=1,
        help=(
            "Split each model's condition range into this many independent Slurm "
            "arrays. Each chunk keeps the same resource and time-limit request."
        ),
    )
    parser.add_argument(
        "--max-conditions-per-array",
        type=int,
        default=0,
        help=(
            "Prefer this operational limit over --array-chunks. For example, "
            "14 turns an 84-condition model grid into six 14-task arrays."
        ),
    )
    parser.add_argument("--cpus-per-task", type=int, default=0, help="Override tier CPU default.")
    parser.add_argument("--memory", default="", help="Override tier memory default, for example 48G.")
    parser.add_argument("--time-limit", default="", help="Override tier time default, for example 03:59:00.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.array_concurrency < 1:
        raise SystemExit("--array-concurrency must be at least 1.")
    if args.array_chunks < 1:
        raise SystemExit("--array-chunks must be at least 1.")
    if args.max_conditions_per_array < 0:
        raise SystemExit("--max-conditions-per-array must be zero or positive.")
    if not args.dry_run and shutil.which("sbatch") is None:
        raise SystemExit("sbatch is not available. Use --dry-run locally or run on a Slurm login node.")

    jobs = discover_jobs(args)
    print(f"Agent B model jobs: {len(jobs)}")
    submitted = 0
    failed: list[tuple[ModelJob, int, str]] = []
    for index, job in enumerate(jobs, start=1):
        cpus, memory, time_limit = resources_for(job, args)
        chunks = (
            condition_chunks_by_size(job.condition_count, args.max_conditions_per_array)
            if args.max_conditions_per_array
            else condition_chunks(job.condition_count, args.array_chunks)
        )
        print(
            f"{index:02d}. {job.family}/{job.tier} | Agent A={job.agent_a_type} | "
            f"Agent B={job.model_name} | profile={job.model_profile} | provider={job.provider} | "
            f"model_mem={job.agent_b_memory_gb:g}G | conditions={job.condition_count} | "
            f"chunks={len(chunks)} | max_array_conditions="
            f"{args.max_conditions_per_array or 'auto'} | cpus={cpus} mem={memory} time={time_limit}"
        )
        for array_start, array_end, chunk_index, chunk_total in chunks:
            command = sbatch_command(
                job,
                args,
                array_start=array_start,
                array_end=array_end,
                chunk_index=chunk_index,
                chunk_total=chunk_total,
            )
            print("    " + " ".join(command))
            if args.dry_run:
                continue
            result = subprocess.run(
                command,
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            output = "\n".join(
                part.strip()
                for part in (result.stdout, result.stderr)
                if part and part.strip()
            )
            if result.returncode:
                failed.append((job, result.returncode, output))
                print(f"    SUBMIT FAILED rc={result.returncode}")
                if output:
                    print("    " + output.replace("\n", "\n    "))
                continue
            submitted += 1
            if output:
                print("    " + output.replace("\n", "\n    "))
    if not args.dry_run:
        print(f"Submission summary: submitted={submitted} failed={len(failed)}")
        for job, return_code, output in failed:
            print(
                f"FAILED {job.family}/{job.tier} {job.model_name} "
                f"rc={return_code}: {output or 'no Slurm message captured'}"
            )
        return 1 if failed else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
