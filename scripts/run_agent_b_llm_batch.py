"""Preview or sequentially run focused Agent B comparison manifests."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
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
from coop_navigation_sds.Configuration.schema import resolve_result_group, resolve_results_root  # noqa: E402
from coop_navigation_sds.experiments import build_condition_grid  # noqa: E402
from coop_navigation_sds.ResultsAndArtifacts.comparison import compare_runs  # noqa: E402


def load_batch_manifest(path, seen=None):
    """Resolve nested batch manifests into an ordered, duplicate-free job list."""
    path = Path(path).expanduser().resolve()
    seen = set(seen or ())
    if path in seen:
        raise ValueError(f"Cyclic batch manifest include: {path}")
    seen.add(path)
    document = json.loads(path.read_text(encoding="utf-8"))
    if int(document.get("schema_version", 1)) != 1:
        raise ValueError(f"Unsupported batch manifest schema: {path}")
    jobs = []
    for include in document.get("includes", ()):
        jobs.extend(load_batch_manifest(path.parent / include, seen)["jobs"])
    for job in document.get("jobs", ()):
        job_path = (path.parent / job).resolve()
        if not job_path.is_file():
            raise FileNotFoundError(job_path)
        jobs.append(job_path)
    jobs = list(dict.fromkeys(jobs))
    if not jobs:
        raise ValueError(f"Batch manifest contains no jobs: {path}")
    return {"name": str(document.get("name") or path.stem), "source": path, "jobs": jobs}


def job_condition_count(path):
    job = load_experiment_job(path)
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


def job_overview(path, results_root="results"):
    job = load_experiment_job(path)
    config = job["config"]
    parameters = job.get("parameter_values") or {}
    return {
        "job": job["name"],
        "path": str(Path(path).resolve()),
        "agent_a": config.get("agent_a_type"),
        "agent_a_model": (
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            if config.get("agent_a_type") == "tinyllama"
            else config.get("agent_a_model_name")
        ),
        "agent_b_model": config.get("model_name"),
        "agent_b_size": (parameters.get("agent_b_llm_size") or [None])[0],
        "model_role": (parameters.get("agent_b_model_role") or [None])[0],
        "conditions": job_condition_count(path),
        "result_group": config.get("result_group"),
        "result_path": resolve_result_group(results_root, config.get("result_group")),
    }


def _write_table(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_manifest(manifest_path, results_root="results", continue_on_error=False):
    manifest = load_batch_manifest(manifest_path)
    results_root = Path(resolve_results_root(results_root))
    rows = [
        {
            **job_overview(path, results_root),
            "status": "pending",
            "started_at_utc": "",
            "finished_at_utc": "",
            "exit_code": "",
            "result_run_id": "",
            "result_run_path": "",
        }
        for path in manifest["jobs"]
    ]
    table = results_root / "agent_b" / "experiment_run_table.csv"
    _write_table(table, rows)
    for row in rows:
        row["status"] = "running"
        row["started_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_table(table, rows)
        result_group = Path(row["result_path"])
        before = set(result_group.glob("*/run_summary.json")) if result_group.is_dir() else set()
        command = [
            sys.executable, "-u", "-m", "coop_navigation_sds.batch",
            "--job-file", row["path"],
            "--results-dir", str(results_root),
            "--progress",
        ]
        completed = subprocess.run(command, cwd=ROOT, check=False)
        after = set(result_group.glob("*/run_summary.json")) if result_group.is_dir() else set()
        created = sorted(after - before, key=lambda item: item.stat().st_mtime, reverse=True)
        row["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        row["exit_code"] = completed.returncode
        if completed.returncode == 0 and len(created) == 1:
            row["status"] = "completed"
            row["result_run_id"] = created[0].parent.name
            row["result_run_path"] = str(created[0].parent)
        else:
            row["status"] = "failed"
        _write_table(table, rows)
        if row["status"] == "failed" and not continue_on_error:
            raise RuntimeError(f"Agent B job failed: {row['job']}")
    completed = [row for row in rows if row["status"] == "completed"]
    if completed:
        compare_runs(
            [results_root / "agent_b"],
            results_root / "agent_b" / "comparison",
        )
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True, help="Batch manifest JSON.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args(argv)
    manifest = load_batch_manifest(args.batch)
    overviews = [job_overview(path, args.results_dir) for path in manifest["jobs"]]
    print(f"Batch: {manifest['name']} | jobs={len(overviews)} | conditions={sum(row['conditions'] for row in overviews)}")
    for index, row in enumerate(overviews, start=1):
        print(
            f"{index}. {row['agent_b_size']} {row['model_role']} | "
            f"Agent B={row['agent_b_model']} | Agent A={row['agent_a']} "
            f"({row['agent_a_model']}) | n={row['conditions']} | {row['result_group']}"
        )
    if args.preview:
        return
    table = run_manifest(args.batch, args.results_dir, args.continue_on_error)
    print(f"Run table: {table}")
    report = Path(args.results_dir).resolve() / "agent_b" / "comparison" / "comparison_report.html"
    if report.is_file():
        print(f"Outcome and metric indicator report: {report}")


if __name__ == "__main__":
    main()
