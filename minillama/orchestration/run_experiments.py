"""Execute one exported scheduler condition through the standard batch runtime."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

from coop_navigation_sds.Configuration.schema import JOB_SCHEMA_VERSION
from coop_navigation_sds.Configuration.slurm_grid import SLURM_GRID_SCHEMA_VERSION


def _load_condition(path):
    path = Path(path).expanduser().resolve()
    document = json.loads(path.read_text(encoding="utf-8"))
    if int(document.get("schema_version", -1)) != SLURM_GRID_SCHEMA_VERSION:
        raise ValueError(f"Unsupported scheduler condition schema in {path}.")
    required = {
        "condition_id", "grid_name", "index", "agent_b", "persona_key",
        "test_case_key", "run_mode", "seed", "repetition", "agent_a_type",
    }
    missing = sorted(required - set(document))
    if missing:
        raise ValueError(f"Condition file is missing: {', '.join(missing)}")
    return path, document


def condition_job_document(condition):
    """Translate one scheduler condition into the existing one-row job schema."""
    backend = dict(condition["agent_b"])
    config = {
        **dict(condition.get("base_config") or {}),
        **{key: value for key, value in backend.items() if key not in {"key", "plugin"}},
        **dict(condition.get("runtime_overrides") or {}),
        "agent_a_type": condition["agent_a_type"],
        "agent_b_plugin": backend["plugin"],
        "paired_audio_text_runs": False,
        "speech_playback_enabled": False,
        "speech_realtime_enabled": False,
        "log_profile": "full",
    }
    config.setdefault("network_seed", condition["seed"])
    config.setdefault("agent_a_seed", condition["seed"])
    config.setdefault("agent_b_seed", condition["seed"] + 1)
    if condition["run_mode"] == "speech":
        speech = dict(condition.get("speech") or {})
        tts_engine = speech["tts_engine"]
        asr_engine = speech["asr_engine"]
        config.update(speech)
        speech_pattern = condition["speech_pattern_key"]
    else:
        tts_engine = asr_engine = "file"
        speech_pattern = "clean"
        config.update({"tts_engine": "file", "asr_engine": "file"})
    model_key = backend.get("model_name") or backend["key"]
    return {
        "schema_version": JOB_SCHEMA_VERSION,
        "name": condition["condition_id"],
        "description": "One immutable scheduler-array condition.",
        "coverage_strategy": "full_factorial",
        "iterations": 1,
        "config": config,
        "grid": {
            "test_cases": [condition["test_case_key"]],
            "personas": [condition["persona_key"]],
            "speech_patterns": [speech_pattern],
            "model_params": ["greedy"],
            "objective_modes": ["shortest_valid_route_with_constraints"],
            "agent_a_audio_personas": [config.get("agent_a_audio_persona", "neutral_caller")],
            "agent_b_audio_personas": [config.get("agent_b_audio_persona", "clear_operator")],
            "tts_engines": [tts_engine],
            "asr_engines": [asr_engine],
            "agent_b_models": [model_key],
        },
        "parameter_values": {
            "network_seed": [condition["seed"]],
            "experiment_seed": [condition["seed"]],
            "repetition": [condition["repetition"]],
            "slurm_condition_index": [condition["index"]],
            "slurm_grid_name": [condition["grid_name"]],
            "run_mode": [condition["run_mode"]],
        },
    }


def run_condition(condition_file, results_dir, python_executable=None):
    """Run one condition and return the standard result directory."""
    condition_path, condition = _load_condition(condition_file)
    results_dir = Path(results_dir).expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    job_path = results_dir / "resolved_experiment.job"
    job_path.write_text(
        json.dumps(condition_job_document(condition), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    command = [
        str(python_executable or sys.executable),
        "-m", "coop_navigation_sds.batch",
        "--job-file", str(job_path),
        "--results-dir", str(results_dir),
        "--progress",
        "--no-update-coverage-registry",
    ]
    started = datetime.now(timezone.utc).isoformat()
    completed = subprocess.run(command, check=False)
    run_directories = sorted(
        path.parent for path in results_dir.glob("*/run_summary.json")
    )
    status = {
        "schema_version": 1,
        "condition_id": condition["condition_id"],
        "command": command,
        "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "return_code": completed.returncode,
        "result_directories": [str(path) for path in run_directories],
    }
    (results_dir / "task_summary.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command)
    if len(run_directories) != 1:
        raise RuntimeError(
            f"Expected one finalized result folder, found {len(run_directories)}."
        )
    shutil.copyfile(condition_path, run_directories[0] / "slurm_condition.json")
    return run_directories[0]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run one exported MiniLlama/CoopNavigationSDS scheduler condition."
    )
    parser.add_argument("--condition-file", required=True)
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args(argv)
    result = run_condition(args.condition_file, args.results_dir)
    print(f"Condition result: {result}", flush=True)


if __name__ == "__main__":
    main()
