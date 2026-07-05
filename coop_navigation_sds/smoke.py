"""Fast deterministic end-to-end experiment used by local and CI checks."""
from __future__ import annotations

import argparse
from pathlib import Path

from coop_navigation_sds.DialogManagement.result import NullEventQueue
from coop_navigation_sds.app import (
    conversation_worker,
    default_run_config,
    validate_run_config_for_start,
)


def smoke_run_config(results_dir="results"):
    """Return a dependency-light configuration that traverses every runtime phase."""
    config = default_run_config()
    config.update({
        "agent_a_type": "staged",
        "llm_agent_a": False,
        "agent_b_plugin": "simple",
        "model_profile": "custom",
        "num_turns": 10,
        "agent_a_objective_mode": "shortest_valid_route_with_constraints",
        "maximum_progressive_constraints": 1,
        "minimum_compared_routes": 1,
        "tts_engine": "file",
        "asr_engine": "file",
        "speech_pattern_key": "clean",
        "speech_playback_enabled": False,
        "speech_realtime_enabled": False,
        "paired_audio_text_runs": False,
        "results_root": str(Path(results_dir)),
    })
    return config


def run_smoke(results_dir="results", event_queue=None):
    """Run the deterministic pipeline and return its result and artifact paths."""
    config = validate_run_config_for_start(smoke_run_config(results_dir))
    completed = conversation_worker(event_queue or NullEventQueue(), None, config)
    if completed is None:
        raise RuntimeError("Smoke experiment failed; inspect the emitted warning and run folder.")
    return completed


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the deterministic SDS smoke experiment.")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args(argv)
    result, paths = run_smoke(args.results_dir)
    print(f"Smoke outcome: {result.extra.get('conversation_outcome', 'unknown')}")
    print(f"Run folder: {paths['run_dir']}")
    print(f"Manifest: {paths['run_manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
