import csv
import json
from pathlib import Path

from coop_navigation_sds.ResultsAndArtifacts.comparison import (
    compare_runs,
    discover_run_directories,
)


def _write_run(root, run_id, asr_engine, metric_value):
    run = root / run_id
    run.mkdir()
    condition = {
        "result_run_id": run_id,
        "condition_id": f"{run_id}-condition",
        "run_type": "audio_variant",
        "agent_a_type": "tinyllama",
        "agent_b_model": "TinyLlama",
        "tts_engine": "piper",
        "asr_engine": asr_engine,
        "route_valid": True,
        "task_success": metric_value > 0.5,
    }
    (run / "conditions.jsonl").write_text(json.dumps(condition) + "\n", encoding="utf-8")
    (run / "run_summary.json").write_text(
        json.dumps({"result_run_id": run_id, "condition_count": 1}),
        encoding="utf-8",
    )
    with (run / "metrics_long.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "condition_id", "run_type", "metric_key", "phase", "metric_label",
            "value_numeric", "available", "higher_is_better", "range_min", "range_max",
        ))
        writer.writeheader()
        writer.writerow({
            "condition_id": condition["condition_id"],
            "run_type": "audio_variant",
            "metric_key": "whole_dialogue.task_success",
            "phase": "whole_dialogue",
            "metric_label": "Task success",
            "value_numeric": metric_value,
            "available": True,
            "higher_is_better": True,
            "range_min": 0,
            "range_max": 1,
        })
    return run


def test_comparison_discovers_runs_and_writes_portable_artifacts(tmp_path):
    first = _write_run(tmp_path, "fwh-run", "faster_whisper", 1.0)
    _write_run(tmp_path, "vosk-run", "vosk", 0.0)

    assert discover_run_directories([tmp_path])[0] == first
    paths = compare_runs([tmp_path], tmp_path / "comparison")

    assert all(path.is_file() for path in paths.values())
    summary = list(csv.DictReader(paths["summary"].open(encoding="utf-8")))
    deltas = list(csv.DictReader(paths["deltas"].open(encoding="utf-8")))
    report = paths["report"].read_text(encoding="utf-8")
    assert {row["asr_engine"] for row in summary} == {"faster_whisper", "vosk"}
    assert len(deltas) == 1
    assert "<svg" in report
    assert "Task success" in report
