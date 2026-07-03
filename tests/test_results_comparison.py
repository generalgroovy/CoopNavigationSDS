import csv
import json
from pathlib import Path

from coop_navigation_sds.ResultsAndArtifacts.comparison import (
    compare_runs,
    discover_run_directories,
    identify_metric_outliers,
    summarize_metric_indicators,
    summarize_run_outcomes,
)


def _write_run(
    root, run_id, asr_engine, metric_value,
    metric_key="whole_dialogue.task_success", phase="whole_dialogue",
    metric_label="Task success",
):
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
            "metric_key": metric_key,
            "phase": phase,
            "metric_label": metric_label,
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
    assert "Run task outcomes" in report
    assert "Metric outliers and outcome alignment" in report
    assert "Metrics indicating observed outcomes" in report
    outcomes = list(csv.DictReader(paths["run_outcomes"].open(encoding="utf-8")))
    assert {row["task_outcome_status"] for row in outcomes} == {"all_successful", "all_failed"}


def test_pre_outcome_outlier_is_linked_to_failed_condition_without_label_leakage():
    conditions = []
    metrics = []
    for index, value in enumerate((0.88, 0.90, 0.91, 0.92, 0.20)):
        run = f"run-{index}"
        condition = f"condition-{index}"
        succeeded = index < 4
        conditions.append({
            "source_run": run,
            "source_path": run,
            "condition_id": condition,
            "task_success": succeeded,
            "route_valid": succeeded,
        })
        metric = {
            "source_run": run,
            "source_path": run,
            "condition_id": condition,
            "run_type": "audio_variant",
            "metric_key": "automatic_speech_recognition.semantic_accuracy",
            "metric_label": "Semantic recognition accuracy",
            "phase": "automatic_speech_recognition",
            "value_numeric": value,
            "available": True,
            "higher_is_better": True,
        }
        metrics.append(metric)
        metrics.append({
            **metric,
            "metric_key": "whole_dialogue.task_success",
            "phase": "whole_dialogue",
            "value_numeric": float(succeeded),
        })

    outliers = identify_metric_outliers(metrics, conditions)
    assert len(outliers) == 1
    assert outliers[0]["source_run"] == "run-4"
    assert outliers[0]["outlier_direction"] == "adverse"
    assert outliers[0]["outcome_alignment"] == "failure_aligned"
    assert all(row["phase"] != "whole_dialogue" for row in outliers)
    outcomes = summarize_run_outcomes(conditions, outliers)
    failed = next(row for row in outcomes if row["source_run"] == "run-4")
    assert failed["task_outcome_status"] == "all_failed"
    assert failed["failure_aligned_metric_outliers"] == 1
    indicators = summarize_metric_indicators(outliers)
    assert indicators[0]["metric_key"] == "automatic_speech_recognition.semantic_accuracy"
    assert indicators[0]["failure_aligned_count"] == 1
    assert indicators[0]["outcome_alignment_rate"] == 1.0


def test_comparison_report_exports_and_colors_failure_aligned_metric_outlier(tmp_path):
    for index, value in enumerate((0.88, 0.90, 0.91, 0.92, 0.20)):
        _write_run(
            tmp_path,
            f"run-{index}",
            "faster_whisper",
            value,
            metric_key="automatic_speech_recognition.semantic_accuracy",
            phase="automatic_speech_recognition",
            metric_label="Semantic recognition accuracy",
        )

    paths = compare_runs([tmp_path], tmp_path / "comparison")
    outliers = list(csv.DictReader(paths["outliers"].open(encoding="utf-8")))
    indicators = list(csv.DictReader(paths["metric_indicators"].open(encoding="utf-8")))
    report = paths["report"].read_text(encoding="utf-8")

    assert len(outliers) == 1
    assert outliers[0]["outcome_alignment"] == "failure_aligned"
    assert indicators[0]["failure_aligned_count"] == "1"
    assert '<tr class="failure_aligned">' in report
    assert "Semantic recognition accuracy" in report
