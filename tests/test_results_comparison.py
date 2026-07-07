import csv
import json
from pathlib import Path
import pytest

from coop_navigation_sds.ResultsAndArtifacts.comparison import (
    build_performance_band_summary,
    build_run_phase_scorecard,
    compare_runs,
    discover_run_directories,
    identify_metric_outliers,
    summarize_metric_indicators,
    summarize_run_outcomes,
    write_evidence_comparison,
)


def test_performance_band_summary_reports_floor_ceiling_separation():
    rows = []
    for rank, band in enumerate(("floor", "challenging", "nominal", "ceiling")):
        rows.append({
            "source_run": "run",
            "agent_a_type": "tinyllama",
            "agent_b_model": "TinyLlama",
            "configured_tts_engine": "piper",
            "configured_asr_engine": "vosk",
            "speech_performance_band": band,
            "run_type": "audio_variant",
            "task_success": rank >= 2,
            "route_valid": rank >= 1,
            "constraint_satisfaction": rank / 3,
            "automatic_eval_score": 0.2 + 0.2 * rank,
            "word_error_rate": 0.8 - 0.2 * rank,
            "entity_error_rate": 0.6 - 0.15 * rank,
            "repair_success_rate": 0.25 * rank,
            "turn_count": 12 - rank,
            "runtime_sec": 10 + rank,
        })

    summary = build_performance_band_summary(rows)

    assert [row["speech_performance_band"] for row in summary] == [
        "floor", "challenging", "nominal", "ceiling",
    ]
    assert all(row["complete_band_set"] for row in summary)
    assert all(row["automatic_eval_order_monotonic"] for row in summary)
    assert summary[0]["ceiling_minus_floor_automatic_eval"] == pytest.approx(0.6)


def test_phase_scorecard_uses_recorded_pipeline_order():
    rows = build_run_phase_scorecard([
        {
            "source_run": "run",
            "condition_id": "c",
            "phase": "asr",
            "phase_order": 2,
            "metric_key": "asr_accuracy",
            "value_numeric": 0.8,
            "available": True,
            "higher_is_better": True,
            "range_min": 0,
            "range_max": 1,
        },
        {
            "source_run": "run",
            "condition_id": "c",
            "phase": "audio_input",
            "phase_order": 1,
            "metric_key": "capture_success",
            "value_numeric": 1.0,
            "available": True,
            "higher_is_better": True,
            "range_min": 0,
            "range_max": 1,
        },
    ])

    assert [row["phase"] for row in rows] == ["audio_input", "asr"]


def _write_run(
    root, run_id, asr_engine, metric_value,
    metric_key="whole_dialogue.task_success", phase="whole_dialogue",
    metric_label="Task success", llm_size="small", model="TinyLlama",
):
    run = root / run_id
    run.mkdir()
    condition = {
        "result_run_id": run_id,
        "condition_id": f"{run_id}-condition",
        "run_type": "audio_variant",
        "agent_a_type": "tinyllama",
        "agent_b_model": model,
        "agent_b_llm_size": llm_size,
        "agent_b_model_role": "primary",
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
    assert {row["asr_engine"] for row in summary} == {"faster_whisper", "vosk"}
    assert len(deltas) == 1
    matrix = list(csv.DictReader(paths["run_metric_matrix"].open(encoding="utf-8")))
    matrix_report = paths["run_metric_matrix_report"].read_text(encoding="utf-8")
    condition_analysis = list(csv.DictReader(paths["condition_analysis"].open(encoding="utf-8")))
    metric_column = "whole_dialogue | whole_dialogue.task_success"
    assert len(matrix) == 2
    assert {float(row[metric_column]) for row in matrix} == {0.0, 1.0}
    assert 'background:#C6EFCE' in matrix_report
    assert 'background:#F4CCCC' in matrix_report
    assert len(condition_analysis) == 2
    assert [row["outcome"] for row in condition_analysis] == ["success", "failure"]
    assert (first / "analysis_overview.html").is_file()
    assert (first / "condition_analysis.csv").is_file()
    assert "Matrix CSV" in matrix_report
    assert "Metric evidence" in matrix_report
    assert not (tmp_path / "comparison" / "comparison_report.html").exists()
    assert not (tmp_path / "comparison" / "phase_metric_overview.html").exists()


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
    matrix_report = paths["run_metric_matrix_report"].read_text(encoding="utf-8")

    assert len(outliers) == 1
    assert outliers[0]["outcome_alignment"] == "failure_aligned"
    assert indicators[0]["failure_aligned_count"] == "1"
    assert 'class="metric-cell failure-outlier"' in matrix_report
    assert "Semantic recognition accuracy" in matrix_report
    assert "automatic_speech_recognition" in matrix_report


def test_run_metric_matrix_sorts_models_smallest_to_largest(tmp_path):
    _write_run(tmp_path, "a-large", "vosk", 1.0, llm_size="large", model="Llama 3.1 8B")
    _write_run(tmp_path, "z-small", "vosk", 1.0, llm_size="small", model="Llama 3.2 1B")

    paths = compare_runs([tmp_path], tmp_path / "comparison")
    rows = list(csv.DictReader(paths["run_metric_matrix"].open(encoding="utf-8")))
    report = paths["run_metric_matrix_report"].read_text(encoding="utf-8")

    assert [row["agent_b_llm_size"] for row in rows] == ["small", "large"]
    assert [row["source_run"] for row in rows] == ["z-small", "a-large"]
    assert 'class="phase-header"' in report


def test_evidence_comparison_includes_partial_and_preflight_runs_without_mutation(tmp_path):
    completed = tmp_path / "run-partial"
    preflight = tmp_path / "run-preflight"
    completed.mkdir()
    preflight.mkdir()
    fields = [
        "condition_id", "pair_id", "run_type", "scenario_key", "persona_key",
        "speech_pattern_key", "agent_a_type", "agent_b_model", "agent_b_llm_size",
        "configured_tts_engine", "configured_asr_engine", "asr_beam_size", "network_seed",
    ]
    planned_rows = [
        {
            "condition_id": "condition-complete", "pair_id": "pair", "run_type": "audio_variant",
            "scenario_key": "scenario", "persona_key": "persona", "speech_pattern_key": "clean",
            "agent_a_type": "userlm", "agent_b_model": "/models/qwen", "agent_b_llm_size": "small",
            "configured_tts_engine": "piper", "configured_asr_engine": "faster_whisper",
            "asr_beam_size": "6", "network_seed": "42",
        },
        {
            "condition_id": "condition-missing", "pair_id": "pair", "run_type": "text_only",
            "scenario_key": "scenario", "persona_key": "persona", "speech_pattern_key": "clean",
            "agent_a_type": "userlm", "agent_b_model": "/models/qwen", "agent_b_llm_size": "small",
            "configured_tts_engine": "piper", "configured_asr_engine": "faster_whisper",
            "asr_beam_size": "6", "network_seed": "42",
        },
    ]
    for directory, rows in ((completed, planned_rows), (preflight, planned_rows[:1])):
        (directory / "experiment_job.json").write_text('{"schema_version":1}', encoding="utf-8")
        with (directory / "condition_configuration_breakdown.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    events = [
        {"name": "batch.condition.start", "payload": {"condition_id": "condition-complete"}},
        {"name": "telemetry.speech", "payload": {
            "turn": 1, "speaker": "Agent A", "generated_text": "Need a route.",
            "outgoing_text": "Need a route.", "raw_asr_transcript": "Need a route.",
            "agent_input_transcript": "Need a route.", "pipeline_ok": True,
            "tts_latency_sec": 0.1, "asr_latency_sec": 0.2, "pipeline_latency_sec": 0.3,
        }},
        {"name": "telemetry.phase_timing", "payload": {
            "turn": 1, "natural_language_generation_sec": 0.05,
            "text_to_speech_processing_sec": 0.1,
            "automatic_speech_recognition_processing_sec": 0.2,
            "speech_pipeline_wall_sec": 0.3,
        }},
        {"name": "metrics", "payload": {"metrics": (
            "[Run]\nOutcome: satisfied\nTask success: True\nStop reason: agent_a_closed\n"
            "[Task]\nRoute valid: True\nDestination reached: True\nDuration: 12 min\n"
            "Duration limit: 20 min\nConstraints satisfied: 2/2 met\n"
            "[Comparison]\nOptimal duration: 10 min\nDuration gap: 2 min\nCandidates compared: 2\n"
            "Revisions: 1\n[Execution]\nTurns: 3\nRuntime: 4.5 s\n"
        )}},
        {"name": "batch.condition.end", "payload": {"status": "ok"}},
    ]
    event_path = completed / "batch-condition.jsonl"
    event_path.write_text("".join(json.dumps(row) + "\n" for row in events), encoding="utf-8")
    source_before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    paths = write_evidence_comparison([tmp_path], tmp_path / "analysis")

    source_after = {path: path.read_bytes() for path in source_before}
    assert source_before == source_after
    inventory = list(csv.DictReader(paths["run_inventory"].open(encoding="utf-8")))
    outcomes = list(csv.DictReader(paths["outcomes"].open(encoding="utf-8")))
    assert {row["run_state"] for row in inventory} == {"partial", "preflight_only"}
    completed_row = next(row for row in outcomes if row["condition_id"] == "condition-complete")
    missing_rows = [row for row in outcomes if row["condition_state"] == "not_started"]
    assert completed_row["task_success"] == "True"
    assert missing_rows and all(row["task_success"] == "" for row in missing_rows)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["source_evidence_unchanged"] is True
    assert manifest["discovered_run_count"] == 2
    assert paths["overview"].is_file()
