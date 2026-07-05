"""Combine completed experiment runs into portable comparison tables and charts."""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path


REQUIRED_RUN_FILES = ("run_summary.json", "conditions.jsonl", "metrics_long.csv")
DISPLAY_METRIC_HINTS = (
    "task_success",
    "constraint_satisfaction",
    "route_valid",
    "word_error_rate",
    "repair_success",
    "turn_count",
    "automatic_eval",
)
OUTLIER_EXCLUDED_PHASES = frozenset({"task_outcome", "whole_dialogue", "metric_validity"})
OUTLIER_MINIMUM_SAMPLES = 5
OUTLIER_MODIFIED_Z_THRESHOLD = 3.5
CONDITION_ANALYSIS_METRICS = (
    ("route_validity", "task_outcome_route_validity"),
    ("constraint_satisfaction", "task_outcome_constraint_satisfaction_rate"),
    ("word_error_rate", "asr_wer"),
    ("entity_error_rate", "asr_entity_error_rate"),
    ("repair_success_rate", "dialogue_management_repair_success_rate"),
    ("grounded_proposal_score", "agent_b_grounded_proposal_score"),
    ("task_focus_score", "whole_dialogue_task_focus_score"),
    ("dialogue_cost", "whole_dialogue_dialogue_cost"),
    ("failure_localization_score", "whole_dialogue_failure_localization_score"),
)
SPEECH_BAND_RANK = {"floor": 0, "challenging": 1, "nominal": 2, "ceiling": 3}
SPEECH_BAND_SCORE_TOLERANCE = 0.02


def discover_run_directories(paths):
    """Return unique standard run directories found at or below ``paths``."""
    discovered = set()
    for value in paths:
        root = Path(value).expanduser().resolve()
        candidates = [root] if root.is_dir() else []
        if root.is_dir():
            candidates.extend(path.parent for path in root.rglob("run_summary.json"))
        for candidate in candidates:
            if all((candidate / name).is_file() for name in REQUIRED_RUN_FILES):
                discovered.add(candidate)
    return sorted(discovered, key=lambda path: str(path).casefold())


def _read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, rows):
    rows = list(rows)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        if not fieldnames:
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _truthy(value):
    if isinstance(value, bool):
        return value
    return str(value or "").strip().casefold() in {"1", "true", "yes"}


def _mean(values):
    numeric = [value for value in values if value is not None]
    return statistics.fmean(numeric) if numeric else None


def _phase_order(value):
    number = _number(value)
    return int(number) if number is not None else 999


def _joined_values(rows, key):
    return ";".join(sorted({str(row.get(key)) for row in rows if row.get(key) not in (None, "")}))


def build_condition_analysis_rows(conditions, metrics):
    """Join headline factors, outcomes, and selected metrics per condition."""
    metric_lookup = {}
    text_lookup = {}
    for metric in metrics:
        if not _truthy(metric.get("available", True)):
            continue
        key = (str(metric.get("source_run", "")), str(metric.get("condition_id", "")), str(metric.get("metric_key", "")))
        number = _number(metric.get("value_numeric"))
        if number is not None:
            metric_lookup[key] = number
        text_value = metric.get("value_text") or (
            metric.get("value") if number is None else None
        )
        if text_value not in (None, ""):
            text_lookup[key] = str(text_value)
    rows = []
    for condition in conditions:
        run_id = str(condition.get("source_run", ""))
        condition_id = str(condition.get("condition_id", ""))
        success = _truthy(condition.get("task_success"))
        row = {
            "source_run": run_id,
            "source_path": condition.get("source_path", ""),
            "condition_id": condition_id,
            "pair_id": condition.get("pair_id", ""),
            "outcome": "success" if success else "failure",
            "task_success": success,
            "execution_status": condition.get("execution_status", ""),
            "failure_phase": text_lookup.get((run_id, condition_id, "whole_dialogue_failure_phase"), ""),
            "pipeline_failure_type": condition.get("pipeline_failure_type", ""),
            "scenario_key": condition.get("scenario_key", ""),
            "persona_key": condition.get("persona_key", ""),
            "run_type": condition.get("run_type", ""),
            "speech_pattern_key": condition.get("speech_pattern_key", ""),
            "speech_performance_band": condition.get("speech_performance_band", ""),
            "speech_performance_rank": condition.get("speech_performance_rank", ""),
            "agent_a_type": condition.get("agent_a_type", ""),
            "agent_b_model": condition.get("agent_b_model", ""),
            "agent_b_llm_size": condition.get("agent_b_llm_size", ""),
            "configured_tts_engine": condition.get("configured_tts_engine", condition.get("tts_engine", "")),
            "configured_asr_engine": condition.get("configured_asr_engine", condition.get("asr_engine", "")),
            "asr_search_width": condition.get("asr_search_width", ""),
            "experiment_seed": condition.get("experiment_seed", ""),
            "iteration": condition.get("iteration", ""),
            "route_valid": _truthy(condition.get("route_valid")),
            "route_duration_min": _number(condition.get("route_duration_min")),
            "turn_count": _number(condition.get("turn_count")),
            "runtime_sec": _number(condition.get("runtime_sec")),
            "automatic_eval_score": _number(condition.get("automatic_eval_score")),
        }
        for output_key, metric_key in CONDITION_ANALYSIS_METRICS:
            row[output_key] = metric_lookup.get((run_id, condition_id, metric_key))
        rows.append(row)
    size_rank = {"small": 0, "medium": 1, "large": 2, "hosted": 3}
    rows.sort(key=lambda row: (
        size_rank.get(str(row["agent_b_llm_size"]).casefold(), 99),
        str(row["agent_b_model"]).casefold(),
        str(row["source_run"]).casefold(),
        0 if row["outcome"] == "failure" else 1,
        str(row["condition_id"]).casefold(),
    ))
    return rows


def build_run_analysis_rows(condition_rows):
    """Aggregate the condition explorer into one comparable row per run."""
    grouped = defaultdict(list)
    for row in condition_rows:
        grouped[str(row.get("source_run", ""))].append(row)
    output = []
    for run_id, rows in grouped.items():
        successes = sum(row["task_success"] for row in rows)
        audio = [row for row in rows if row.get("run_type") == "audio_variant"]
        text = [row for row in rows if row.get("run_type") == "text_only"]
        paired = defaultdict(dict)
        for row in rows:
            if row.get("pair_id"):
                paired[str(row["pair_id"])][str(row.get("run_type"))] = row
        pair_deltas = [
            float(pair["audio_variant"]["task_success"]) - float(pair["text_only"]["task_success"])
            for pair in paired.values()
            if "audio_variant" in pair and "text_only" in pair
        ]
        failure_phases = defaultdict(int)
        for row in rows:
            if not row["task_success"]:
                failure_phases[row.get("failure_phase") or row.get("pipeline_failure_type") or "task_outcome"] += 1
        output.append({
            "agent_b_llm_size": _joined_values(rows, "agent_b_llm_size") or "unknown",
            "agent_a_type": _joined_values(rows, "agent_a_type") or "unknown",
            "agent_b_model": _joined_values(rows, "agent_b_model") or "unknown",
            "source_run": run_id,
            "source_path": rows[0].get("source_path", ""),
            "condition_count": len(rows),
            "successful_condition_count": successes,
            "failed_condition_count": len(rows) - successes,
            "task_success_rate": successes / len(rows) if rows else None,
            "audio_condition_count": len(audio),
            "text_condition_count": len(text),
            "audio_task_success_rate": _mean([float(row["task_success"]) for row in audio]),
            "text_task_success_rate": _mean([float(row["task_success"]) for row in text]),
            "paired_condition_count": len(pair_deltas),
            "paired_task_success_delta": _mean(pair_deltas),
            "scenario_count": len({row.get("scenario_key") for row in rows}),
            "persona_count": len({row.get("persona_key") for row in rows}),
            "configured_tts_engines": _joined_values(rows, "configured_tts_engine"),
            "configured_asr_engines": _joined_values(rows, "configured_asr_engine"),
            "mean_route_validity": _mean([row.get("route_validity") for row in rows]),
            "mean_constraint_satisfaction": _mean([row.get("constraint_satisfaction") for row in rows]),
            "mean_audio_word_error_rate": _mean([row.get("word_error_rate") for row in audio]),
            "mean_audio_entity_error_rate": _mean([row.get("entity_error_rate") for row in audio]),
            "mean_repair_success_rate": _mean([row.get("repair_success_rate") for row in rows]),
            "mean_grounded_proposal_score": _mean([row.get("grounded_proposal_score") for row in rows]),
            "mean_task_focus_score": _mean([row.get("task_focus_score") for row in rows]),
            "mean_turn_count": _mean([row.get("turn_count") for row in rows]),
            "mean_runtime_sec": _mean([row.get("runtime_sec") for row in rows]),
            "failure_phase_counts": ";".join(
                f"{phase}:{count}" for phase, count in sorted(failure_phases.items())
            ),
        })
    size_rank = {"small": 0, "medium": 1, "large": 2, "hosted": 3}
    output.sort(key=lambda row: (
        size_rank.get(str(row["agent_b_llm_size"]).casefold(), 99),
        str(row["agent_b_model"]).casefold(),
        str(row["agent_a_type"]).casefold(),
        str(row["source_run"]).casefold(),
    ))
    return output


def build_performance_band_summary(condition_rows):
    """Aggregate outcome and diagnostic measures by preregistered speech band."""
    grouped = defaultdict(list)
    for row in condition_rows:
        band = str(row.get("speech_performance_band") or "").strip().lower()
        if row.get("run_type") != "audio_variant" or band not in SPEECH_BAND_RANK:
            continue
        key = (
            row.get("source_run"), row.get("agent_a_type"), row.get("agent_b_model"),
            row.get("configured_tts_engine"), row.get("configured_asr_engine"), band,
        )
        grouped[key].append(row)
    summaries = []
    for key, rows in grouped.items():
        summaries.append({
            "source_run": key[0],
            "agent_a_type": key[1],
            "agent_b_model": key[2],
            "configured_tts_engine": key[3],
            "configured_asr_engine": key[4],
            "speech_performance_band": key[5],
            "expected_rank": SPEECH_BAND_RANK[key[5]],
            "condition_count": len(rows),
            "task_success_rate": _mean([float(row["task_success"]) for row in rows]),
            "route_validity_rate": _mean([float(row["route_valid"]) for row in rows]),
            "constraint_satisfaction_rate": _mean([row.get("constraint_satisfaction") for row in rows]),
            "mean_automatic_eval_score": _mean([row.get("automatic_eval_score") for row in rows]),
            "mean_word_error_rate": _mean([row.get("word_error_rate") for row in rows]),
            "mean_entity_error_rate": _mean([row.get("entity_error_rate") for row in rows]),
            "mean_repair_success_rate": _mean([row.get("repair_success_rate") for row in rows]),
            "mean_turn_count": _mean([row.get("turn_count") for row in rows]),
            "mean_runtime_sec": _mean([row.get("runtime_sec") for row in rows]),
        })
    treatment_groups = defaultdict(list)
    for row in summaries:
        identity = tuple(row[key] for key in (
            "source_run", "agent_a_type", "agent_b_model",
            "configured_tts_engine", "configured_asr_engine",
        ))
        treatment_groups[identity].append(row)
    for rows in treatment_groups.values():
        rows.sort(key=lambda row: row["expected_rank"])
        complete = {row["speech_performance_band"] for row in rows} == set(SPEECH_BAND_RANK)
        scores = [row["mean_automatic_eval_score"] for row in rows]
        successes = [row["task_success_rate"] for row in rows]
        monotonic_score = complete and all(
            left is not None and right is not None
            and left <= right + SPEECH_BAND_SCORE_TOLERANCE
            for left, right in zip(scores, scores[1:])
        )
        monotonic_success = complete and all(
            left is not None and right is not None and left <= right
            for left, right in zip(successes, successes[1:])
        )
        floor = next((row for row in rows if row["speech_performance_band"] == "floor"), None)
        ceiling = next((row for row in rows if row["speech_performance_band"] == "ceiling"), None)
        score_gap = (
            ceiling["mean_automatic_eval_score"] - floor["mean_automatic_eval_score"]
            if floor and ceiling
            and floor["mean_automatic_eval_score"] is not None
            and ceiling["mean_automatic_eval_score"] is not None
            else None
        )
        success_gap = (
            ceiling["task_success_rate"] - floor["task_success_rate"]
            if floor and ceiling else None
        )
        for row in rows:
            row["complete_band_set"] = complete
            row["automatic_eval_order_monotonic"] = monotonic_score
            row["automatic_eval_order_tolerance"] = SPEECH_BAND_SCORE_TOLERANCE
            row["task_success_order_monotonic"] = monotonic_success
            row["ceiling_minus_floor_automatic_eval"] = score_gap
            row["ceiling_minus_floor_task_success"] = success_gap
    return sorted(summaries, key=lambda row: (
        str(row["agent_b_model"]).casefold(),
        str(row["source_run"]).casefold(),
        row["expected_rank"],
    ))


def build_run_phase_scorecard(metrics):
    """Build comparable per-run phase scores from declared-range metrics only."""
    summaries = build_phase_metric_summary(metrics)
    registered = defaultdict(set)
    available = defaultdict(set)
    scores = defaultdict(list)
    observations = defaultdict(int)
    phase_orders = {}
    for row in metrics:
        key = (
            str(row.get("source_run") or row.get("result_run_id") or "current_run"),
            str(row.get("phase") or "unassigned"),
        )
        registered[key].add(str(row.get("metric_key") or "unnamed_metric"))
        phase_orders[key] = min(
            phase_orders.get(key, 999), _phase_order(row.get("phase_order"))
        )
        if _truthy(row.get("available", True)) and _number(row.get("value_numeric")) is not None:
            available[key].add(str(row.get("metric_key") or "unnamed_metric"))
    for row in summaries:
        key = (row["source_run"], row["phase"])
        observations[key] += int(row["observation_count"])
        if row["range_source"] == "declared":
            scores[key].append(float(row["direction_adjusted_score"]))
    output = []
    for key in sorted(registered):
        run_id, phase = key
        total = len(registered[key])
        available_count = len(available[key])
        phase_scores = scores[key]
        output.append({
            "source_run": run_id,
            "phase": phase,
            "phase_order": phase_orders.get(key, 999),
            "phase_score": _mean(phase_scores),
            "declared_range_metric_count": len(phase_scores),
            "available_metric_count": available_count,
            "registered_metric_count": total,
            "metric_coverage_rate": available_count / total if total else None,
            "observation_count": observations[key],
            "interpretation": "equal-weighted descriptive score over available declared-range metrics; inspect component metrics before inference",
        })
    return sorted(output, key=lambda row: (
        row["source_run"], row["phase_order"], row["phase"]
    ))


def load_comparison_data(run_directories):
    """Load and join canonical condition and long-form metric records."""
    conditions = []
    metrics = []
    for run_dir in run_directories:
        summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
        run_id = str(summary.get("result_run_id") or run_dir.name)
        condition_rows = _read_jsonl(run_dir / "conditions.jsonl")
        by_condition = {str(row.get("condition_id")): row for row in condition_rows}
        for row in condition_rows:
            conditions.append({"source_run": run_id, "source_path": str(run_dir), **row})
        for row in _read_csv(run_dir / "metrics_long.csv"):
            condition = by_condition.get(str(row.get("condition_id")), {})
            metrics.append({
                "source_run": run_id,
                "source_path": str(run_dir),
                "matrix_family": condition.get("matrix_family", ""),
                "experiment_platform": condition.get("experiment_platform", ""),
                "agent_a_type": condition.get("agent_a_type", ""),
                "agent_a_audio_persona": condition.get("agent_a_audio_persona", ""),
                "agent_b_audio_persona": condition.get("agent_b_audio_persona", ""),
                "agent_b_model": condition.get("agent_b_model", row.get("model_name", "")),
                "agent_b_llm_size": condition.get("agent_b_llm_size", ""),
                "tts_engine": condition.get("tts_engine", row.get("factor_tts_engine", "")),
                "asr_engine": condition.get("asr_engine", row.get("factor_asr_engine", "")),
                "asr_search_width": condition.get("asr_search_width", ""),
                "scenario_key": condition.get("scenario_key", row.get("scenario_key", "")),
                "persona_key": condition.get("persona_key", row.get("persona_key", "")),
                "speech_pattern_key": condition.get("speech_pattern_key", row.get("speech_pattern_key", "")),
                **row,
            })
    return conditions, metrics


def summarize_metrics(metrics):
    """Aggregate each numeric metric by run and experimental component tuple."""
    groups = defaultdict(list)
    metadata = {}
    dimensions = (
        "source_run", "matrix_family", "experiment_platform",
        "scenario_key", "persona_key", "speech_pattern_key",
        "agent_a_type", "agent_a_audio_persona", "agent_b_audio_persona",
        "agent_b_model", "agent_b_llm_size", "tts_engine", "asr_engine",
        "asr_search_width",
        "run_type", "metric_key", "phase", "metric_label", "unit",
        "higher_is_better", "range_min", "range_max",
    )
    for row in metrics:
        if str(row.get("available", "true")).casefold() not in {"true", "1", "yes"}:
            continue
        value = _number(row.get("value_numeric"))
        if value is None:
            continue
        key = tuple(str(row.get(name, "")) for name in dimensions)
        groups[key].append(value)
        metadata[key] = {name: row.get(name, "") for name in dimensions}
    summaries = []
    for key, values in sorted(groups.items()):
        row = metadata[key]
        summaries.append({
            **row,
            "sample_count": len(values),
            "mean": statistics.fmean(values),
            "standard_deviation": statistics.stdev(values) if len(values) > 1 else 0.0,
            "minimum": min(values),
            "maximum": max(values),
        })
    return summaries


def calculate_run_deltas(summaries):
    """Calculate pairwise mean differences for matching metrics and run types."""
    grouped = defaultdict(list)
    for row in summaries:
        key = (
            row.get("matrix_family"), row.get("scenario_key"), row.get("persona_key"),
            row.get("speech_pattern_key"), row.get("agent_a_type"),
            row.get("agent_a_audio_persona"), row.get("agent_b_audio_persona"),
            row.get("agent_b_model"), row.get("agent_b_llm_size"),
            row.get("tts_engine"), row.get("asr_search_width"),
            row.get("run_type"), row.get("metric_key"), row.get("phase"),
        )
        grouped[key].append(row)
    deltas = []
    for rows in grouped.values():
        rows = sorted(rows, key=lambda row: (str(row.get("asr_engine")), str(row.get("source_run"))))
        for index, left in enumerate(rows):
            for right in rows[index + 1:]:
                if (
                    left["source_run"] == right["source_run"]
                    and left.get("asr_engine") == right.get("asr_engine")
                ):
                    continue
                difference = float(right["mean"]) - float(left["mean"])
                direction = str(left.get("higher_is_better", "")).casefold()
                improvement = (
                    difference if direction == "true"
                    else -difference if direction == "false"
                    else None
                )
                deltas.append({
                    "baseline_run": left["source_run"],
                    "comparison_run": right["source_run"],
                    "baseline_asr": left.get("asr_engine", ""),
                    "comparison_asr": right.get("asr_engine", ""),
                    "baseline_platform": left.get("experiment_platform", ""),
                    "comparison_platform": right.get("experiment_platform", ""),
                    "matrix_family": left.get("matrix_family", ""),
                    "scenario_key": left.get("scenario_key", ""),
                    "persona_key": left.get("persona_key", ""),
                    "speech_pattern_key": left.get("speech_pattern_key", ""),
                    "agent_a_type": left.get("agent_a_type", ""),
                    "agent_a_audio_persona": left.get("agent_a_audio_persona", ""),
                    "agent_b_audio_persona": left.get("agent_b_audio_persona", ""),
                    "agent_b_model": left.get("agent_b_model", ""),
                    "agent_b_llm_size": left.get("agent_b_llm_size", ""),
                    "tts_engine": left.get("tts_engine", ""),
                    "asr_search_width": left.get("asr_search_width", ""),
                    "run_type": left.get("run_type", ""),
                    "phase": left.get("phase", ""),
                    "metric_key": left.get("metric_key", ""),
                    "baseline_mean": left["mean"],
                    "comparison_mean": right["mean"],
                    "difference": difference,
                    "direction_adjusted_improvement": improvement,
                })
    return deltas


def identify_metric_outliers(
    metrics,
    conditions,
    minimum_samples=OUTLIER_MINIMUM_SAMPLES,
    modified_z_threshold=OUTLIER_MODIFIED_Z_THRESHOLD,
):
    """Find robust pre-outcome metric outliers and relate them to task outcomes.

    Detection uses the median absolute deviation within metric, phase, and run
    type. Labels are descriptive associations, not validated failure predictors.
    """
    outcomes = {
        (str(row.get("source_run", "")), str(row.get("condition_id", ""))): row
        for row in conditions
    }
    grouped = defaultdict(list)
    for row in metrics:
        phase = str(row.get("phase", ""))
        if phase in OUTLIER_EXCLUDED_PHASES:
            continue
        if str(row.get("available", "true")).casefold() not in {"true", "1", "yes"}:
            continue
        value = _number(row.get("value_numeric"))
        if value is None:
            continue
        key = (str(row.get("metric_key", "")), phase, str(row.get("run_type", "")))
        grouped[key].append((row, value))

    outliers = []
    for observations in grouped.values():
        if len(observations) < int(minimum_samples):
            continue
        values = [value for _row, value in observations]
        median = statistics.median(values)
        deviations = [abs(value - median) for value in values]
        mad = statistics.median(deviations)
        if mad <= 0:
            continue
        for row, value in observations:
            modified_z = 0.67448975 * (value - median) / mad
            if abs(modified_z) < float(modified_z_threshold):
                continue
            higher_is_better = str(row.get("higher_is_better", "")).casefold()
            high_outlier = value > median
            if higher_is_better in {"true", "1", "yes"}:
                direction = "favorable" if high_outlier else "adverse"
            elif higher_is_better in {"false", "0", "no"}:
                direction = "adverse" if high_outlier else "favorable"
            else:
                direction = "unknown"
            outcome = outcomes.get(
                (str(row.get("source_run", "")), str(row.get("condition_id", ""))),
                {},
            )
            succeeded = bool(outcome.get("task_success"))
            if direction == "adverse" and not succeeded:
                signal = "failure_aligned"
            elif direction == "favorable" and succeeded:
                signal = "success_aligned"
            elif direction == "unknown":
                signal = "direction_unknown"
            else:
                signal = "outcome_contradicting"
            outliers.append({
                "source_run": row.get("source_run", ""),
                "source_path": row.get("source_path", ""),
                "condition_id": row.get("condition_id", ""),
                "run_type": row.get("run_type", ""),
                "task_success": succeeded,
                "route_valid": bool(outcome.get("route_valid")),
                "metric_key": row.get("metric_key", ""),
                "metric_label": row.get("metric_label", row.get("metric_key", "")),
                "phase": row.get("phase", ""),
                "value": value,
                "reference_median": median,
                "median_absolute_deviation": mad,
                "modified_z_score": modified_z,
                "threshold": float(modified_z_threshold),
                "higher_is_better": row.get("higher_is_better", ""),
                "outlier_direction": direction,
                "outcome_alignment": signal,
                "interpretation": "descriptive association; not a validated causal failure indicator",
            })
    return sorted(
        outliers,
        key=lambda row: (str(row["source_run"]), str(row["condition_id"]), -abs(row["modified_z_score"])),
    )


def summarize_run_outcomes(conditions, outliers=()):
    """Summarize condition-level task outcomes and aligned metric signals per run."""
    grouped = defaultdict(list)
    for row in conditions:
        grouped[(str(row.get("source_run", "")), str(row.get("source_path", "")))].append(row)
    signal_counts = defaultdict(lambda: defaultdict(int))
    for row in outliers:
        signal_counts[str(row.get("source_run", ""))][str(row.get("outcome_alignment", ""))] += 1
    output = []
    for (run_id, source_path), rows in sorted(grouped.items()):
        successes = sum(bool(row.get("task_success")) for row in rows)
        failures = len(rows) - successes
        status = "all_successful" if not failures else "all_failed" if not successes else "mixed"
        output.append({
            "source_run": run_id,
            "source_path": source_path,
            "condition_count": len(rows),
            "successful_condition_count": successes,
            "failed_condition_count": failures,
            "task_success_rate": successes / len(rows) if rows else None,
            "task_outcome_status": status,
            "failure_aligned_metric_outliers": signal_counts[run_id]["failure_aligned"],
            "success_aligned_metric_outliers": signal_counts[run_id]["success_aligned"],
            "contradicting_metric_outliers": signal_counts[run_id]["outcome_contradicting"],
        })
    return output


def build_run_phase_metric_matrix(conditions, metrics, outliers=()):
    """Return one completed-run row with one mean-value column per phase metric."""
    outcomes = {
        row["source_run"]: row
        for row in summarize_run_outcomes(conditions)
    }
    grouped = defaultdict(list)
    condition_metadata = {}
    specifications = {}
    observed = defaultdict(list)
    for row in metrics:
        if str(row.get("available", "true")).casefold() not in {"true", "1", "yes"}:
            continue
        value = _number(row.get("value_numeric"))
        if value is None:
            continue
        phase = str(row.get("phase") or "unassigned")
        metric_key = str(row.get("metric_key") or "unnamed_metric")
        column = f"{phase} | {metric_key}"
        grouped[(str(row.get("source_run", "")), column)].append(value)
        observed[column].append(value)
        specifications.setdefault(column, {
            "phase": phase,
            "phase_order": _phase_order(row.get("phase_order")),
            "metric_key": metric_key,
            "metric_label": str(row.get("metric_label") or metric_key),
            "higher_is_better": str(row.get("higher_is_better", "")).casefold() in {"true", "1", "yes"},
            "range_min": _number(row.get("range_min")),
            "range_max": _number(row.get("range_max")),
        })
    for condition in conditions:
        condition_metadata.setdefault(str(condition.get("source_run", "")), condition)
    for column, specification in specifications.items():
        values = observed[column]
        if specification["range_min"] is None:
            specification["range_min"] = min(values)
        if specification["range_max"] is None:
            specification["range_max"] = max(values)

    metric_columns = sorted(
        specifications,
        key=lambda column: (
            specifications[column]["phase_order"],
            specifications[column]["phase"],
            specifications[column]["metric_key"],
        ),
    )
    cell_outliers = defaultdict(list)
    for outlier in outliers:
        column = f'{outlier.get("phase") or "unassigned"} | {outlier.get("metric_key") or "unnamed_metric"}'
        cell_outliers[(str(outlier.get("source_run", "")), column)].append(outlier)
    size_rank = {"small": 0, "medium": 1, "large": 2, "hosted": 3}
    rows = []
    for run_id, outcome in outcomes.items():
        metadata = condition_metadata.get(run_id, {})
        run_outliers = [item for (source_run, _column), items in cell_outliers.items() if source_run == run_id for item in items]
        row = {
            "agent_b_llm_size": metadata.get("agent_b_llm_size") or "unknown",
            "agent_a_type": metadata.get("agent_a_type") or "unknown",
            "agent_b_model": metadata.get("agent_b_model") or "unknown",
            "agent_b_model_role": metadata.get("agent_b_model_role") or "unknown",
            "source_run": run_id,
            "source_path": outcome["source_path"],
            "task_outcome_status": outcome["task_outcome_status"],
            "condition_count": outcome["condition_count"],
            "successful_condition_count": outcome["successful_condition_count"],
            "failed_condition_count": outcome["failed_condition_count"],
            "task_success_rate": outcome["task_success_rate"],
            "metric_outlier_count": len(run_outliers),
            "failure_aligned_outlier_count": sum(item.get("outcome_alignment") == "failure_aligned" for item in run_outliers),
            "success_aligned_outlier_count": sum(item.get("outcome_alignment") == "success_aligned" for item in run_outliers),
            "contradicting_outlier_count": sum(item.get("outcome_alignment") == "outcome_contradicting" for item in run_outliers),
        }
        for column in metric_columns:
            values = grouped.get((run_id, column), ())
            row[column] = statistics.fmean(values) if values else None
        rows.append(row)
    rows.sort(key=lambda row: (
        size_rank.get(str(row["agent_b_llm_size"]).casefold(), 99),
        str(row["agent_b_model"]).casefold(),
        str(row["agent_a_type"]).casefold(),
        str(row["source_run"]).casefold(),
    ))
    return rows, specifications, cell_outliers


def build_phase_metric_summary(metrics):
    """Aggregate available metric observations by finalized run and phase."""
    grouped = defaultdict(list)
    metadata = {}
    for row in metrics:
        if str(row.get("available", "true")).casefold() not in {"true", "1", "yes"}:
            continue
        value = _number(row.get("value_numeric"))
        if value is None:
            continue
        key = (
            str(row.get("source_run") or row.get("result_run_id") or "current_run"),
            str(row.get("phase") or "unassigned"),
            str(row.get("metric_key") or "unnamed_metric"),
        )
        grouped[key].append(value)
        metadata[key] = {
            "source_run": key[0],
            "phase": key[1],
            "phase_order": _phase_order(row.get("phase_order")),
            "metric_key": key[2],
            "metric_label": str(row.get("metric_label") or key[2]),
            "unit": str(row.get("unit") or ""),
            "higher_is_better": str(row.get("higher_is_better", "")).casefold() in {"true", "1", "yes"},
            "range_min": _number(row.get("range_min")),
            "range_max": _number(row.get("range_max")),
        }
    output = []
    for key, values in sorted(grouped.items()):
        row = metadata[key]
        minimum = row["range_min"] if row["range_min"] is not None else min(values)
        maximum = row["range_max"] if row["range_max"] is not None else max(values)
        mean = statistics.fmean(values)
        position = 0.5 if maximum == minimum else min(1.0, max(0.0, (mean - minimum) / (maximum - minimum)))
        score = position if row["higher_is_better"] else 1.0 - position
        output.append({
            **row,
            "observation_count": len(values),
            "mean": mean,
            "observed_minimum": min(values),
            "observed_maximum": max(values),
            "effective_range_min": minimum,
            "effective_range_max": maximum,
            "direction_adjusted_score": score,
            "range_source": (
                "declared" if row["range_min"] is not None and row["range_max"] is not None
                else "observed"
            ),
        })
    return sorted(output, key=lambda row: (
        row["phase_order"], row["phase"], row["source_run"], row["metric_key"]
    ))


def write_phase_metric_overview(path, rows):
    """Write an auditable phase-grouped metric visualization."""
    by_phase = defaultdict(list)
    for row in rows:
        by_phase[row["phase"]].append(row)
    sections = []
    for phase, phase_rows in sorted(
        by_phase.items(),
        key=lambda item: (min(row.get("phase_order", 999) for row in item[1]), item[0]),
    ):
        phase_score = statistics.fmean(row["direction_adjusted_score"] for row in phase_rows)
        table_rows = []
        for row in phase_rows:
            specification = {
                "range_min": 0.0,
                "range_max": 1.0,
                "higher_is_better": True,
            }
            color = _metric_cell_color(row["direction_adjusted_score"], specification)
            expected = f'{row["effective_range_min"]:.6g} to {row["effective_range_max"]:.6g}'
            table_rows.append(
                f'<tr><td>{html.escape(row["source_run"])}</td><td>{html.escape(row["metric_label"])}</td>'
                f'<td>{row["observation_count"]}</td><td>{row["mean"]:.6g}</td>'
                f'<td>{row["observed_minimum"]:.6g} to {row["observed_maximum"]:.6g}</td>'
                f'<td>{expected} ({row["range_source"]})</td>'
                f'<td>{"higher" if row["higher_is_better"] else "lower"}</td>'
                f'<td style="background:{color}">{100.0 * row["direction_adjusted_score"]:.1f}%</td></tr>'
            )
        phase_color = _metric_cell_color(
            phase_score,
            {"range_min": 0.0, "range_max": 1.0, "higher_is_better": True},
        )
        sections.append(
            f'<section><h2>{html.escape(phase)}</h2><p><strong style="background:{phase_color};padding:4px 8px">'
            f'Phase score {100.0 * phase_score:.1f}%</strong> Mean of direction-adjusted metric positions; not a validated composite outcome.</p>'
            f'<table><thead><tr><th>Run</th><th>Metric</th><th>N</th><th>Mean</th><th>Observed range</th>'
            f'<th>Evaluation range</th><th>Preferred direction</th><th>Normalized score</th></tr></thead>'
            f'<tbody>{"".join(table_rows)}</tbody></table></section>'
        )
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Metrics by pipeline phase</title>
<style>body{{font:14px system-ui,sans-serif;background:#f3f5f7;color:#202a35;margin:0}}main{{max-width:1600px;margin:auto;padding:20px}}section{{background:#fff;border:1px solid #cbd3dc;padding:14px;margin:12px 0;overflow:auto}}table{{border-collapse:collapse;width:100%}}th,td{{border-bottom:1px solid #dce2e8;padding:6px;text-align:left;white-space:nowrap}}th{{background:#eaf1ef}}</style></head>
<body><main><h1>Metric results by pipeline phase</h1><p>Raw means remain authoritative. Green indicates the favorable declared extreme, amber the midpoint, and red the adverse extreme after applying each metric's preferred direction.</p>{''.join(sections) or '<p>No available numeric metrics.</p>'}</main></body></html>"""
    Path(path).write_text(document, encoding="utf-8")


def write_phase_metric_outputs(metrics_or_path, output_directory):
    """Write graphable and visual phase metric summaries from long-form metrics."""
    metrics = _read_csv(metrics_or_path) if isinstance(metrics_or_path, (str, Path)) else list(metrics_or_path)
    output = Path(output_directory)
    paths = {
        "phase_metric_summary": output / "phase_metric_summary.csv",
        "phase_metric_report": output / "phase_metric_overview.html",
    }
    rows = build_phase_metric_summary(metrics)
    _write_csv(paths["phase_metric_summary"], rows)
    write_phase_metric_overview(paths["phase_metric_report"], rows)
    return paths


def _metric_cell_color(value, specification):
    """Map a metric to red/amber/green using its declared or observed extremes."""
    minimum = specification["range_min"]
    maximum = specification["range_max"]
    if value is None or minimum is None or maximum is None:
        return "#EEF1F4"
    position = 0.5 if maximum == minimum else (float(value) - minimum) / (maximum - minimum)
    position = min(1.0, max(0.0, position))
    if not specification["higher_is_better"]:
        position = 1.0 - position
    low, middle, high = (244, 204, 204), (255, 235, 156), (198, 239, 206)
    start, end, fraction = (low, middle, position * 2) if position <= 0.5 else (middle, high, (position - 0.5) * 2)
    rgb = tuple(round(left + (right - left) * fraction) for left, right in zip(start, end))
    return "#" + "".join(f"{channel:02X}" for channel in rgb)


def write_run_phase_metric_matrix_html(path, rows, specifications, cell_outliers=None):
    """Write the human-readable color matrix while retaining numeric cell values."""
    metric_columns = sorted(
        specifications,
        key=lambda column: (
            specifications[column]["phase_order"],
            specifications[column]["phase"],
            specifications[column]["metric_key"],
        ),
    )
    cell_outliers = cell_outliers or {}
    fixed_columns = (
        "agent_b_llm_size", "agent_a_type", "agent_b_model", "agent_b_model_role",
        "source_run", "task_outcome_status", "condition_count",
        "successful_condition_count", "failed_condition_count", "task_success_rate",
        "metric_outlier_count",
    )
    fixed_labels = {
        "agent_b_llm_size": "Size", "agent_a_type": "Agent A", "agent_b_model": "Agent B model",
        "agent_b_model_role": "Model role", "source_run": "Run", "task_outcome_status": "Outcome",
        "condition_count": "Conditions", "successful_condition_count": "Successful",
        "failed_condition_count": "Failed", "task_success_rate": "Success rate",
        "metric_outlier_count": "Metric outliers",
    }
    by_phase = defaultdict(list)
    for column in metric_columns:
        by_phase[specifications[column]["phase"]].append(column)
    fixed_header_html = "".join(
        f'<th rowspan="2">{html.escape(fixed_labels[column])}</th>' for column in fixed_columns
    )
    phase_header_html = "".join(
        f'<th colspan="{len(columns)}" class="phase-header">{html.escape(phase)}</th>'
        for phase, columns in by_phase.items()
    )
    metric_header_html = "".join(
        f'<th title="{html.escape(specifications[column]["metric_key"])}">'
        f'{html.escape(specifications[column]["metric_label"])}</th>'
        for columns in by_phase.values() for column in columns
    )
    body = []
    for row in rows:
        status = str(row["task_outcome_status"])
        cells = []
        for column in fixed_columns:
            value = row.get(column)
            display = f"{100 * value:.1f}%" if column == "task_success_rate" and value is not None else value
            cells.append(f'<td class="{html.escape(status)}">{html.escape(str(display))}</td>')
        for column in metric_columns:
            value = row.get(column)
            specification = specifications[column]
            title = (
                f'{specification["metric_label"]}; range {specification["range_min"]} to '
                f'{specification["range_max"]}; '
                f'{"higher" if specification["higher_is_better"] else "lower"} is better'
            )
            display = "" if value is None else f"{value:.6g}"
            outlier_rows = cell_outliers.get((str(row["source_run"]), column), ())
            alignments = {str(item.get("outcome_alignment")) for item in outlier_rows}
            outlier_class = (
                "failure-outlier" if "failure_aligned" in alignments
                else "contradicting-outlier" if "outcome_contradicting" in alignments
                else "success-outlier" if "success_aligned" in alignments
                else "unknown-outlier" if outlier_rows
                else ""
            )
            badge = f'<span class="outlier-badge">{len(outlier_rows)} outlier</span>' if outlier_rows else ""
            if outlier_rows:
                title += "; outliers=" + ", ".join(sorted(alignments))
            cells.append(
                f'<td class="metric-cell {outlier_class}" style="background:{_metric_cell_color(value, specification)}" '
                f'title="{html.escape(title)}">{display}{badge}</td>'
            )
        body.append("<tr>" + "".join(cells) + "</tr>")
    empty = '<tr><td colspan="6">No finalized runs were found.</td></tr>'
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Run phase metric matrix</title><style>
body{{font:14px system-ui,sans-serif;color:#202A35;background:#F3F5F7;margin:0;padding:20px}}
table{{border-collapse:separate;border-spacing:0;background:white;box-shadow:0 1px 4px #BBC3CC}}
th,td{{border-right:1px solid #DCE2E8;border-bottom:1px solid #DCE2E8;padding:6px 8px;white-space:nowrap}}
thead tr:first-child th{{position:sticky;top:0;background:#244B5A;color:white;text-align:left;z-index:3}}
thead tr:nth-child(2) th{{position:sticky;top:31px;background:#376879;color:white;text-align:left;z-index:2}}
.phase-header{{text-align:center!important;background:#173D4B!important}}
.all_successful{{background:#DFF1E7}}.mixed{{background:#FFF0C9}}.all_failed{{background:#F7DADA}}
.metric-cell{{position:relative;padding-right:9px}}.outlier-badge{{display:block;font-size:10px;font-weight:700;margin-top:3px}}
.failure-outlier{{box-shadow:inset 0 0 0 3px #B42318}}.failure-outlier .outlier-badge{{color:#8A1710}}
.success-outlier{{box-shadow:inset 0 0 0 3px #14804A}}.success-outlier .outlier-badge{{color:#0D6338}}
.contradicting-outlier,.unknown-outlier{{box-shadow:inset 0 0 0 3px #6F4BA8}}.contradicting-outlier .outlier-badge,.unknown-outlier .outlier-badge{{color:#563884}}
a{{color:#174F69}}nav{{margin-bottom:12px}}nav a{{margin-right:14px;font-weight:650}}
</style></head><body><h1>Completed runs by phase metric</h1>
<nav><a href="run_phase_metric_matrix.csv">Matrix CSV</a><a href="combined_metrics_long.csv">Metric evidence</a><a href="condition_analysis.csv">Conditions</a><a href="metric_outliers.csv">Outliers</a></nav>
<p>Every finalized run appears once, ordered from smallest to largest Agent B. Outcome fields are green, amber, or red. Metric fill is normalized between declared or observed extremes and respects metric direction. Outlier borders are red for failure-aligned, green for success-aligned, and violet for contradictory or unknown alignment.</p>
<table><thead><tr>{fixed_header_html}{phase_header_html}</tr><tr>{metric_header_html}</tr></thead><tbody>{''.join(body) or empty}</tbody></table></body></html>"""
    Path(path).write_text(document, encoding="utf-8")


def summarize_metric_indicators(outliers):
    """Aggregate which metric outliers align with observed success or failure."""
    grouped = defaultdict(list)
    for row in outliers:
        grouped[(str(row.get("phase", "")), str(row.get("metric_key", "")))].append(row)
    output = []
    for (phase, metric_key), rows in sorted(grouped.items()):
        failure_aligned = sum(row.get("outcome_alignment") == "failure_aligned" for row in rows)
        success_aligned = sum(row.get("outcome_alignment") == "success_aligned" for row in rows)
        contradicting = sum(row.get("outcome_alignment") == "outcome_contradicting" for row in rows)
        aligned = failure_aligned + success_aligned
        output.append({
            "phase": phase,
            "metric_key": metric_key,
            "metric_label": rows[0].get("metric_label", metric_key),
            "outlier_count": len(rows),
            "failure_aligned_count": failure_aligned,
            "success_aligned_count": success_aligned,
            "contradicting_count": contradicting,
            "direction_unknown_count": sum(row.get("outcome_alignment") == "direction_unknown" for row in rows),
            "outcome_alignment_rate": aligned / len(rows),
            "interpretation": "descriptive outlier alignment; requires held-out validation",
        })
    return output


def _selected_chart_rows(summaries, limit=12):
    audio = [row for row in summaries if row.get("run_type") == "audio_variant"]
    preferred = [
        row for row in audio
        if any(hint in str(row.get("metric_key", "")).casefold() for hint in DISPLAY_METRIC_HINTS)
    ]
    keys = []
    for row in preferred:
        metric_key = row.get("metric_key")
        if metric_key not in keys:
            keys.append(metric_key)
        if len(keys) >= limit:
            break
    return [row for row in preferred if row.get("metric_key") in keys]


def _bar_svg(rows):
    values = [abs(float(row["mean"])) for row in rows]
    scale = max(values, default=1.0) or 1.0
    width = 760
    row_height = 28
    height = max(80, 24 + row_height * len(rows))
    fragments = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Metric means">']
    colors = ("#176B5B", "#315E86", "#9A5A12", "#7A4E8A")
    for index, row in enumerate(rows):
        y = 10 + index * row_height
        label = f"{row.get('asr_engine')}: {float(row['mean']):.3f}"
        bar_width = 400 * abs(float(row["mean"])) / scale
        fragments.append(
            f'<text x="0" y="{y + 15}" font-size="12">{html.escape(label)}</text>'
            f'<rect x="250" y="{y}" width="{bar_width:.1f}" height="18" '
            f'fill="{colors[index % len(colors)]}" rx="2" />'
        )
    fragments.append("</svg>")
    return "".join(fragments)


def _display_number(value, *, percentage=False, decimals=3):
    number = _number(value)
    if number is None:
        return "-"
    return f"{100.0 * number:.1f}%" if percentage else f"{number:.{decimals}f}"


def _relative_artifact_href(report_path, source_path, artifact):
    target = Path(source_path) / artifact
    return Path(os.path.relpath(target, Path(report_path).parent)).as_posix()


def write_analysis_overview(
    path,
    run_analysis,
    condition_analysis,
    phase_scorecard,
    outliers=(),
):
    """Write the primary analysis-first landing page for runs and batches."""
    path = Path(path)
    total_conditions = len(condition_analysis)
    successful = sum(row["task_success"] for row in condition_analysis)
    failed = total_conditions - successful
    run_rows = []
    for row in run_analysis:
        status = "all_successful" if not row["failed_condition_count"] else "all_failed" if not row["successful_condition_count"] else "mixed"
        run_href = _relative_artifact_href(path, row["source_path"], "analysis_overview.html")
        evidence_href = _relative_artifact_href(path, row["source_path"], "conversation_transcripts.txt")
        values = (
            f'<a href="{html.escape(run_href)}">{html.escape(row["source_run"])}</a>',
            row["agent_b_llm_size"], row["agent_a_type"], row["agent_b_model"],
            row["configured_tts_engines"], row["configured_asr_engines"],
            row["condition_count"], _display_number(row["task_success_rate"], percentage=True),
            _display_number(row["audio_task_success_rate"], percentage=True),
            _display_number(row["text_task_success_rate"], percentage=True),
            _display_number(row["paired_task_success_delta"], percentage=True),
            _display_number(row["mean_route_validity"], percentage=True),
            _display_number(row["mean_constraint_satisfaction"], percentage=True),
            _display_number(row["mean_audio_word_error_rate"], percentage=True),
            _display_number(row["mean_repair_success_rate"], percentage=True),
            _display_number(row["mean_turn_count"], decimals=1),
            _display_number(row["mean_runtime_sec"], decimals=1),
            row["failure_phase_counts"] or "-",
            f'<a href="{html.escape(evidence_href)}">transcript</a>',
        )
        run_rows.append(
            f'<tr class="{status}">' + "".join(
                f"<td>{value if str(value).startswith('<a ') else html.escape(str(value))}</td>"
                for value in values
            ) + "</tr>"
        )

    phase_orders = {
        phase: min(
            int(row.get("phase_order", 999))
            for row in phase_scorecard if row["phase"] == phase
        )
        for phase in {row["phase"] for row in phase_scorecard}
    }
    phases = sorted(phase_orders, key=lambda phase: (phase_orders[phase], phase))
    score_lookup = {
        (row["source_run"], row["phase"]): row for row in phase_scorecard
    }
    phase_rows = []
    for run in run_analysis:
        cells = [f'<th>{html.escape(run["source_run"])}</th>']
        for phase in phases:
            score = score_lookup.get((run["source_run"], phase))
            if not score or score["phase_score"] is None:
                cells.append('<td class="unavailable">-</td>')
                continue
            color = _metric_cell_color(
                score["phase_score"],
                {"range_min": 0.0, "range_max": 1.0, "higher_is_better": True},
            )
            title = (
                f'{score["declared_range_metric_count"]} declared-range metrics; '
                f'{score["available_metric_count"]}/{score["registered_metric_count"]} metrics available; '
                f'{score["observation_count"]} observations'
            )
            cells.append(
                f'<td style="background:{color}" title="{html.escape(title)}">'
                f'<strong>{100.0 * score["phase_score"]:.1f}%</strong><br>'
                f'<small>{score["available_metric_count"]}/{score["registered_metric_count"]} available</small></td>'
            )
        phase_rows.append("<tr>" + "".join(cells) + "</tr>")

    condition_rows = []
    for row in condition_analysis:
        transcript_href = _relative_artifact_href(path, row["source_path"], "conversation_transcripts.txt")
        status = "success" if row["task_success"] else "failure"
        values = (
            status,
            row["source_run"],
            f'<a href="{html.escape(transcript_href)}">{html.escape(row["condition_id"])}</a>',
            row["scenario_key"], row["persona_key"], row["run_type"], row["speech_pattern_key"],
            row["configured_tts_engine"], row["configured_asr_engine"], row["asr_search_width"],
            row["experiment_seed"], row["route_valid"],
            _display_number(row["constraint_satisfaction"], percentage=True),
            _display_number(row["word_error_rate"], percentage=True),
            _display_number(row["entity_error_rate"], percentage=True),
            _display_number(row["repair_success_rate"], percentage=True),
            _display_number(row["grounded_proposal_score"], percentage=True),
            _display_number(row["turn_count"], decimals=0),
            _display_number(row["runtime_sec"], decimals=1),
            row["failure_phase"] or row["pipeline_failure_type"] or "-",
        )
        condition_rows.append(
            f'<tr class="condition-row {status}" data-outcome="{status}" data-run-type="{html.escape(str(row["run_type"]))}">'
            + "".join(
                f"<td>{value if str(value).startswith('<a ') else html.escape(str(value))}</td>"
                for value in values
            ) + "</tr>"
        )

    outlier_rows = "".join(
        f'<tr class="{html.escape(str(row.get("outcome_alignment", "")))}">'
        f'<td>{html.escape(str(row.get("source_run", "")))}</td>'
        f'<td>{html.escape(str(row.get("condition_id", "")))}</td>'
        f'<td>{html.escape(str(row.get("phase", "")))}</td>'
        f'<td>{html.escape(str(row.get("metric_label", row.get("metric_key", ""))))}</td>'
        f'<td>{_display_number(row.get("value"), decimals=4)}</td>'
        f'<td>{_display_number(row.get("modified_z_score"), decimals=2)}</td>'
        f'<td>{html.escape(str(row.get("outcome_alignment", "")))}</td></tr>'
        for row in outliers
    ) or '<tr><td colspan="7">No robust pre-outcome outliers detected.</td></tr>'

    phase_headers = "".join(f"<th>{html.escape(phase)}</th>" for phase in phases)
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Experiment analysis overview</title>
<style>
:root{{--ink:#202a35;--muted:#5d6975;--line:#cbd3dc;--head:#244b5a;--head2:#376879;--bg:#f3f5f7;--ok:#dff1e7;--warn:#fff0c9;--bad:#f7dada}}
*{{box-sizing:border-box}}body{{font:14px system-ui,sans-serif;color:var(--ink);background:var(--bg);margin:0}}main{{max-width:1800px;margin:auto;padding:18px}}
h1{{margin:0 0 4px;font-size:24px}}h2{{font-size:17px;margin:0 0 8px}}p{{margin:5px 0 10px}}nav{{position:sticky;top:0;z-index:9;background:var(--head);padding:8px 12px;margin:0 -18px 14px}}nav a{{color:white;margin-right:18px;text-decoration:none;font-weight:650}}
.summary{{display:grid;grid-template-columns:repeat(4,minmax(120px,1fr));gap:1px;background:var(--line);border:1px solid var(--line);margin:12px 0}}.summary div{{background:white;padding:10px}}.summary strong{{display:block;font-size:20px}}.summary small{{color:var(--muted)}}
section{{background:white;border:1px solid var(--line);margin:12px 0;padding:12px;overflow:auto}}table{{border-collapse:separate;border-spacing:0;width:100%}}th,td{{border-right:1px solid #dce2e8;border-bottom:1px solid #dce2e8;padding:6px 7px;text-align:left;white-space:nowrap}}thead th{{position:sticky;top:0;background:var(--head2);color:white;z-index:2}}tbody th{{background:#eaf1ef}}.all_successful,.success{{background:var(--ok)}}.mixed{{background:var(--warn)}}.all_failed,.failure{{background:var(--bad)}}.unavailable{{background:#eef1f4;color:var(--muted)}}.outcome_contradicting,.direction_unknown{{background:#e8e5f2}}
.controls{{display:flex;gap:8px;align-items:center;margin:8px 0}}input{{min-width:300px;padding:7px;border:1px solid var(--line)}}button{{padding:7px 10px;border:1px solid #9eabb7;background:white;cursor:pointer}}button:hover{{background:#eaf1ef}}code{{font-size:12px}}small{{color:#44515d}}
@media(max-width:800px){{.summary{{grid-template-columns:repeat(2,1fr)}}main{{padding:10px}}nav{{margin:0 -10px 10px}}}}
</style></head><body><main>
<nav><a href="#runs">Runs</a><a href="#phases">Phases</a><a href="#conditions">Conditions</a><a href="#outliers">Outliers</a></nav>
<h1>Experiment analysis overview</h1><p>Analysis-first index over canonical result tables. Raw metrics and protocol evidence remain authoritative.</p>
<div class="summary"><div><strong>{len(run_analysis)}</strong><small>finalized runs</small></div><div><strong>{total_conditions}</strong><small>conditions</small></div><div><strong>{successful}</strong><small>successful</small></div><div><strong>{failed}</strong><small>failed</small></div></div>
<section id="runs"><h2>Run comparison</h2><p>Matched audio/text delta is audio success minus its paired text control. A dash means the comparison is unavailable.</p>
<table><thead><tr><th>Run</th><th>Size</th><th>Agent A</th><th>Agent B</th><th>TTS</th><th>ASR</th><th>N</th><th>Success</th><th>Audio</th><th>Text</th><th>Audio delta</th><th>Route valid</th><th>Constraints</th><th>Audio WER</th><th>Repair</th><th>Turns</th><th>Runtime s</th><th>Failure phases</th><th>Evidence</th></tr></thead><tbody>{''.join(run_rows)}</tbody></table></section>
<section id="phases"><h2>Phase scorecard</h2><p>Equal-weighted descriptive mean over available metrics with declared ranges. Coverage is shown in every cell; compare component metrics before drawing conclusions.</p>
<table><thead><tr><th>Run</th>{phase_headers}</tr></thead><tbody>{''.join(phase_rows)}</tbody></table></section>
<section id="conditions"><h2>Condition explorer</h2><div class="controls"><input id="condition-search" type="search" placeholder="Search run, condition, scenario, persona, provider..."><button type="button" data-filter="all">All</button><button type="button" data-filter="failure">Failures</button><button type="button" data-filter="audio_variant">Audio</button><button type="button" data-filter="text_only">Text</button></div>
<table id="condition-table"><thead><tr><th>Outcome</th><th>Run</th><th>Condition</th><th>Scenario</th><th>Persona</th><th>Type</th><th>Speech</th><th>TTS</th><th>ASR</th><th>Beam</th><th>Seed</th><th>Route valid</th><th>Constraints</th><th>WER</th><th>Entity error</th><th>Repair</th><th>Grounding</th><th>Turns</th><th>Runtime s</th><th>Failure phase</th></tr></thead><tbody>{''.join(condition_rows)}</tbody></table></section>
<section id="outliers"><h2>Robust metric outliers</h2><p>Modified z-scores are descriptive associations and are not validated causal indicators.</p><table><thead><tr><th>Run</th><th>Condition</th><th>Phase</th><th>Metric</th><th>Value</th><th>Modified z</th><th>Alignment</th></tr></thead><tbody>{outlier_rows}</tbody></table></section>
<script>
const rows=[...document.querySelectorAll('.condition-row')];const search=document.getElementById('condition-search');let filter='all';
function apply(){{const q=search.value.toLowerCase();for(const row of rows){{const matchesText=!q||row.textContent.toLowerCase().includes(q);const matchesFilter=filter==='all'||row.dataset.outcome===filter||row.dataset.runType===filter;row.hidden=!(matchesText&&matchesFilter);}}}}
search.addEventListener('input',apply);document.querySelectorAll('[data-filter]').forEach(button=>button.addEventListener('click',()=>{{filter=button.dataset.filter;apply();}}));
</script></main></body></html>"""
    path.write_text(document, encoding="utf-8")


def write_run_analysis_outputs(run_directory):
    """Regenerate compact analysis artifacts for one finalized result folder."""
    run_directory = Path(run_directory).resolve()
    conditions, metrics = load_comparison_data([run_directory])
    condition_rows = build_condition_analysis_rows(conditions, metrics)
    run_rows = build_run_analysis_rows(condition_rows)
    phase_rows = build_run_phase_scorecard(metrics)
    performance_rows = build_performance_band_summary(condition_rows)
    outliers = identify_metric_outliers(metrics, conditions)
    paths = {
        "condition_analysis": run_directory / "condition_analysis.csv",
        "run_analysis": run_directory / "run_analysis.csv",
        "run_phase_scorecard": run_directory / "run_phase_scorecard.csv",
        "analysis_overview": run_directory / "analysis_overview.html",
        "performance_band_summary": run_directory / "performance_band_summary.csv",
    }
    _write_csv(paths["condition_analysis"], condition_rows)
    _write_csv(paths["run_analysis"], run_rows)
    _write_csv(paths["run_phase_scorecard"], phase_rows)
    _write_csv(paths["performance_band_summary"], performance_rows)
    write_analysis_overview(
        paths["analysis_overview"], run_rows, condition_rows, phase_rows, outliers
    )
    return paths


def write_html_report(
    path, run_directories, conditions, summaries, deltas,
    run_outcomes, outliers, metric_indicators,
):
    """Write a dependency-free HTML report with inline SVG comparisons."""
    chart_rows = _selected_chart_rows(summaries)
    by_metric = defaultdict(list)
    for row in chart_rows:
        by_metric[row["metric_key"]].append(row)
    charts = "".join(
        f"<section><h3>{html.escape(metric)}</h3>{_bar_svg(rows)}</section>"
        for metric, rows in by_metric.items()
    ) or "<p>No available numeric audio metrics were found.</p>"
    table_rows = "".join(
        f'<tr class="{html.escape(row["task_outcome_status"])}">'
        + "".join(f"<td>{html.escape(str(value))}</td>" for value in (
            row["source_run"], row["task_outcome_status"], row["condition_count"],
            row["successful_condition_count"], row["failed_condition_count"],
            f'{100.0 * row["task_success_rate"]:.1f}%',
            row["failure_aligned_metric_outliers"], row["success_aligned_metric_outliers"],
        )) + "</tr>"
        for row in run_outcomes
    )
    outlier_rows = "".join(
        f'<tr class="{html.escape(row["outcome_alignment"])}">'
        + "".join(f"<td>{html.escape(str(value))}</td>" for value in (
            row["source_run"], row["condition_id"], "success" if row["task_success"] else "failure",
            row["phase"], row["metric_label"], f'{row["value"]:.4g}',
            f'{row["reference_median"]:.4g}', f'{row["modified_z_score"]:.2f}',
            row["outlier_direction"], row["outcome_alignment"],
        )) + "</tr>"
        for row in outliers
    ) or '<tr><td colspan="10">No robust pre-outcome metric outliers detected.</td></tr>'
    indicator_rows = "".join(
        f'<tr class="{"success_aligned" if row["outcome_alignment_rate"] >= 0.75 else "mixed" if row["outcome_alignment_rate"] >= 0.5 else "failure_aligned"}">'
        + "".join(f"<td>{html.escape(str(value))}</td>" for value in (
            row["phase"], row["metric_label"], row["outlier_count"],
            row["failure_aligned_count"], row["success_aligned_count"],
            row["contradicting_count"], f'{100.0 * row["outcome_alignment_rate"]:.1f}%',
        )) + "</tr>"
        for row in metric_indicators
    ) or '<tr><td colspan="7">No metric indicator summary is available.</td></tr>'
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CoopNavigationSDS batch comparison</title>
<style>body{{font:15px system-ui,sans-serif;color:#202A35;background:#F3F5F7;margin:0}}main{{max-width:1500px;margin:auto;padding:24px}}section{{background:white;border:1px solid #CBD3DC;border-radius:6px;padding:16px;margin:12px 0;overflow:auto}}h1,h2,h3{{margin:0 0 12px}}table{{border-collapse:collapse;width:100%}}th,td{{border-bottom:1px solid #DCE2E8;padding:7px;text-align:left;white-space:nowrap}}th{{background:#EAF1EF}}.all_successful,.success_aligned{{background:#DFF1E7}}.mixed{{background:#FFF0C9}}.all_failed,.failure_aligned{{background:#F7DADA}}.outcome_contradicting,.direction_unknown{{background:#E8E5F2}}.legend span{{display:inline-block;padding:4px 8px;margin-right:6px;border:1px solid #CBD3DC}}svg{{width:100%;max-height:420px}}</style></head>
<body><main><h1>Batch comparison</h1><p>{len(run_directories)} runs, {len(conditions)} conditions, {len(summaries)} metric aggregates, {len(deltas)} pairwise deltas.</p>
<p class="legend"><span class="all_successful">All successful</span><span class="mixed">Mixed</span><span class="all_failed">All failed</span><span class="outcome_contradicting">Outlier contradicts outcome</span></p>
<section><h2>Run task outcomes</h2><table><thead><tr><th>Run</th><th>Status</th><th>Conditions</th><th>Task success count</th><th>Task failure count</th><th>Success rate</th><th>Failure-aligned outliers</th><th>Success-aligned outliers</th></tr></thead><tbody>{table_rows}</tbody></table></section>
<section><h2>Metric outliers and outcome alignment</h2><p>Modified z-score uses the median absolute deviation within metric, phase, and run type. Only pre-outcome phases with at least {OUTLIER_MINIMUM_SAMPLES} observations are eligible. Color denotes descriptive alignment, not causation or validated prediction.</p><table><thead><tr><th>Run</th><th>Condition</th><th>Task outcome</th><th>Phase</th><th>Metric</th><th>Value</th><th>Median</th><th>Modified z</th><th>Direction</th><th>Alignment</th></tr></thead><tbody>{outlier_rows}</tbody></table></section>
<section><h2>Metrics indicating observed outcomes</h2><p>Counts summarize how often each metric's robust outliers aligned with failure or success. Green means at least 75% alignment, amber at least 50%, and red below 50%; this is descriptive and must be validated on held-out runs.</p><table><thead><tr><th>Phase</th><th>Metric</th><th>Outliers</th><th>Failure-aligned</th><th>Success-aligned</th><th>Contradicting</th><th>Alignment rate</th></tr></thead><tbody>{indicator_rows}</tbody></table></section>
<h2>Selected audio metrics</h2>{charts}
<section><h2>Analysis files</h2><p>Use <code>metric_summary.csv</code> for aggregate plots and <code>metric_deltas.csv</code> for provider contrasts. Raw joined evidence remains in the combined CSV files.</p></section>
</main></body></html>"""
    Path(path).write_text(document, encoding="utf-8")


def compare_runs(inputs, output_directory):
    """Build the canonical cross-run tables and one phase-wise HTML matrix."""
    run_directories = discover_run_directories(inputs)
    if not run_directories:
        raise ValueError("No standard run folders containing run_summary.json, conditions.jsonl, and metrics_long.csv were found.")
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    conditions, metrics = load_comparison_data(run_directories)
    summaries = summarize_metrics(metrics)
    deltas = calculate_run_deltas(summaries)
    outliers = identify_metric_outliers(metrics, conditions)
    metric_indicators = summarize_metric_indicators(outliers)
    condition_analysis = build_condition_analysis_rows(conditions, metrics)
    performance_bands = build_performance_band_summary(condition_analysis)
    run_metric_matrix, run_metric_specifications, run_metric_outliers = build_run_phase_metric_matrix(
        conditions,
        metrics,
        outliers,
    )
    paths = {
        "conditions": output / "combined_conditions.csv",
        "metrics": output / "combined_metrics_long.csv",
        "summary": output / "metric_summary.csv",
        "deltas": output / "metric_deltas.csv",
        "outliers": output / "metric_outliers.csv",
        "metric_indicators": output / "metric_indicator_summary.csv",
        "run_metric_matrix": output / "run_phase_metric_matrix.csv",
        "run_metric_matrix_report": output / "run_phase_metric_matrix.html",
        "condition_analysis": output / "condition_analysis.csv",
        "performance_band_summary": output / "performance_band_summary.csv",
    }
    for obsolete in (
        "analysis_overview.html", "comparison_report.html",
        "phase_metric_overview.html", "phase_metric_summary.csv",
        "run_analysis.csv", "run_outcomes.csv", "run_phase_scorecard.csv",
    ):
        (output / obsolete).unlink(missing_ok=True)
    _write_csv(paths["conditions"], conditions)
    _write_csv(paths["metrics"], metrics)
    _write_csv(paths["summary"], summaries)
    _write_csv(paths["deltas"], deltas)
    _write_csv(paths["outliers"], outliers)
    _write_csv(paths["metric_indicators"], metric_indicators)
    _write_csv(paths["run_metric_matrix"], run_metric_matrix)
    _write_csv(paths["condition_analysis"], condition_analysis)
    _write_csv(paths["performance_band_summary"], performance_bands)
    write_run_phase_metric_matrix_html(
        paths["run_metric_matrix_report"],
        run_metric_matrix,
        run_metric_specifications,
        run_metric_outliers,
    )
    for run_directory in run_directories:
        write_run_analysis_outputs(run_directory)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare standard CoopNavigationSDS result folders.")
    parser.add_argument("runs", nargs="+", help="Run folders or result roots to discover recursively.")
    parser.add_argument("--output", required=True, help="Directory for combined CSV files and the HTML report.")
    args = parser.parse_args(argv)
    paths = compare_runs(args.runs, args.output)
    for label, path in paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
