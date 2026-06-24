"""Canonical long-form metric tables for plotting, joins, and audit."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from coop_navigation_sds.EvaluationMetrics.catalog import (
    METRIC_FAMILY_SPECS,
    global_metric_key,
    metric_local_name,
    metric_metadata,
    metric_scale_percentage,
    phase_key,
)


BASE_COLUMNS = (
    "condition_id",
    "pair_id",
    "run_type",
    "test_case_key",
    "persona_key",
    "scenario_key",
    "speech_pattern_key",
    "agent_a_audio_persona",
    "agent_b_audio_persona",
    "model_name",
    "model_param_key",
)


def metric_long_rows(records):
    """Return one normalized row per condition and metric."""
    rows = []
    family_order = {phase_key(family): family["order"] for family in METRIC_FAMILY_SPECS}
    family_titles = {phase_key(family): family["title"] for family in METRIC_FAMILY_SPECS}
    for record in records:
        identifiers = {
            column: getattr(record, column, None)
            for column in BASE_COLUMNS
        }
        factors = {
            f"factor_{key}": value
            for key, value in dict(getattr(record, "experimental_factors", {}) or {}).items()
        }
        for phase, metrics in record.metric_families.items():
            for local_name, value in metrics.items():
                if local_name in {
                    "available",
                    "coverage_rate",
                    "available_metric_count",
                    "configured_metric_count",
                }:
                    continue
                metric_key = global_metric_key(phase, local_name)
                metadata = metric_metadata(metric_key, phase)
                calculation = getattr(record, "metric_calculations", {}).get(metric_key, {})
                bounds = metadata.get("range") or [None, None]
                numeric_value = (
                    float(value)
                    if isinstance(value, (bool, int, float))
                    and not (isinstance(value, float) and not math.isfinite(value))
                    else None
                )
                rows.append({
                    **identifiers,
                    **factors,
                    "phase_order": family_order.get(phase),
                    "phase": phase,
                    "phase_title": family_titles.get(phase, phase.replace("_", " ").title()),
                    "metric_key": metric_key,
                    "metric_name": local_name,
                    "metric_label": metadata.get("meaning", metric_local_name(metric_key)),
                    "value": value,
                    "value_numeric": numeric_value,
                    "value_text": None if numeric_value is not None or value is None else str(value),
                    "available": bool(calculation.get("available", value is not None)),
                    "unit": metadata.get("unit"),
                    "evidence_class": metadata.get("class"),
                    "scope": metadata.get("scope"),
                    "higher_is_better": metadata.get("higher_is_better"),
                    "range_min": bounds[0],
                    "range_max": bounds[1],
                    "normalized_percentage": metric_scale_percentage(metric_key, value),
                    "selection_rationale": metadata.get("selection_rationale"),
                    "formula": calculation.get("formula", metadata.get("calculation")),
                    "operands_json": json.dumps(calculation.get("operands", {}), sort_keys=True),
                    "substitution": calculation.get("substitution"),
                    "unavailable_reason": calculation.get("reason"),
                })
    return rows


def write_metric_long_exports(records, output_dir):
    """Write matching CSV and JSONL long-form metric datasets."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = metric_long_rows(list(records))
    csv_path = output_dir / "metrics_long.csv"
    jsonl_path = output_dir / "metrics_long.jsonl"
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    return {"metric_long_csv": csv_path, "metric_long_jsonl": jsonl_path}
