"""Combine completed experiment runs into portable comparison tables and charts."""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
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
    """Build a complete comparison dataset and return its artifact paths."""
    run_directories = discover_run_directories(inputs)
    if not run_directories:
        raise ValueError("No standard run folders containing run_summary.json, conditions.jsonl, and metrics_long.csv were found.")
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    conditions, metrics = load_comparison_data(run_directories)
    summaries = summarize_metrics(metrics)
    deltas = calculate_run_deltas(summaries)
    outliers = identify_metric_outliers(metrics, conditions)
    run_outcomes = summarize_run_outcomes(conditions, outliers)
    metric_indicators = summarize_metric_indicators(outliers)
    paths = {
        "conditions": output / "combined_conditions.csv",
        "metrics": output / "combined_metrics_long.csv",
        "summary": output / "metric_summary.csv",
        "deltas": output / "metric_deltas.csv",
        "outliers": output / "metric_outliers.csv",
        "run_outcomes": output / "run_outcomes.csv",
        "metric_indicators": output / "metric_indicator_summary.csv",
        "report": output / "comparison_report.html",
    }
    _write_csv(paths["conditions"], conditions)
    _write_csv(paths["metrics"], metrics)
    _write_csv(paths["summary"], summaries)
    _write_csv(paths["deltas"], deltas)
    _write_csv(paths["outliers"], outliers)
    _write_csv(paths["run_outcomes"], run_outcomes)
    _write_csv(paths["metric_indicators"], metric_indicators)
    write_html_report(
        paths["report"], run_directories, conditions, summaries, deltas,
        run_outcomes, outliers, metric_indicators,
    )
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
