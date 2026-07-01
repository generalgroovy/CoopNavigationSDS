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


def write_html_report(path, run_directories, conditions, summaries, deltas):
    """Write a dependency-free HTML report with inline SVG comparisons."""
    chart_rows = _selected_chart_rows(summaries)
    by_metric = defaultdict(list)
    for row in chart_rows:
        by_metric[row["metric_key"]].append(row)
    charts = "".join(
        f"<section><h3>{html.escape(metric)}</h3>{_bar_svg(rows)}</section>"
        for metric, rows in by_metric.items()
    ) or "<p>No available numeric audio metrics were found.</p>"
    outcome_rows = defaultdict(lambda: {"conditions": 0, "success": 0, "valid": 0})
    for row in conditions:
        key = (row.get("source_run", ""), row.get("asr_engine", ""), row.get("run_type", ""))
        outcome_rows[key]["conditions"] += 1
        outcome_rows[key]["success"] += int(bool(row.get("task_success")))
        outcome_rows[key]["valid"] += int(bool(row.get("route_valid")))
    table_rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in (*key, data["conditions"], data["success"], data["valid"])) + "</tr>"
        for key, data in sorted(outcome_rows.items())
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CoopNavigationSDS batch comparison</title>
<style>body{{font:15px system-ui,sans-serif;color:#202A35;background:#F3F5F7;margin:0}}main{{max-width:1180px;margin:auto;padding:24px}}section{{background:white;border:1px solid #CBD3DC;border-radius:6px;padding:16px;margin:12px 0}}h1,h2,h3{{margin:0 0 12px}}table{{border-collapse:collapse;width:100%}}th,td{{border-bottom:1px solid #DCE2E8;padding:7px;text-align:left}}th{{background:#EAF1EF}}svg{{width:100%;max-height:420px}}</style></head>
<body><main><h1>Batch comparison</h1><p>{len(run_directories)} runs, {len(conditions)} conditions, {len(summaries)} metric aggregates, {len(deltas)} pairwise deltas.</p>
<section><h2>Outcome counts</h2><table><thead><tr><th>Run</th><th>ASR</th><th>Type</th><th>Conditions</th><th>Task success</th><th>Valid route</th></tr></thead><tbody>{table_rows}</tbody></table></section>
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
    paths = {
        "conditions": output / "combined_conditions.csv",
        "metrics": output / "combined_metrics_long.csv",
        "summary": output / "metric_summary.csv",
        "deltas": output / "metric_deltas.csv",
        "report": output / "comparison_report.html",
    }
    _write_csv(paths["conditions"], conditions)
    _write_csv(paths["metrics"], metrics)
    _write_csv(paths["summary"], summaries)
    _write_csv(paths["deltas"], deltas)
    write_html_report(paths["report"], run_directories, conditions, summaries, deltas)
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
