#!/usr/bin/env python3
"""Create completed-dialogue metric/outcome analysis tables.

The script intentionally excludes provider/runtime failures and invalid
conditions. Those belong in execution coverage reports. The output here is for
research questions about which metrics indicate successful, semi-successful, or
unsuccessful completed dialogues.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_AGENT_A = "userlm"
DEFAULT_MODELS = (
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "microsoft/Phi-3-mini-4k-instruct",
    "Qwen/Qwen2.5-7B-Instruct",
)
COMPLETED_STATUSES = {"completed", "complete", "success"}
TRUE_VALUES = {"true", "1", "yes", "y"}
DIRECT_OUTCOME_METRICS = {
    "success",
    "route_valid",
    "route_reaches_goal",
    "task_outcome_completion",
    "task_outcome_route_validity",
    "task_outcome_correct_route_selection",
    "task_outcome_constraint_satisfaction",
    "task_outcome_constraint_satisfaction_rate",
    "task_outcome_acceptable_duration_completion",
}
CONSTRUCT_OVERLAP_PATTERNS = (
    "constraint_mention",
    "constraint_satisfaction",
    "active_constraint_compliance",
    "joint_goal_accuracy",
    "guard_intervention",
    "dialogue_success_score",
    "interaction_quality_trajectory",
    "success_confidence_interval",
    "floor_rate",
    "ceiling_rate",
)
PHASE_PREFIXES = (
    ("dialogue_management_", "dialogue_management"),
    ("dialogue_state_", "dialogue_state"),
    ("whole_dialogue_", "whole_dialogue"),
    ("task_outcome_", "task_outcome"),
    ("agent_b_", "agent_b_response"),
    ("agent_a_", "agent_a_evaluation"),
    ("asr_", "automatic_speech_recognition"),
    ("tts_", "text_to_speech"),
    ("nlu_", "natural_language_understanding"),
    ("nlg_", "natural_language_generation"),
    ("metric_validity_", "metric_validity"),
)


def truth(value) -> bool:
    return str(value).strip().lower() in TRUE_VALUES


def number(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "null"}:
        return None
    try:
        result = float(text)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def mean(values):
    return sum(values) / len(values) if values else None


def pearson(xs, ys):
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    x_mean = mean(xs)
    y_mean = mean(ys)
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    if x_var <= 0 or y_var <= 0:
        return None
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / math.sqrt(x_var * y_var)


def phase_for_metric(metric_name: str) -> str:
    for prefix, phase in PHASE_PREFIXES:
        if metric_name.startswith(prefix):
            return phase
    if metric_name in {"duration_score", "quality_score", "automatic_eval_score"}:
        return "whole_dialogue"
    if metric_name in {"candidate_route_count", "route_revision_count", "best_candidate_turn"}:
        return "dialogue_management"
    if metric_name in {"station_mentions", "task_focus_score", "clarification_rate"}:
        return "dialogue_state"
    if metric_name.endswith("_latency_sec") or "elapsed_sec" in metric_name:
        return "audio_turn_taking"
    return "other"


def is_construct_overlap_metric(metric_name: str) -> bool:
    """Return true when a metric is too close to the target outcome.

    These metrics are still useful as outcome-confirming evidence. They should
    not be presented as independent predictors of task success.
    """
    if metric_name in DIRECT_OUTCOME_METRICS or metric_name.startswith("task_outcome_"):
        return True
    if metric_name in {"automatic_eval_score", "quality_score"}:
        return True
    if metric_name.startswith("metric_validity_"):
        return True
    return any(pattern in metric_name for pattern in CONSTRUCT_OVERLAP_PATTERNS)


def read_first_jsonl(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                return json.loads(line)
    return {}


def read_first_csv(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return next(csv.DictReader(handle), {})


def classify(condition: dict, metrics: dict) -> str:
    task_success = truth(condition.get("task_success")) or truth(metrics.get("success"))
    route_valid = truth(condition.get("route_valid")) or truth(metrics.get("route_valid"))
    route_reaches_goal = truth(condition.get("route_reaches_goal")) or truth(metrics.get("route_reaches_goal"))
    if task_success:
        return "successful"
    if route_valid or route_reaches_goal:
        return "semi_successful"
    return "unsuccessful_dialogue"


def discover_completed_runs(results_dir: Path, agent_a_type: str, model_names: set[str]) -> list[dict]:
    rows = []
    for condition_path in sorted(results_dir.rglob("conditions.jsonl")):
        try:
            condition = read_first_jsonl(condition_path)
        except (OSError, json.JSONDecodeError):
            continue
        if str(condition.get("agent_a_type", "")).lower() != agent_a_type.lower():
            continue
        if model_names and str(condition.get("agent_b_model", "")) not in model_names:
            continue
        if str(condition.get("execution_status", "")).lower() not in COMPLETED_STATUSES:
            continue
        metrics_path = condition_path.parent / "metrics_wide.csv"
        if not metrics_path.is_file():
            continue
        try:
            metrics = read_first_csv(metrics_path)
        except OSError:
            continue
        outcome_band = classify(condition, metrics)
        rows.append(
            {
                "source_path": str(condition_path.parent),
                "result_run_id": condition.get("result_run_id") or metrics.get("result_run_id") or condition_path.parent.name,
                "condition_id": condition.get("condition_id") or metrics.get("condition_id"),
                "agent_a_type": condition.get("agent_a_type"),
                "agent_b_model": condition.get("agent_b_model") or metrics.get("model_name"),
                "agent_b_size": condition.get("agent_b_llm_size") or metrics.get("factor_agent_b_llm_size"),
                "scenario_key": condition.get("scenario_key") or metrics.get("scenario_key"),
                "persona_key": condition.get("persona_key") or metrics.get("persona_key"),
                "speech_pattern_key": condition.get("speech_pattern_key") or metrics.get("speech_pattern_key"),
                "tts_engine": condition.get("tts_engine") or condition.get("configured_tts_engine") or metrics.get("factor_tts_engine"),
                "asr_engine": condition.get("asr_engine") or condition.get("configured_asr_engine") or metrics.get("factor_asr_engine"),
                "asr_search_width": condition.get("asr_search_width") or metrics.get("factor_asr_beam_size"),
                "task_success": truth(condition.get("task_success")) or truth(metrics.get("success")),
                "route_valid": truth(condition.get("route_valid")) or truth(metrics.get("route_valid")),
                "route_reaches_goal": truth(condition.get("route_reaches_goal")) or truth(metrics.get("route_reaches_goal")),
                "outcome_band": outcome_band,
                "outcome_rank": {"unsuccessful_dialogue": 0.0, "semi_successful": 0.5, "successful": 1.0}[outcome_band],
                "metrics": metrics,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_outcome_rows(runs: list[dict]) -> list[dict]:
    fields = (
        "source_path",
        "result_run_id",
        "condition_id",
        "agent_a_type",
        "agent_b_model",
        "agent_b_size",
        "scenario_key",
        "persona_key",
        "speech_pattern_key",
        "tts_engine",
        "asr_engine",
        "asr_search_width",
        "task_success",
        "route_valid",
        "route_reaches_goal",
        "outcome_band",
        "outcome_rank",
    )
    return [{field: run.get(field, "") for field in fields} for run in runs]


def model_summary_rows(runs: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for run in runs:
        grouped[(run["agent_b_size"], run["agent_b_model"])].append(run)
    rows = []
    for (size, model), items in sorted(grouped.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))):
        counts = Counter(run["outcome_band"] for run in items)
        total = len(items)
        rows.append(
            {
                "agent_b_size": size,
                "agent_b_model": model,
                "completed_runs": total,
                "successful": counts["successful"],
                "semi_successful": counts["semi_successful"],
                "unsuccessful_dialogue": counts["unsuccessful_dialogue"],
                "success_rate": counts["successful"] / total if total else None,
                "route_or_better_rate": (counts["successful"] + counts["semi_successful"]) / total if total else None,
            }
        )
    return rows


def numeric_metric_names(runs: list[dict]) -> list[str]:
    names = set()
    for run in runs:
        for name, value in run["metrics"].items():
            if number(value) is not None:
                names.add(name)
    blocked_prefixes = ("factor_",)
    blocked = {"condition_id", "result_run_id", "iteration", "repetition"}
    return sorted(name for name in names if name not in blocked and not name.startswith(blocked_prefixes))


def metric_mean_rows(runs: list[dict], metric_names: list[str]) -> list[dict]:
    rows = []
    for metric in metric_names:
        grouped = defaultdict(list)
        for run in runs:
            value = number(run["metrics"].get(metric))
            if value is not None:
                grouped[run["outcome_band"]].append(value)
        if not grouped:
            continue
        all_values = [value for values in grouped.values() for value in values]
        rows.append(
            {
                "phase": phase_for_metric(metric),
                "metric": metric,
                "is_direct_outcome_metric": metric in DIRECT_OUTCOME_METRICS or metric.startswith("task_outcome_"),
                "is_construct_overlap_metric": is_construct_overlap_metric(metric),
                "n_successful": len(grouped["successful"]),
                "mean_successful": mean(grouped["successful"]),
                "n_semi_successful": len(grouped["semi_successful"]),
                "mean_semi_successful": mean(grouped["semi_successful"]),
                "n_unsuccessful_dialogue": len(grouped["unsuccessful_dialogue"]),
                "mean_unsuccessful_dialogue": mean(grouped["unsuccessful_dialogue"]),
                "min_value": min(all_values),
                "max_value": max(all_values),
                "overall_mean": mean(all_values),
                "overall_median": statistics.median(all_values),
            }
        )
    return rows


def metric_correlation_rows(runs: list[dict], metric_names: list[str], minimum_samples: int) -> list[dict]:
    rows = []
    for metric in metric_names:
        pairs = []
        for run in runs:
            value = number(run["metrics"].get(metric))
            if value is None:
                continue
            pairs.append(
                {
                    "metric_value": value,
                    "task_success": 1.0 if run["outcome_band"] == "successful" else 0.0,
                    "route_or_better": 0.0 if run["outcome_band"] == "unsuccessful_dialogue" else 1.0,
                    "outcome_rank": run["outcome_rank"],
                }
            )
        if len(pairs) < minimum_samples:
            continue
        metric_values = [row["metric_value"] for row in pairs]
        if len(set(metric_values)) < 2:
            continue
        corr_success = pearson(metric_values, [row["task_success"] for row in pairs])
        corr_route = pearson(metric_values, [row["route_or_better"] for row in pairs])
        corr_rank = pearson(metric_values, [row["outcome_rank"] for row in pairs])
        rows.append(
            {
                "phase": phase_for_metric(metric),
                "metric": metric,
                "is_direct_outcome_metric": metric in DIRECT_OUTCOME_METRICS or metric.startswith("task_outcome_"),
                "is_construct_overlap_metric": is_construct_overlap_metric(metric),
                "n": len(pairs),
                "correlation_with_task_success": corr_success,
                "correlation_with_route_or_better": corr_route,
                "correlation_with_outcome_rank": corr_rank,
                "mean": mean(metric_values),
                "min": min(metric_values),
                "max": max(metric_values),
                "interpretation": interpretation(corr_success, corr_route, corr_rank),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            -(abs(row["correlation_with_outcome_rank"]) if row["correlation_with_outcome_rank"] is not None else 0.0),
            row["phase"],
            row["metric"],
        ),
    )


def interpretation(corr_success, corr_route, corr_rank) -> str:
    strongest = max(
        (value for value in (corr_success, corr_route, corr_rank) if value is not None),
        key=abs,
        default=0.0,
    )
    if abs(strongest) >= 0.75:
        strength = "strong"
    elif abs(strongest) >= 0.4:
        strength = "moderate"
    elif abs(strongest) >= 0.2:
        strength = "weak"
    else:
        strength = "minimal"
    direction = "positive" if strongest >= 0 else "negative"
    return f"{strength} {direction} descriptive association; not causal"


def write_markdown_summary(path: Path, runs: list[dict], model_rows: list[dict], mean_rows: list[dict], corr_rows: list[dict]) -> None:
    counts = Counter(run["outcome_band"] for run in runs)
    selected_metrics = [
        "automatic_eval_score",
        "quality_score",
        "duration_score",
        "agent_b_grounded_proposal_score",
        "agent_b_actionability_score",
        "nlg_faithfulness",
        "whole_dialogue_goal_progress_auc",
        "whole_dialogue_abandonment_rate",
        "asr_word_error_rate",
        "asr_station_f1",
        "nlu_route_valid_rate",
        "nlu_goal_reached_rate",
        "tts_text_change_rate",
        "candidate_route_count",
        "station_mentions",
        "dialogue_management_repair_success_rate",
    ]
    means_by_metric = {row["metric"]: row for row in mean_rows}
    top_corr = [row for row in corr_rows if not row["is_construct_overlap_metric"]][:12]
    lines = [
        "# Completed Dialogue Metric-Success Analysis",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Scope: completed UserLM-Agent-A runs for the active selected Agent B models. Execution-failed and invalid-condition rows are excluded.",
        "",
        "## Outcome counts",
        "",
        "| Outcome | Runs |",
        "| --- | ---: |",
        f"| successful | {counts['successful']} |",
        f"| semi_successful | {counts['semi_successful']} |",
        f"| unsuccessful_dialogue | {counts['unsuccessful_dialogue']} |",
        f"| total_completed | {len(runs)} |",
        "",
        "## Model summary",
        "",
        "| Agent B size | Agent B model | Completed | Successful | Semi | Unsuccessful | Success rate |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in model_rows:
        lines.append(
            "| {agent_b_size} | {agent_b_model} | {completed_runs} | {successful} | {semi_successful} | "
            "{unsuccessful_dialogue} | {success_rate:.2%} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Metric means by outcome",
            "",
            "| Metric | Successful | Semi-successful | Unsuccessful dialogue |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for metric in selected_metrics:
        row = means_by_metric.get(metric)
        if not row:
            continue
        lines.append(
            f"| `{metric}` | {format_number(row['mean_successful'])} | "
            f"{format_number(row['mean_semi_successful'])} | {format_number(row['mean_unsuccessful_dialogue'])} |"
        )
    lines.extend(
        [
            "",
            "## Strongest diagnostic metric correlations",
            "",
            "| Phase | Metric | n | corr(task success) | corr(outcome rank) | Interpretation |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in top_corr:
        lines.append(
            f"| {row['phase']} | `{row['metric']}` | {row['n']} | "
            f"{format_number(row['correlation_with_task_success'])} | "
            f"{format_number(row['correlation_with_outcome_rank'])} | {row['interpretation']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation rule",
            "",
            "Direct and construct-overlapping metrics confirm the task result. Diagnostic phase metrics are more useful for explaining where successful, semi-successful, and unsuccessful dialogues diverge. Correlations are descriptive associations and must not be presented as causal proof.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def format_number(value) -> str:
    if value is None:
        return ""
    return f"{float(value):.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--agent-a-type", default=DEFAULT_AGENT_A)
    parser.add_argument("--agent-b-model", action="append", dest="agent_b_models", default=None)
    parser.add_argument("--minimum-correlation-samples", type=int, default=10)
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else results_dir / "general"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_names = set(args.agent_b_models or DEFAULT_MODELS)

    runs = discover_completed_runs(results_dir, args.agent_a_type, model_names)
    if not runs:
        raise SystemExit(f"No completed runs found under {results_dir}")

    metric_names = numeric_metric_names(runs)
    outcome_rows = run_outcome_rows(runs)
    model_rows = model_summary_rows(runs)
    mean_rows = metric_mean_rows(runs, metric_names)
    correlation_rows = metric_correlation_rows(runs, metric_names, args.minimum_correlation_samples)

    outputs = {
        "completed_dialogue_run_outcomes": output_dir / "completed_dialogue_run_outcomes.csv",
        "completed_dialogue_outcome_summary": output_dir / "completed_dialogue_outcome_summary.csv",
        "completed_metric_indicator_means": output_dir / "completed_metric_indicator_means.csv",
        "metric_success_correlations_completed": output_dir / "metric_success_correlations_completed.csv",
        "completed_metric_success_summary": output_dir / "completed_metric_success_summary.md",
        "completed_metric_success_manifest": output_dir / "completed_metric_success_manifest.json",
    }

    write_csv(outputs["completed_dialogue_run_outcomes"], outcome_rows)
    write_csv(outputs["completed_dialogue_outcome_summary"], model_rows)
    write_csv(outputs["completed_metric_indicator_means"], mean_rows)
    write_csv(outputs["metric_success_correlations_completed"], correlation_rows)
    write_markdown_summary(outputs["completed_metric_success_summary"], runs, model_rows, mean_rows, correlation_rows)
    outputs["completed_metric_success_manifest"].write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "results_dir": str(results_dir),
                "output_dir": str(output_dir),
                "agent_a_type": args.agent_a_type,
                "agent_b_models": sorted(model_names),
                "completed_run_count": len(runs),
                "outcome_counts": Counter(run["outcome_band"] for run in runs),
                "metric_count": len(metric_names),
                "correlation_metric_count": len(correlation_rows),
                "method": "completed runs only; execution-failed and invalid-condition rows excluded",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    for name, path in outputs.items():
        print(f"{name}: {path}")
    print(f"completed_runs={len(runs)}")
    print(f"outcomes={dict(Counter(run['outcome_band'] for run in runs))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
