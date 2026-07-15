"""Generate thesis-grade result interpretation documents.

The script reads active CoopNavigationSDS result folders and writes derived
documentation. It does not modify raw run folders. The generated documents are
intended for thesis writing, not as a replacement for the canonical evidence in
each run directory.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


RELEVANT_AGENT_A = {"userlm", "tinyllama"}
RELEVANT_AGENT_B = {
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": ("small1", "TinyLlama 1.1B", "small"),
    "Qwen/Qwen2.5-0.5B-Instruct": ("small2", "Qwen2.5 0.5B", "small"),
    "Qwen/Qwen2.5-1.5B-Instruct": ("medium1", "Qwen2.5 1.5B", "medium"),
    "microsoft/Phi-3-mini-4k-instruct": ("medium2", "Phi-3 Mini", "medium"),
    "Qwen/Qwen2.5-7B-Instruct": ("large1", "Qwen2.5 7B", "large"),
}
RELEVANT_DIRS = [
    "01-small-tinyllama-1.1b",
    "01-small-qwen2.5-0.5b",
    "02-medium-qwen2.5-1.5b",
    "02-medium-phi3-mini",
    "03-large-qwen2.5-7b",
]

METRIC_PREFIX_PHASES = {
    "audio_": "Audio / turn-taking",
    "asr_": "Automatic speech recognition",
    "nlu_": "Natural language understanding",
    "dialogue_state_": "Dialogue state tracking",
    "dialogue_management_": "Dialogue management",
    "agent_b_": "Agent B response / grounding",
    "agent_a_": "Agent A evaluation",
    "nlg_": "Natural language generation",
    "tts_": "Text-to-speech",
    "task_outcome_": "Task outcome",
    "whole_dialogue_": "Whole dialogue",
    "metric_validity_": "Metric validity",
}

OUTCOME_CONFIRMING = {
    "success",
    "route_valid",
    "route_reaches_goal",
    "automatic_eval_score",
    "duration_score",
    "quality_score",
    "task_outcome_completion",
    "task_outcome_route_validity",
    "task_outcome_constraint_satisfaction",
    "task_outcome_constraint_satisfaction_rate",
    "task_outcome_stage_completion_rate",
    "task_outcome_duration_quality",
    "task_outcome_correct_route_selection",
    "agent_a_closure_correctness",
}

LOW_VALIDITY_IF_CONSTANT = {
    "pipeline_success_rate",
    "tts_success_rate",
    "asr_success_rate",
    "tts_audio_validity_rate",
    "tts_playback_success_rate",
    "agent_b_plugin_execution_success",
}

DIAGNOSTIC_PRIORITY = {
    "asr_word_error_rate",
    "asr_wer",
    "asr_station_f1",
    "asr_entity_error_rate",
    "asr_critical_slot_accuracy",
    "nlu_slot_f1",
    "nlu_goal_reached_rate",
    "nlu_route_valid_rate",
    "nlu_constraint_extraction_f1",
    "dialogue_state_trip_fact_completeness",
    "dialogue_state_missing_trip_slot_rate",
    "dialogue_state_constraint_retention_rate",
    "dialogue_management_repair_success_rate",
    "dialogue_management_stagnation_rate",
    "dialogue_management_clarification_calibration",
    "agent_b_grounded_proposal_score",
    "agent_b_actionability_score",
    "agent_b_active_constraint_compliance",
    "nlg_faithfulness",
    "nlg_executable_utterance_rate",
    "whole_dialogue_goal_progress_auc",
    "whole_dialogue_abandonment_rate",
    "whole_dialogue_task_focus_score",
    "whole_dialogue_failure_localization_score",
}


@dataclass(frozen=True)
class Scope:
    results_root: Path
    docs_dir: Path


def read_jsonl(path: Path) -> Iterable[dict]:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            yield json.loads(line)


def is_archived(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    return any(part.startswith("_archive") for part in parts)


def load_conditions(scope: Scope) -> pd.DataFrame:
    rows: list[dict] = []
    for folder in RELEVANT_DIRS:
        for path in (scope.results_root / folder).rglob("conditions.jsonl"):
            if is_archived(path):
                continue
            run_dir = path.parent
            for row in read_jsonl(path):
                if row.get("agent_a_type") not in RELEVANT_AGENT_A:
                    continue
                if row.get("agent_b_model") not in RELEVANT_AGENT_B:
                    continue
                if row.get("run_type") not in {"text_only", "audio_variant"}:
                    continue
                row = dict(row)
                slot, short_name, size = RELEVANT_AGENT_B[row["agent_b_model"]]
                row["agent_b_slot"] = slot
                row["agent_b_short"] = short_name
                row["agent_b_size"] = size
                row["_run_dir"] = str(run_dir)
                rows.append(row)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["_dedup_key"] = frame.apply(dedup_key, axis=1)
    frame = (
        frame.sort_values("_run_dir")
        .drop_duplicates("_dedup_key", keep="last")
        .reset_index(drop=True)
    )
    frame["completed"] = frame["execution_status"].eq("completed")
    frame["success_bool"] = frame["task_success"].fillna(False).astype(bool)
    frame["route_valid_bool"] = frame["route_valid"].fillna(False).astype(bool)
    frame["semi_success_bool"] = frame["route_valid_bool"] & ~frame["success_bool"]
    frame["unsuccessful_bool"] = frame["completed"] & ~frame["route_valid_bool"] & ~frame["success_bool"]
    return frame


def dedup_key(row: pd.Series) -> str:
    fields = [
        "agent_a_type",
        "run_type",
        "agent_b_model",
        "test_case_key",
        "scenario_key",
        "persona_key",
        "speech_pattern_key",
        "speech_performance_band",
        "agent_a_audio_persona",
        "agent_b_audio_persona",
        "configured_tts_engine",
        "configured_asr_engine",
        "asr_search_width",
        "model_param_key",
        "objective_mode",
        "iteration",
        "network_seed",
        "transfer_tolerance",
        "dialogue_stagnation_limit",
        "experiment_seed",
        "repetition",
        "matrix_family",
    ]
    return "|".join(str(row.get(field)) for field in fields)


def matched_key(row: pd.Series) -> str:
    fields = [
        "test_case_key",
        "scenario_key",
        "persona_key",
        "speech_pattern_key",
        "speech_performance_band",
        "agent_a_audio_persona",
        "agent_b_audio_persona",
        "configured_tts_engine",
        "configured_asr_engine",
        "asr_search_width",
        "model_param_key",
        "objective_mode",
        "iteration",
        "network_seed",
        "transfer_tolerance",
        "dialogue_stagnation_limit",
        "experiment_seed",
        "repetition",
    ]
    return "|".join(str(row.get(field)) for field in fields)


def load_metrics(scope: Scope, conditions: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    wanted_dirs = set(conditions["_run_dir"].dropna().astype(str))
    for run_dir in sorted(wanted_dirs):
        path = Path(run_dir) / "metrics_wide.csv"
        if not path.is_file():
            continue
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        frame["_run_dir"] = run_dir
        rows.append(frame)
    if not rows:
        return pd.DataFrame()
    metrics = pd.concat(rows, ignore_index=True, sort=False)
    metrics = (
        metrics.sort_values("_run_dir")
        .drop_duplicates("_run_dir", keep="last")
        .reset_index(drop=True)
    )
    metrics["success_bool"] = metrics["success"].fillna(False).astype(bool)
    metrics["route_valid_bool"] = metrics["route_valid"].fillna(False).astype(bool)
    metrics["semi_success_bool"] = metrics["route_valid_bool"] & ~metrics["success_bool"]
    return metrics


def pct(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{100 * value:.1f}%"


def fmt(value: float, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.{digits}f}"


def grouped_effects(frame: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    rows = []
    for factor in factors:
        if factor not in frame.columns:
            continue
        grouped = frame.groupby(factor, dropna=False)
        for value, part in grouped:
            completed = int(part["completed"].sum())
            if completed == 0:
                continue
            rows.append(
                {
                    "factor": factor,
                    "level": str(value),
                    "runs": len(part),
                    "completed": completed,
                    "success": int(part["success_bool"].sum()),
                    "semi_success": int(part["semi_success_bool"].sum()),
                    "unsuccessful": int(part["unsuccessful_bool"].sum()),
                    "success_rate_completed": part.loc[part["completed"], "success_bool"].mean(),
                    "route_valid_rate_completed": part.loc[part["completed"], "route_valid_bool"].mean(),
                    "mean_turns_completed": pd.to_numeric(part.loc[part["completed"], "turn_count"], errors="coerce").mean(),
                    "mean_runtime_sec_completed": pd.to_numeric(part.loc[part["completed"], "runtime_sec"], errors="coerce").mean(),
                }
            )
    return pd.DataFrame(rows)


def fully_matched_subset(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame[frame["completed"]].copy()
    if frame.empty:
        return frame
    frame["_matched_key"] = frame.apply(matched_key, axis=1)
    agents = set(frame["agent_a_type"].dropna().unique())
    models = set(frame["agent_b_model"].dropna().unique())
    run_types = {"text_only", "audio_variant"}
    required = {(a, b, run_type) for a in agents for b in models for run_type in run_types}
    selected_keys = []
    for key, part in frame.groupby("_matched_key"):
        cells = set(zip(part["agent_a_type"], part["agent_b_model"], part["run_type"]))
        if required.issubset(cells):
            selected_keys.append(key)
    return frame[frame["_matched_key"].isin(selected_keys)].copy()


def text_audio_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (agent_a, agent_b, run_type), part in frame.groupby(["agent_a_type", "agent_b_short", "run_type"]):
        rows.append(
            {
                "agent_a": agent_a,
                "agent_b": agent_b,
                "run_type": run_type,
                "n": len(part),
                "success": int(part["success_bool"].sum()),
                "semi": int(part["semi_success_bool"].sum()),
                "unsuccessful": int(part["unsuccessful_bool"].sum()),
                "success_rate": part["success_bool"].mean(),
                "mean_turns": pd.to_numeric(part["turn_count"], errors="coerce").mean(),
            }
        )
    base = pd.DataFrame(rows)
    if base.empty:
        return base
    pivot_rows = []
    for (agent_a, agent_b), part in base.groupby(["agent_a", "agent_b"]):
        text = part[part["run_type"].eq("text_only")]
        audio = part[part["run_type"].eq("audio_variant")]
        if text.empty or audio.empty:
            continue
        text_row = text.iloc[0]
        audio_row = audio.iloc[0]
        pivot_rows.append(
            {
                "agent_a": agent_a,
                "agent_b": agent_b,
                "total": int(text_row["n"] + audio_row["n"]),
                "text_success_semi_fail": f"{int(text_row['success'])}/{int(text_row['semi'])}/{int(text_row['unsuccessful'])}",
                "text_success_rate": text_row["success_rate"],
                "audio_success_semi_fail": f"{int(audio_row['success'])}/{int(audio_row['semi'])}/{int(audio_row['unsuccessful'])}",
                "audio_success_rate": audio_row["success_rate"],
                "audio_text_delta_pp": 100 * (audio_row["success_rate"] - text_row["success_rate"]),
                "text_mean_turns": text_row["mean_turns"],
                "audio_mean_turns": audio_row["mean_turns"],
            }
        )
    return pd.DataFrame(pivot_rows)


def metric_phase(metric: str) -> str:
    for prefix, phase in METRIC_PREFIX_PHASES.items():
        if metric.startswith(prefix):
            return phase
    return "Other / task context"


def metric_validity_table(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    ignore = {
        "result_scope",
        "result_run_id",
        "condition_id",
        "test_case_key",
        "persona_key",
        "scenario_key",
        "speech_pattern_key",
        "agent_a_audio_persona",
        "agent_b_audio_persona",
        "model_name",
        "model_param_key",
        "conversation_outcome",
        "stated_constraints",
        "unsatisfied_constraints",
        "route_line_sequence",
        "reference_line_sequence",
        "constraint_line_sequence",
        "allowed_modes",
        "pipeline_mode",
        "pair_id",
        "run_type",
        "_run_dir",
        "_dedup_key",
        "success_bool",
        "route_valid_bool",
        "semi_success_bool",
    }
    numeric = []
    for column in metrics.columns:
        if column in ignore or column.startswith("factor_"):
            continue
        series = pd.to_numeric(metrics[column], errors="coerce")
        if series.notna().sum() > 0:
            numeric.append(column)
    rows = []
    success = metrics["success_bool"].astype(float)
    outcome_rank = metrics["success_bool"].astype(int) + metrics["semi_success_bool"].astype(int) * 0.5
    for column in numeric:
        values = pd.to_numeric(metrics[column], errors="coerce")
        n = int(values.notna().sum())
        missing = 1 - n / len(metrics)
        unique = int(values.nunique(dropna=True))
        if n >= 3 and unique > 1:
            corr_success = values.corr(success)
            corr_outcome = values.corr(outcome_rank)
        else:
            corr_success = float("nan")
            corr_outcome = float("nan")
        ceiling = float((values == values.max()).mean()) if n else float("nan")
        floor = float((values == values.min()).mean()) if n else float("nan")
        if column.startswith("metric_validity_"):
            role = "metric quality indicator"
            validity = "valid for reporting metric coverage/confidence, not for explaining task success directly"
        elif column in OUTCOME_CONFIRMING:
            role = "outcome-confirming"
            validity = "high for outcome reporting; construct-overlapping, so do not use as independent predictor"
        elif column in DIAGNOSTIC_PRIORITY:
            role = "diagnostic phase metric"
            validity = "high diagnostic value if evidence coverage is present; association is descriptive, not causal"
        elif column in LOW_VALIDITY_IF_CONSTANT or unique <= 1:
            role = "execution/constant metric"
            validity = "valid for coverage/preflight reporting, weak for differentiating completed dialogues"
        elif missing > 0.6:
            role = "sparse metric"
            validity = "limited in current result set because required evidence is often unavailable"
        elif abs(corr_success) >= 0.5 if not math.isnan(corr_success) else False:
            role = "associated diagnostic metric"
            validity = "usable as descriptive indicator; verify construct overlap and matched-condition stability"
        else:
            role = "supplementary metric"
            validity = "valid as supporting context; weak standalone indicator in current data"
        rows.append(
            {
                "metric": column,
                "phase": metric_phase(column),
                "role": role,
                "n_available": n,
                "missingness_rate": missing,
                "unique_values": unique,
                "corr_task_success": corr_success,
                "corr_outcome_rank": corr_outcome,
                "ceiling_rate": ceiling,
                "floor_rate": floor,
                "validity_assessment": validity,
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(
        ["role", "phase", "metric"],
        key=lambda s: s.astype(str),
    ).reset_index(drop=True)


def markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    if frame.empty:
        return "_No rows._"
    table = frame[columns].copy()
    if max_rows is not None:
        table = table.head(max_rows)
    headers = list(table.columns)

    def clean(value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, float) and math.isnan(value):
            return ""
        text = str(value)
        return text.replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(clean(row[column]) for column in headers) + " |")
    return "\n".join(lines)


def write_configuration_effect_doc(scope: Scope, conditions: pd.DataFrame, matched: pd.DataFrame, effects: pd.DataFrame, ta_table: pd.DataFrame) -> None:
    lines = [
        "# Thesis Result Configuration Effects",
        "",
        "Scope: active, non-archived, thesis-relevant runs with Agent A in `{UserLM, TinyLlama}` and Agent B in the five selected Transformer backends. Raw run evidence is unchanged; this document is derived from `conditions.jsonl` files.",
        "",
        "## Denominators",
        "",
        f"- Deduplicated active thesis rows: `{len(conditions)}`.",
        f"- Completed active thesis rows: `{int(conditions['completed'].sum())}`.",
        f"- Fully crossed text/audio subset: `{matched['_matched_key'].nunique() if not matched.empty else 0}` condition groups, `{len(matched)}` runs.",
        "- Fully crossed means every condition has both Agent A choices, all five Agent B choices, and both text/audio counterparts.",
        "",
        "## Fully Crossed Text Versus Audio Table",
        "",
    ]
    display = ta_table.copy()
    if not display.empty:
        display["text_success_rate"] = display["text_success_rate"].map(pct)
        display["audio_success_rate"] = display["audio_success_rate"].map(pct)
        display["audio_text_delta_pp"] = display["audio_text_delta_pp"].map(lambda v: f"{v:.1f} pp")
        display["text_mean_turns"] = display["text_mean_turns"].map(lambda v: fmt(v, 1))
        display["audio_mean_turns"] = display["audio_mean_turns"].map(lambda v: fmt(v, 1))
    lines += [
        markdown_table(
            display.sort_values(["agent_a", "agent_b"]) if not display.empty else display,
            [
                "agent_a",
                "agent_b",
                "total",
                "text_success_semi_fail",
                "text_success_rate",
                "audio_success_semi_fail",
                "audio_success_rate",
                "audio_text_delta_pp",
                "text_mean_turns",
                "audio_mean_turns",
            ],
        ),
        "",
        "Safe inference:",
        "",
        "- Text controls are near-ceiling in the fully crossed subset, so they mainly confirm that the route-dialogue policy and task validation can solve these conditions when speech degradation is removed.",
        "- Audio variants reduce task success by roughly 18 to 27 percentage points in the fully crossed subset. This supports a speech-channel degradation claim, not a universal TTS/ASR benchmark claim.",
        "- The matched subset is internally valid but small. Use it for direct Agent A/Agent B comparisons; use broader active rows only for descriptive coverage and association analysis.",
        "",
        "## Factor Effects Across Active Completed Rows",
        "",
        "The table below reports descriptive associations. It is not causal because factors are not always fully balanced.",
        "",
    ]
    effect_display = effects.copy()
    if not effect_display.empty:
        effect_display["success_rate_completed"] = effect_display["success_rate_completed"].map(pct)
        effect_display["route_valid_rate_completed"] = effect_display["route_valid_rate_completed"].map(pct)
        effect_display["mean_turns_completed"] = effect_display["mean_turns_completed"].map(lambda v: fmt(v, 2))
        effect_display["mean_runtime_sec_completed"] = effect_display["mean_runtime_sec_completed"].map(lambda v: fmt(v, 1))
    for factor in [
        "agent_a_type",
        "agent_b_short",
        "scenario_key",
        "persona_key",
        "run_type",
        "speech_performance_band",
        "speech_pattern_key",
        "agent_a_audio_persona",
        "agent_b_audio_persona",
        "configured_asr_engine",
        "asr_search_width",
        "model_param_key",
    ]:
        subset = effect_display[effect_display["factor"].eq(factor)] if not effect_display.empty else pd.DataFrame()
        lines += [
            f"### {factor}",
            "",
            markdown_table(
                subset.sort_values(["success_rate_completed", "completed"], ascending=[False, False]) if not subset.empty else subset,
                [
                    "level",
                    "completed",
                    "success",
                    "semi_success",
                    "unsuccessful",
                    "success_rate_completed",
                    "route_valid_rate_completed",
                    "mean_turns_completed",
                    "mean_runtime_sec_completed",
                ],
            ),
            "",
        ]
    lines += [
        "## Configuration-Level Inference Rules",
        "",
        "- Agent A effects should be reported separately for UserLM and TinyLlama unless the subset is fully matched.",
        "- Agent B effects are safest in the fully crossed subset; broader counts are useful for runtime feasibility and coverage.",
        "- Scenario/persona/audio-persona effects are pressure-test effects: they indicate which conditions stress the pipeline, but they are not independent causal variables unless balanced.",
        "- TTS is not varied in the fully crossed subset (`Piper` dominates), so current results do not support a comparative TTS-framework claim.",
        "- ASR comparisons are descriptive because ASR engine and search width are partly tied to condition generation. Report as association unless a paired ASR-only subset is explicitly selected.",
        "- Text/audio comparison is the strongest currently supported channel-level inference because paired counterparts exist.",
    ]
    (scope.docs_dir / "THESIS_RESULT_CONFIGURATION_EFFECTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_metric_validity_doc(scope: Scope, validity: pd.DataFrame) -> None:
    validity_csv = scope.docs_dir / "THESIS_METRIC_VALIDITY_TABLE.csv"
    validity.to_csv(validity_csv, index=False)
    lines = [
        "# Thesis Metric Validity Assessment",
        "",
        "Scope: metrics found in active thesis-relevant completed result rows. This document assesses what each metric can legitimately support in the thesis. It does not change metric values or raw evidence.",
        "",
        "## Validity Principle",
        "",
        "- A metric is valid only relative to a construct, required logged evidence, and interpretation boundary.",
        "- Outcome-confirming metrics are valid for reporting task outcome but are not independent predictors of that outcome.",
        "- Diagnostic phase metrics are useful when they capture earlier pipeline evidence that can explain later success or failure.",
        "- Sparse, constant, or unavailable metrics should be reported as coverage/execution evidence, not as dialogue-quality evidence.",
        "- Correlations are descriptive associations. They do not prove causality without controlled manipulation or held-out validation.",
        "",
        "## Metric Role Counts",
        "",
    ]
    counts = validity.groupby(["role"], dropna=False).size().reset_index(name="metric_count")
    lines.append(markdown_table(counts, ["role", "metric_count"]))
    lines += [
        "",
        "## Strongest Diagnostic Associations",
        "",
    ]
    diag = validity[validity["role"].isin(["diagnostic phase metric", "associated diagnostic metric"])].copy()
    if not diag.empty:
        diag["abs_corr"] = diag["corr_task_success"].abs()
        diag = diag.sort_values("abs_corr", ascending=False).head(30)
        diag["corr_task_success"] = diag["corr_task_success"].map(lambda v: fmt(v, 3))
        diag["missingness_rate"] = diag["missingness_rate"].map(pct)
        diag["ceiling_rate"] = diag["ceiling_rate"].map(pct)
        lines.append(
            markdown_table(
                diag,
                [
                    "phase",
                    "metric",
                    "role",
                    "n_available",
                    "missingness_rate",
                    "corr_task_success",
                    "ceiling_rate",
                    "validity_assessment",
                ],
            )
        )
    else:
        lines.append("_No diagnostic rows._")
    lines += [
        "",
        "## Phase-Level Interpretation",
        "",
    ]
    phase_summary = (
        validity.groupby(["phase", "role"], dropna=False)
        .size()
        .reset_index(name="metric_count")
        .sort_values(["phase", "role"])
    )
    lines.append(markdown_table(phase_summary, ["phase", "role", "metric_count"]))
    lines += [
        "",
        "## Metrics To Emphasize In The Thesis",
        "",
        "- Task outcome: route validity, task success, constraint satisfaction, duration quality, stage completion.",
        "- ASR: WER, entity error rate, station F1, critical slot accuracy, numeric/constraint preservation where available.",
        "- NLU: slot F1, route-valid rate, goal-reached rate, semantic frame accuracy, origin/destination accuracy.",
        "- Dialogue state/management: trip fact completeness, missing trip slot rate, constraint retention, repair success, stagnation, policy progress.",
        "- Agent B/NLG: grounded proposal score, actionability, active constraint compliance, NLG faithfulness, executable utterance rate.",
        "- Whole dialogue: goal progress AUC, abandonment rate, turn count, repair count, task focus, failure localization.",
        "",
        "## Metrics To Treat Cautiously",
        "",
        "- Generic lexical metrics such as BLEU/ROUGE/METEOR are supplementary because station/line correctness and route executability matter more than surface overlap.",
        "- TTS quality metrics such as NISQA, DNSMOS, PESQ, POLQA, STOI, and SI-SDR are valid only when the required audio/reference evidence exists; unavailable values must not be imputed as bad quality.",
        "- Runtime and latency metrics can indicate feasibility and cost but can be confounded by cluster node load and model-loading conditions.",
        "- Metrics close to the task-success definition should be used to decompose the outcome, not to claim independent prediction.",
        "",
        f"Full per-metric table: `{validity_csv.as_posix()}`.",
    ]
    (scope.docs_dir / "THESIS_METRIC_VALIDITY_ASSESSMENT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_result_inference_doc(scope: Scope, conditions: pd.DataFrame, matched: pd.DataFrame, validity: pd.DataFrame) -> None:
    completed = conditions[conditions["completed"]]
    lines = [
        "# Thesis Result Inference Guide",
        "",
        "This document translates the finalized result snapshot into thesis-safe claims. It separates direct evidence from cautious interpretation.",
        "",
        "## Current Evidence Base",
        "",
        f"- Active thesis-relevant deduplicated rows: `{len(conditions)}`.",
        f"- Completed rows: `{len(completed)}`.",
        f"- Fully crossed matched rows: `{len(matched)}`.",
        f"- Fully crossed condition groups: `{matched['_matched_key'].nunique() if not matched.empty else 0}`.",
        "- Raw evidence remains in per-run folders; this document is derived.",
        "",
        "## Strongest Defensible Claims",
        "",
        "1. Phase-wise automatic evaluation is feasible in this controlled route-dialogue task.",
        "   - The result folders contain turn evidence, task state, route validation, metric inputs, and phase metrics.",
        "   - This supports retrospective analysis without rerunning dialogues.",
        "2. Final task success alone is insufficient.",
        "   - Semi-successful runs and unsuccessful completed runs need route-validity, constraint, repair, ASR/NLU, and grounding evidence to be interpreted.",
        "3. Speech-channel degradation is visible in paired comparisons.",
        "   - In the fully crossed subset, text controls are at ceiling while audio variants lose success rate.",
        "4. Direct model ranking must be cautious.",
        "   - Model comparisons are strongest only in matched subsets.",
        "   - Broader completed counts mix coverage, runtime feasibility, scenario pressure, and backend behavior.",
        "5. Metric-outcome correlations are useful diagnostic evidence, not causal proof.",
        "   - Use the strongest phase metrics to explain likely failure origin and cooperation quality.",
        "",
        "## Model A Interpretation",
        "",
        "- UserLM is the main thesis caller because it is the intended user-simulation model.",
        "- TinyLlama-Agent-A is useful as a control stratum.",
        "- Do not merge UserLM and TinyLlama rows for headline claims unless explicitly using the fully crossed subset.",
        "- If Agent A changes, caller behavior, constraint revelation, repair behavior, and closure behavior also change; those are not merely random variation.",
        "",
        "## Model B Interpretation",
        "",
        "- The five selected Agent B backends cover small, medium, and large Transformer-style local models.",
        "- The current evidence does not justify a simple 'larger is always better' conclusion.",
        "- Qwen2.5 1.5B often has a strong practical profile because it combines high completion volume with high completed-run success.",
        "- Qwen2.5 7B is valuable for size comparison but runtime cost and coverage must be reported beside success.",
        "- TinyLlama and Qwen2.5 0.5B are useful small-model baselines; similar outcomes can indicate task/pipeline limitations rather than model-specific behavior.",
        "",
        "## Scenario, Persona, and Audio Persona Interpretation",
        "",
        "- Scenarios represent controlled task pressure: route complexity, constraint pressure, and dialogue-stage difficulty.",
        "- Personas change interaction behavior and constraint priorities; they are part of the experimental condition, not noise.",
        "- Audio personas and speech performance bands act as channel stressors. They are expected to create a performance range from ceiling to failure.",
        "- Analyze these factors as pressure conditions unless the subset is explicitly balanced for causal comparison.",
        "",
        "## TTS and ASR Interpretation",
        "",
        "- Text/audio pairing is currently the strongest channel-level evidence.",
        "- TTS framework comparison is limited if Piper dominates matched conditions.",
        "- ASR engine and search width can be discussed as descriptive associations; claim comparative ASR effects only for matched ASR-only subsets.",
        "- ASR WER should be interpreted alongside station F1, slot accuracy, and route-state outcomes because route dialogue is entity-sensitive.",
        "",
        "## Thesis Conclusion Skeleton",
        "",
        "The experiment supports the thesis that a controlled spoken route-dialogue framework can produce useful automatic evaluation evidence across SDS phases. The most valid conclusions are about the evaluation method, phase-wise diagnosis, text/audio degradation, and metric usefulness. Claims about model superiority, TTS superiority, or ASR superiority require matched subsets and should be worded cautiously.",
    ]
    (scope.docs_dir / "THESIS_RESULT_INFERENCE_GUIDE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_writing_aid(scope: Scope, conditions: pd.DataFrame, matched: pd.DataFrame, ta_table: pd.DataFrame) -> None:
    path = scope.docs_dir / "THESIS_WRITING_AID.md"
    text = path.read_text(encoding="utf-8")
    marker_start = "<!-- AUTO_RESULT_SYNTHESIS_START -->"
    marker_end = "<!-- AUTO_RESULT_SYNTHESIS_END -->"
    synthesis_lines = [
        marker_start,
        "### Current Result Synthesis For Thesis Writing",
        "",
        "Generated from the finalized active result snapshot. Use as thesis guidance, not as a replacement for the canonical run evidence.",
        "",
        f"- Thesis-relevant deduplicated rows: `{len(conditions)}`.",
        f"- Completed rows: `{int(conditions['completed'].sum())}`.",
        f"- Fully crossed matched condition groups: `{matched['_matched_key'].nunique() if not matched.empty else 0}`.",
        f"- Fully crossed matched runs: `{len(matched)}`.",
        "- Fully crossed means: both Agent A implementations, all five selected Agent B models, and both text/audio counterparts are present for the same non-model condition.",
        "",
        "Safe thesis claims:",
        "",
        "- The strongest method claim is that CoopNavigationSDS preserves enough phase evidence for retrospective automatic SDS evaluation.",
        "- The strongest empirical claim is that the speech channel reduces task success compared with paired text controls in the fully crossed subset.",
        "- Model-backend conclusions must be phrased as matched-condition observations and coverage/runtime trade-offs, not universal model rankings.",
        "- Metrics are valid when tied to their construct and required evidence; outcome metrics confirm task result, while diagnostic phase metrics explain likely failure origin.",
        "",
        "Generated companion documents:",
        "",
        "- `docs/THESIS_RESULT_CONFIGURATION_EFFECTS.md`",
        "- `docs/THESIS_METRIC_VALIDITY_ASSESSMENT.md`",
        "- `docs/THESIS_METRIC_VALIDITY_TABLE.csv`",
        "- `docs/THESIS_RESULT_INFERENCE_GUIDE.md`",
        "",
        marker_end,
    ]
    block = "\n".join(synthesis_lines)
    if marker_start in text and marker_end in text:
        before = text.split(marker_start)[0].rstrip()
        after = text.split(marker_end, 1)[1].lstrip()
        text = before + "\n\n" + block + "\n\n" + after
    else:
        insert_after = "### Current empirical snapshot"
        if insert_after in text:
            text = text.replace(insert_after, block + "\n\n" + insert_after, 1)
        else:
            text = text.rstrip() + "\n\n" + block + "\n"
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--docs-dir", default="docs")
    args = parser.parse_args()
    scope = Scope(Path(args.results_dir), Path(args.docs_dir))
    scope.docs_dir.mkdir(parents=True, exist_ok=True)

    conditions = load_conditions(scope)
    if conditions.empty:
        raise SystemExit("No relevant condition rows found.")
    metrics = load_metrics(scope, conditions[conditions["completed"]])
    matched = fully_matched_subset(conditions)
    effects = grouped_effects(
        conditions,
        [
            "agent_a_type",
            "agent_b_short",
            "agent_b_size",
            "scenario_key",
            "persona_key",
            "run_type",
            "speech_performance_band",
            "speech_pattern_key",
            "agent_a_audio_persona",
            "agent_b_audio_persona",
            "configured_tts_engine",
            "configured_asr_engine",
            "asr_search_width",
            "model_param_key",
            "objective_mode",
        ],
    )
    ta_table = text_audio_table(matched)
    validity = metric_validity_table(metrics)

    effects.to_csv(scope.docs_dir / "THESIS_CONFIGURATION_EFFECTS_TABLE.csv", index=False)
    ta_table.to_csv(scope.docs_dir / "THESIS_FULLY_MATCHED_TEXT_AUDIO_TABLE.csv", index=False)
    write_configuration_effect_doc(scope, conditions, matched, effects, ta_table)
    write_metric_validity_doc(scope, validity)
    write_result_inference_doc(scope, conditions, matched, validity)
    update_writing_aid(scope, conditions, matched, ta_table)

    print(f"conditions={len(conditions)}")
    print(f"completed={int(conditions['completed'].sum())}")
    print(f"matched_runs={len(matched)}")
    print(f"matched_condition_groups={matched['_matched_key'].nunique() if not matched.empty else 0}")
    print("wrote thesis result docs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
