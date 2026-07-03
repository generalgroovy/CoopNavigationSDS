"""Build a results-root registry of planned and completed experiment coverage."""
from __future__ import annotations

import csv
import hashlib
import html
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from coop_navigation_sds.Configuration.jobs import (
    job_linked_profiles,
    job_parameter_grid,
    job_parameter_profiles,
    load_experiment_job,
)
from coop_navigation_sds.experiments import build_condition_grid


COVERAGE_FIELDS = (
    "experiment_platform",
    "matrix_family",
    "scenario_key",
    "persona_key",
    "agent_a_type",
    "agent_a_audio_persona",
    "agent_b_model",
    "agent_b_llm_size",
    "agent_b_audio_persona",
    "configured_tts_engine",
    "configured_asr_engine",
    "asr_search_width",
    "speech_pattern_key",
    "model_param_key",
    "objective_mode",
    "iteration",
    "run_type",
)

MATRIX_DEFINITIONS = (
    ("TTS x speech pattern", "configured_tts_engine", "speech_pattern_key"),
    ("ASR x search width", "configured_asr_engine", "asr_search_width"),
    ("Agent B model x audio persona", "agent_b_model", "agent_b_audio_persona"),
    ("Agent A x Agent B size", "agent_a_type", "agent_b_llm_size"),
    ("Scenario x task persona", "scenario_key", "persona_key"),
)


def _levels(job, key, fallback):
    value = job.get("grid", {}).get(key, fallback)
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _coverage_key(row):
    payload = {field: row.get(field) for field in COVERAGE_FIELDS}
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]


def _planned_rows(job_path):
    job = load_experiment_job(job_path)
    config = job["config"]
    grid = job["grid"]
    conditions = build_condition_grid(
        test_case_keys=_levels(job, "test_cases", config.get("test_case_key", "morning_peak_cross_city")),
        persona_keys=_levels(job, "personas", config.get("persona_key", "focused_commuter")),
        speech_pattern_keys=_levels(job, "speech_patterns", config.get("speech_pattern_key", "clean")),
        model_param_keys=_levels(job, "model_params", "greedy"),
        objective_modes=_levels(job, "objective_modes", config.get("agent_a_objective_mode", "shortest_valid_route_with_constraints")),
        agent_a_audio_persona_keys=_levels(job, "agent_a_audio_personas", config.get("agent_a_audio_persona", "neutral_caller")),
        agent_b_audio_persona_keys=_levels(job, "agent_b_audio_personas", config.get("agent_b_audio_persona", "clear_operator")),
        tts_engine_keys=_levels(job, "tts_engines", config.get("tts_engine", "file")),
        asr_engine_keys=_levels(job, "asr_engines", config.get("asr_engine", "file")),
        agent_b_model_keys=_levels(job, "agent_b_models", config.get("model_name", "default_model")),
        iterations=job["iterations"],
        parameter_grid=job_parameter_grid(job),
        parameter_profiles=job_parameter_profiles(job),
        linked_profiles=job_linked_profiles(job),
        coverage_strategy=job["coverage_strategy"],
        pair_audio_with_text=bool(config.get("paired_audio_text_runs", False)),
    )
    for condition in conditions:
        parameters = dict(condition.parameter_values)
        yield {
            "experiment_platform": parameters.get("experiment_platform", "unspecified"),
            "matrix_family": parameters.get("matrix_family", job["name"]),
            "scenario_key": condition.scenario_key,
            "persona_key": condition.persona_key,
            "agent_a_type": config.get("agent_a_type", "staged"),
            "agent_a_audio_persona": condition.agent_a_audio_persona,
            "agent_b_model": condition.agent_b_model,
            "agent_b_llm_size": parameters.get("agent_b_llm_size"),
            "agent_b_audio_persona": condition.agent_b_audio_persona,
            "configured_tts_engine": condition.tts_engine,
            "configured_asr_engine": condition.asr_engine,
            "asr_search_width": parameters.get("asr_beam_size", config.get("asr_beam_size", "default")),
            "speech_pattern_key": condition.speech_pattern_key,
            "model_param_key": condition.model_param_key,
            "objective_mode": condition.objective_mode,
            "iteration": condition.iteration,
            "run_type": condition.run_type,
            "planned_by_job": job["name"],
        }


def _read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _completed_rows(results_root):
    runs = []
    rows = []
    for summary_path in sorted(Path(results_root).rglob("run_summary.json")):
        run_dir = summary_path.parent
        conditions_path = run_dir / "conditions.jsonl"
        if not conditions_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        completed_at = datetime.fromtimestamp(
            summary_path.stat().st_mtime, tz=timezone.utc
        ).isoformat()
        conditions = _read_jsonl(conditions_path)
        runs.append({
            "result_run_id": summary.get("result_run_id", run_dir.name),
            "result_scope": summary.get("result_scope", "unknown"),
            "condition_count": summary.get("condition_count", len(conditions)),
            "successful_condition_count": summary.get("successful_condition_count", 0),
            "completed_at_utc": completed_at,
            "run_path": str(run_dir.resolve()),
        })
        for condition in conditions:
            row = {
                field: condition.get(field)
                for field in COVERAGE_FIELDS
            }
            row["configured_tts_engine"] = condition.get(
                "configured_tts_engine", condition.get("tts_engine")
            )
            row["configured_asr_engine"] = condition.get(
                "configured_asr_engine", condition.get("asr_engine")
            )
            row.update({
                "result_run_id": summary.get("result_run_id", run_dir.name),
                "completed_at_utc": completed_at,
                "task_success": bool(condition.get("task_success")),
            })
            rows.append(row)
    return runs, rows


def _merge_coverage(planned_rows, completed_rows):
    merged = {}
    for row in planned_rows:
        key = _coverage_key(row)
        record = merged.setdefault(key, {
            "coverage_key": key,
            **{field: row.get(field) for field in COVERAGE_FIELDS},
            "planned_by_jobs": set(),
            "planned": True,
            "completed_count": 0,
            "successful_count": 0,
            "run_ids": set(),
            "last_completed_at_utc": "",
        })
        record["planned_by_jobs"].add(row["planned_by_job"])
    for row in completed_rows:
        key = _coverage_key(row)
        record = merged.setdefault(key, {
            "coverage_key": key,
            **{field: row.get(field) for field in COVERAGE_FIELDS},
            "planned_by_jobs": set(),
            "planned": False,
            "completed_count": 0,
            "successful_count": 0,
            "run_ids": set(),
            "last_completed_at_utc": "",
        })
        record["completed_count"] += 1
        record["successful_count"] += int(bool(row.get("task_success")))
        record["run_ids"].add(str(row.get("result_run_id")))
        record["last_completed_at_utc"] = max(
            record["last_completed_at_utc"], str(row.get("completed_at_utc") or "")
        )
    output = []
    for record in merged.values():
        record["status"] = (
            "completed" if record["planned"] and record["completed_count"]
            else "planned" if record["planned"]
            else "observed_unplanned"
        )
        record["planned_by_jobs"] = ";".join(sorted(record["planned_by_jobs"]))
        record["run_ids"] = ";".join(sorted(record["run_ids"]))
        output.append(record)
    return sorted(output, key=lambda row: tuple(str(row.get(field, "")) for field in COVERAGE_FIELDS))


def _matrix_rows(coverage_rows):
    rows = []
    for title, row_field, column_field in MATRIX_DEFINITIONS:
        cells = defaultdict(lambda: {"planned": set(), "completed": set(), "runs": set()})
        for item in coverage_rows:
            row_level = item.get(row_field)
            column_level = item.get(column_field)
            if row_level in (None, "") or column_level in (None, ""):
                continue
            cell = cells[(str(row_level), str(column_level))]
            if item["planned"]:
                cell["planned"].add(item["coverage_key"])
            if item["completed_count"]:
                cell["completed"].add(item["coverage_key"])
                cell["runs"].update(filter(None, str(item["run_ids"]).split(";")))
        for (row_level, column_level), cell in sorted(cells.items()):
            planned = len(cell["planned"])
            completed = len(cell["completed"] & cell["planned"])
            rows.append({
                "matrix": title,
                "row_factor": row_field,
                "row_level": row_level,
                "column_factor": column_field,
                "column_level": column_level,
                "planned_configuration_count": planned,
                "completed_configuration_count": completed,
                "coverage_percentage": round(100.0 * completed / planned, 3) if planned else None,
                "run_ids": ";".join(sorted(cell["runs"])),
            })
    return rows


def _atomic_csv(path, rows):
    rows = list(rows)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = Path(f"{path}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    temporary.replace(path)


def _html_report(summary, matrix_rows):
    sections = []
    for title, row_field, column_field in MATRIX_DEFINITIONS:
        selected = [row for row in matrix_rows if row["matrix"] == title]
        row_levels = sorted({row["row_level"] for row in selected})
        column_levels = sorted({row["column_level"] for row in selected})
        lookup = {(row["row_level"], row["column_level"]): row for row in selected}
        header = "".join(f"<th>{html.escape(level)}</th>" for level in column_levels)
        body = []
        for row_level in row_levels:
            cells = []
            for column_level in column_levels:
                cell = lookup.get((row_level, column_level), {})
                completed = cell.get("completed_configuration_count", 0)
                planned = cell.get("planned_configuration_count", 0)
                css = "complete" if planned and completed == planned else "partial" if completed else "missing"
                cells.append(f'<td class="{css}">{completed}/{planned}</td>')
            body.append(f"<tr><th>{html.escape(row_level)}</th>{''.join(cells)}</tr>")
        sections.append(
            f"<section><h2>{html.escape(title)}</h2><p>{html.escape(row_field)} by "
            f"{html.escape(column_field)}; cells show completed/planned configurations.</p>"
            f"<table><thead><tr><th>{html.escape(row_field)}</th>{header}</tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table></section>"
        )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Experiment coverage</title>
<style>body{{font:14px system-ui,sans-serif;background:#f3f5f7;color:#202a35;margin:0}}main{{max-width:1500px;margin:auto;padding:20px}}section{{background:#fff;border:1px solid #cbd3dc;padding:14px;margin:12px 0;overflow:auto}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #dce2e8;padding:6px;text-align:center;white-space:nowrap}}th{{background:#eaf1ef}}.complete{{background:#dff1e7}}.partial{{background:#fff0c9}}.missing{{background:#f7dada}}</style></head>
<body><main><h1>CoopNavigationSDS experiment coverage</h1>
<p>{summary['completed_planned_configuration_count']} of {summary['planned_configuration_count']} planned configurations completed ({summary['coverage_percentage']:.3f}%). {summary['completed_run_count']} finalized runs indexed.</p>
{''.join(sections)}</main></body></html>"""


def update_experiment_coverage(results_root, job_roots=None):
    """Rebuild unified coverage artifacts from jobs and finalized result folders."""
    results_root = Path(results_root).resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).resolve().parents[2]
    roots = list(job_roots or (
        project_root / "jobs",
        project_root / "coop_navigation_sds" / "Configuration" / "presets",
    ))
    job_paths = sorted(
        path for root in roots for path in Path(root).rglob("*.job")
    )
    planned = [row for path in job_paths for row in _planned_rows(path)]
    runs, completed = _completed_rows(results_root)
    coverage = _merge_coverage(planned, completed)
    matrices = _matrix_rows(coverage)
    planned_count = sum(bool(row["planned"]) for row in coverage)
    completed_planned = sum(bool(row["planned"] and row["completed_count"]) for row in coverage)
    summary = {
        "schema_version": 1,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_file_count": len(job_paths),
        "completed_run_count": len(runs),
        "planned_configuration_count": planned_count,
        "completed_planned_configuration_count": completed_planned,
        "observed_unplanned_configuration_count": sum(row["status"] == "observed_unplanned" for row in coverage),
        "coverage_percentage": round(100.0 * completed_planned / planned_count, 6) if planned_count else 0.0,
        "files": {
            "conditions": "experiment_coverage_conditions.csv",
            "runs": "experiment_coverage_runs.csv",
            "matrix": "experiment_coverage_matrix.csv",
            "report": "experiment_coverage.html",
        },
    }
    paths = {
        "conditions": results_root / summary["files"]["conditions"],
        "runs": results_root / summary["files"]["runs"],
        "matrix": results_root / summary["files"]["matrix"],
        "summary": results_root / "experiment_coverage_summary.json",
        "report": results_root / summary["files"]["report"],
    }
    _atomic_csv(paths["conditions"], coverage)
    _atomic_csv(paths["runs"], runs)
    _atomic_csv(paths["matrix"], matrices)
    temporary_summary = Path(f"{paths['summary']}.tmp")
    temporary_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    temporary_summary.replace(paths["summary"])
    temporary_report = Path(f"{paths['report']}.tmp")
    temporary_report.write_text(_html_report(summary, matrices), encoding="utf-8")
    temporary_report.replace(paths["report"])
    return paths
