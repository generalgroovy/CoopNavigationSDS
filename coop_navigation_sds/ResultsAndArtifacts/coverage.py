"""Build a results-root registry of planned and completed experiment coverage."""
from __future__ import annotations

import csv
import ctypes
import hashlib
import html
import json
import os
import platform
import uuid
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
from coop_navigation_sds.NaturalLanguageGeneration.models import (
    AGENT_A_TINYLLAMA_PROFILE_KEY,
    AGENT_A_USERLM_PROFILE_KEY,
    MODEL_PROFILE_SPECS,
    model_memory_requirement_gb,
)
from coop_navigation_sds.Configuration.runtime import MAXIMUM_PROGRESSIVE_CONSTRAINTS
from coop_navigation_sds.Configuration.travel import NETWORK_SEED
from coop_navigation_sds.TransportNetwork import network as network_model
from coop_navigation_sds.TransportNetwork.constraints import (
    CONSTRAINT_LABELS,
    stage_viability_report,
)
from coop_navigation_sds.TransportNetwork.test_cases import get_test_case


COVERAGE_FIELDS = (
    "condition_id",
    "experiment_platform",
    "matrix_family",
    "test_case_key",
    "scenario_key",
    "persona_key",
    "agent_a_type",
    "agent_a_audio_persona",
    "agent_b_model",
    "agent_b_llm_size",
    "agent_b_model_slot",
    "agent_b_model_role",
    "agent_b_audio_persona",
    "configured_tts_engine",
    "configured_asr_engine",
    "asr_search_width",
    "speech_pattern_key",
    "model_param_key",
    "objective_mode",
    "network_seed",
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

AGENT_MODEL_SLOTS = (
    ("small1", "small"),
    ("small2", "small"),
    ("medium1", "medium"),
    ("medium2", "medium"),
    ("large1", "large"),
    ("large2", "large"),
)
AGENT_MODEL_MATRIX_FAMILY = "agent_b_llm_comparison_v1"
AGENT_MODEL_MATRIX_FAMILIES = {
    AGENT_MODEL_MATRIX_FAMILY,
    "userlm_8b_agent_b_model_comparison_v2",
    "transformers_agent_b_speech_grid_v1",
}
AGENT_B_MODEL_SLOT_BY_MODEL = {
    "tinyllama/tinyllama-1.1b-chat-v1.0": "small1",
    "qwen/qwen2.5-0.5b-instruct": "small2",
    "huggingfacetb/smollm2-360m-instruct": "small3",
    "huggingfacetb/smollm2-1.7b-instruct": "small4",
    "qwen/qwen2.5-1.5b-instruct": "medium1",
    "microsoft/phi-3-mini-4k-instruct": "medium2",
    "google/gemma-2-2b-it": "medium3",
    "qwen/qwen3-4b-instruct-2507": "medium4",
    "qwen/qwen2.5-7b-instruct": "large1",
    "mistralai/mistral-7b-instruct-v0.3": "large2",
    "meta-llama/llama-3.1-8b-instruct": "large3",
    "tiiuae/falcon3-7b-instruct": "large4",
}
COVERAGE_SCHEMA_VERSION = 2


def _total_physical_memory_gb():
    """Return installed physical memory, or None when the host cannot report it."""
    if platform.system() == "Windows":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        try:
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return round(status.total_physical / (1024 ** 3), 2)
        except (AttributeError, OSError):
            return None
        return None
    try:
        return round(
            os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / (1024 ** 3),
            2,
        )
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def _levels(job, key, fallback):
    value = job.get("grid", {}).get(key, fallback)
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _model_slot_from_model(model):
    return AGENT_B_MODEL_SLOT_BY_MODEL.get(str(model or "").strip().lower())


def _coverage_key(row):
    payload = {
        field: (
            "unspecified"
            if field == "experiment_platform" and row.get(field) in (None, "")
            else row.get(field)
        )
        for field in COVERAGE_FIELDS
    }
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
            "condition_id": condition.condition_id,
            "experiment_platform": parameters.get("experiment_platform", "unspecified"),
            "matrix_family": parameters.get("matrix_family", job["name"]),
            "test_case_key": str(condition.test_case_key).split(":", 1)[0],
            "scenario_key": condition.scenario_key,
            "persona_key": condition.persona_key,
            "agent_a_type": config.get("agent_a_type", "staged"),
            "agent_a_audio_persona": condition.agent_a_audio_persona,
            "agent_b_model": condition.agent_b_model,
            "agent_b_llm_size": parameters.get("agent_b_llm_size"),
            "agent_b_model_slot": parameters.get("agent_b_model_slot"),
            "agent_b_model_role": parameters.get("agent_b_model_role"),
            "agent_b_audio_persona": condition.agent_b_audio_persona,
            "configured_tts_engine": condition.tts_engine,
            "configured_asr_engine": condition.asr_engine,
            "asr_search_width": parameters.get("asr_beam_size", config.get("asr_beam_size", "default")),
            "speech_pattern_key": condition.speech_pattern_key,
            "model_param_key": condition.model_param_key,
            "objective_mode": condition.objective_mode,
            "network_seed": parameters.get("network_seed", NETWORK_SEED),
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
            parameters = dict(condition.get("parameter_values") or {})
            if not parameters and condition.get("parameter_values_json"):
                try:
                    parameters = json.loads(condition["parameter_values_json"])
                except (TypeError, ValueError, json.JSONDecodeError):
                    parameters = {}
            row["matrix_family"] = row.get("matrix_family") or parameters.get("matrix_family")
            row["agent_b_llm_size"] = row.get("agent_b_llm_size") or parameters.get("agent_b_llm_size")
            row["agent_b_model_slot"] = (
                row.get("agent_b_model_slot")
                or parameters.get("agent_b_model_slot")
                or _model_slot_from_model(row.get("agent_b_model"))
            )
            row["agent_b_model_role"] = row.get("agent_b_model_role") or parameters.get("agent_b_model_role")
            row["test_case_key"] = str(
                condition.get("test_case_key") or condition.get("scenario_key") or ""
            ).split(":", 1)[0]
            row["network_seed"] = condition.get(
                "network_seed", condition.get("experiment_seed", NETWORK_SEED)
            )
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


def _case_coverage_rows(coverage_rows):
    """Audit every planned test-case, persona, and network-seed treatment."""
    treatments = sorted({
        (
            str(row.get("test_case_key") or row.get("scenario_key") or ""),
            str(row.get("persona_key") or ""),
            int(row.get("network_seed") or NETWORK_SEED),
        )
        for row in coverage_rows
        if row.get("planned") and (row.get("test_case_key") or row.get("scenario_key"))
    })
    rows = []
    original_seed = network_model.ACTIVE_NETWORK_SEED
    try:
        for test_case_key, persona_key, network_seed in treatments:
            network_model.rebuild_network(network_seed, force=True)
            test_case = get_test_case(test_case_key).with_persona(persona_key)
            report = stage_viability_report(
                test_case.scenario,
                test_case.persona,
                max_constraints=MAXIMUM_PROGRESSIVE_CONSTRAINTS,
            )
            selected = [
                row for row in coverage_rows
                if row.get("planned")
                and str(row.get("test_case_key") or row.get("scenario_key")) == test_case_key
                and str(row.get("persona_key")) == persona_key
                and int(row.get("network_seed") or NETWORK_SEED) == network_seed
            ]
            stages = report["stages"]
            constraint_stages = stages[1:]
            route_changes = sum(
                bool(stage["constraint_changes_optimal_route"])
                for stage in constraint_stages
            )
            rows.append({
                "test_case_key": test_case_key,
                "scenario_key": test_case.scenario_key,
                "persona_key": persona_key,
                "network_seed": network_seed,
                "constraint_order": ";".join(
                    report["constraint_order"][:MAXIMUM_PROGRESSIVE_CONSTRAINTS]
                ),
                "stage_count": len(stages),
                "satisfied_stage_count": sum(
                    bool(stage["requirement_satisfied"]) for stage in stages
                ),
                "constraint_route_change_count": route_changes,
                "required_constraint_route_change_count": len(constraint_stages),
                "minimum_viable_option_count": min(
                    (stage["viable_option_count"] for stage in stages), default=0
                ),
                "minimum_suboptimal_option_count": min(
                    (stage["suboptimal_option_count"] for stage in stages), default=0
                ),
                "planned_configuration_count": len(selected),
                "completed_configuration_count": sum(
                    int(row.get("completed_count") or 0) for row in selected
                ),
                "all_stage_requirements_satisfied": bool(
                    report["all_stage_requirements_satisfied"]
                ),
                "integrity_status": (
                    "pass" if report["all_stage_requirements_satisfied"] else "fail"
                ),
            })
    finally:
        network_model.rebuild_network(original_seed)
    return rows


def _case_coverage_summary(case_rows):
    covered_constraints = sorted({
        key
        for row in case_rows
        for key in str(row.get("constraint_order") or "").split(";")
        if key
    })
    missing_constraints = sorted(set(CONSTRAINT_LABELS) - set(covered_constraints))
    return {
        "treatment_count": len(case_rows),
        "passing_treatment_count": sum(
            row["integrity_status"] == "pass" for row in case_rows
        ),
        "failing_treatment_count": sum(
            row["integrity_status"] == "fail" for row in case_rows
        ),
        "covered_constraint_keys": covered_constraints,
        "missing_constraint_keys": missing_constraints,
        "all_treatments_valid": bool(case_rows) and not any(
            row["integrity_status"] == "fail" for row in case_rows
        ),
    }


def _active_run_rows(results_root):
    """Read non-finalized progress without treating it as completed evidence."""
    active = []
    for breakdown_path in sorted(Path(results_root).rglob("condition_configuration_breakdown.csv")):
        run_dir = breakdown_path.parent
        if (run_dir / "run_summary.json").is_file():
            continue
        with breakdown_path.open("r", encoding="utf-8", newline="") as handle:
            conditions = list(csv.DictReader(handle))
        if not conditions:
            continue
        first = conditions[0]
        failure_path = run_dir / "condition_failures.jsonl"
        failure_count = len(_read_jsonl(failure_path)) if failure_path.is_file() else 0
        finished_count = len(list(run_dir.glob("*-summary.json")))
        active.append({
            "result_run_id": run_dir.name,
            "run_path": str(run_dir.resolve()),
            "agent_a_type": first.get("agent_a_type"),
            "agent_b_model": first.get("agent_b_model"),
            "agent_b_llm_size": first.get("agent_b_llm_size"),
            "agent_b_model_slot": first.get("agent_b_model_slot"),
            "agent_b_model_role": first.get("agent_b_model_role"),
            "matrix_family": first.get("matrix_family"),
            "planned_condition_count": len(conditions),
            "observed_condition_count": min(finished_count, len(conditions)),
            "failed_condition_count": failure_count,
            "status": "incomplete",
        })
    return active


def _agent_model_combination_rows(coverage_rows, active_runs):
    """Build the two-caller by six-model-slot thesis coverage matrix."""
    system_ram_gb = _total_physical_memory_gb()
    agent_a_profiles = {
        "tinyllama": MODEL_PROFILE_SPECS[AGENT_A_TINYLLAMA_PROFILE_KEY],
        "userlm": MODEL_PROFILE_SPECS[AGENT_A_USERLM_PROFILE_KEY],
    }
    rows = []
    canonical = [
        row for row in coverage_rows
        if row.get("matrix_family") in AGENT_MODEL_MATRIX_FAMILIES
    ]
    for agent_a_type in ("tinyllama", "userlm"):
        for slot, size in AGENT_MODEL_SLOTS:
            selected = [
                row for row in canonical
                if row.get("agent_a_type") == agent_a_type
                and row.get("agent_b_model_slot") == slot
            ]
            planned = sum(bool(row.get("planned")) for row in selected)
            completed = sum(bool(row.get("planned") and row.get("completed_count")) for row in selected)
            successes = sum(int(row.get("successful_count") or 0) for row in selected)
            active = [
                row for row in active_runs
                if row.get("matrix_family") in AGENT_MODEL_MATRIX_FAMILIES
                and row.get("agent_a_type") == agent_a_type
                and row.get("agent_b_model_slot") == slot
            ]
            active_observed = sum(int(row["observed_condition_count"]) for row in active)
            status = (
                "complete" if planned and completed == planned
                else "active" if active
                else "partial" if completed
                else "missing"
            )
            rows.append({
                "agent_a_type": agent_a_type,
                "model_slot": slot,
                "agent_b_llm_size": size,
                "agent_b_model_role": ";".join(sorted({str(row.get("agent_b_model_role")) for row in selected if row.get("agent_b_model_role")})),
                "agent_b_models": ";".join(sorted({str(row.get("agent_b_model")) for row in selected})),
                "planned_configuration_count": planned,
                "completed_configuration_count": completed,
                "successful_configuration_count": successes,
                "coverage_percentage": round(100.0 * completed / planned, 3) if planned else 0.0,
                "active_observed_condition_count": active_observed,
                "active_planned_condition_count": sum(int(row["planned_condition_count"]) for row in active),
                "active_run_ids": ";".join(row["result_run_id"] for row in active),
                "status": status,
            })
            row = rows[-1]
            model_names = [name for name in row["agent_b_models"].split(";") if name]
            agent_b_ram = (
                model_memory_requirement_gb(model_names[0])
                if len(model_names) == 1 else None
            )
            agent_a_spec = agent_a_profiles[agent_a_type]
            agent_a_ram = agent_a_spec.approximate_memory_gb
            combined_ram = (
                round(agent_a_ram + agent_b_ram, 1)
                if agent_a_ram is not None and agent_b_ram is not None else None
            )
            row.update({
                "agent_a_model": agent_a_spec.model,
                "agent_a_approximate_memory_gb": agent_a_ram,
                "agent_b_approximate_memory_gb": agent_b_ram,
                "combined_approximate_memory_gb": combined_ram,
            })
            if system_ram_gb is not None:
                row["system_viability"] = (
                    "viable" if combined_ram is not None and combined_ram <= system_ram_gb
                    else "not_viable"
                )
    rows.sort(key=lambda row: (
        float("inf") if row["combined_approximate_memory_gb"] is None
        else row["combined_approximate_memory_gb"],
        row["agent_a_type"],
        row["model_slot"],
    ))
    control_coverage = [
        row for row in coverage_rows
        if row.get("agent_b_model_role") == "support_baseline"
    ]
    control_keys = sorted({
        (str(row.get("agent_a_type")), str(row.get("agent_b_model")))
        for row in control_coverage
    } | {
        (str(row.get("agent_a_type")), str(row.get("agent_b_model")))
        for row in active_runs if row.get("agent_b_model_role") == "support_baseline"
    })
    controls = []
    for agent, model in control_keys:
        selected = [
            row for row in control_coverage
            if str(row.get("agent_a_type")) == agent and str(row.get("agent_b_model")) == model
        ]
        active = [
            row for row in active_runs
            if str(row.get("agent_a_type")) == agent
            and str(row.get("agent_b_model")) == model
            and row.get("agent_b_model_role") == "support_baseline"
        ]
        controls.append({
            "agent_a_type": agent,
            "agent_b_model": model,
            "completed_configuration_count": sum(bool(row.get("planned") and row.get("completed_count")) for row in selected),
            "planned_configuration_count": sum(bool(row.get("planned")) for row in selected),
            "successful_configuration_count": sum(int(row.get("successful_count") or 0) for row in selected),
            "active_observed_condition_count": sum(int(row["observed_condition_count"]) for row in active),
            "active_planned_condition_count": sum(int(row["planned_condition_count"]) for row in active),
            "failed_condition_count": sum(int(row["failed_condition_count"]) for row in active),
            "run_ids": ";".join(sorted({
                run_id
                for row in selected
                for run_id in str(row.get("run_ids") or "").split(";")
                if run_id
            } | {row["result_run_id"] for row in active})),
        })
        control = controls[-1]
        control["status"] = (
            "complete"
            if control["planned_configuration_count"]
            and control["completed_configuration_count"] == control["planned_configuration_count"]
            else "active" if active
            else "partial" if control["completed_configuration_count"]
            else "missing"
        )
    return rows, controls


def _agent_model_html(rows, controls):
    lookup = {(row["agent_a_type"], row["model_slot"]): row for row in rows}
    display_slots = sorted(
        AGENT_MODEL_SLOTS,
        key=lambda slot_spec: min(
            (
                row["agent_b_approximate_memory_gb"]
                for row in rows
                if row["model_slot"] == slot_spec[0]
                and row["agent_b_approximate_memory_gb"] is not None
            ),
            default=float("inf"),
        ),
    )
    headers = "".join(f"<th>{slot}</th>" for slot, _size in display_slots)
    body = []
    for agent in ("tinyllama", "userlm"):
        cells = []
        for slot, _size in display_slots:
            row = lookup[(agent, slot)]
            model = html.escape(row["agent_b_models"] or "not resolved")
            cells.append(
                f'<td class="{row["status"]}"><strong>{row["completed_configuration_count"]}/'
                f'{row["planned_configuration_count"]}</strong><br><small>{model}</small><br>'
                f'<small>success={row["successful_configuration_count"]}; active='
                f'{row["active_observed_condition_count"]}/{row["active_planned_condition_count"]}</small></td>'
            )
        body.append(f"<tr><th>{agent}</th>{''.join(cells)}</tr>")
    viability_rows = []
    show_viability = bool(rows) and all("system_viability" in row for row in rows)
    for row in rows:
        required = row["combined_approximate_memory_gb"]
        viability_cell = ""
        if show_viability:
            viability = row["system_viability"]
            label = "Viable" if viability == "viable" else "Not viable"
            viability_cell = f'<td class="{viability}">{label}</td>'
        viability_rows.append(
            f'<tr><td>{html.escape(row["agent_a_type"])}</td>'
            f'<td>{html.escape(row["agent_a_model"])}</td>'
            f'<td>{html.escape(row["model_slot"])}</td>'
            f'<td>{html.escape(row["agent_b_models"] or "not resolved")}</td>'
            f'<td>{html.escape(row["agent_b_llm_size"])}</td>'
            f'<td>{row["agent_a_approximate_memory_gb"] or "unknown"}</td>'
            f'<td>{row["agent_b_approximate_memory_gb"] or "unknown"}</td>'
            f'<td><strong>{required or "unknown"}</strong></td>{viability_cell}</tr>'
        )
    viability_header = "<th>System viability</th>" if show_viability else ""
    detected_memory = _total_physical_memory_gb() if show_viability else None
    viability_note = (
        f" System viability compares the combined estimate with "
        f"{detected_memory:.2f} GiB detected physical RAM."
        if detected_memory is not None else ""
    )
    control_rows = "".join(
        f'<tr class="{row["status"]}"><td>{html.escape(str(row["agent_a_type"]))}</td><td>{html.escape(str(row["agent_b_model"]))}</td>'
        f'<td>{row["completed_configuration_count"]}/{row["planned_configuration_count"]}; active '
        f'{row["active_observed_condition_count"]}/{row["active_planned_condition_count"]}</td>'
        f'<td>{row["successful_configuration_count"]}/{row["completed_configuration_count"]}</td>'
        f'<td>{row["failed_condition_count"]}</td><td>{html.escape(row["run_ids"])}</td></tr>'
        for row in controls
    ) or '<tr><td colspan="5">No active control baseline.</td></tr>'
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Agent model coverage</title>
<style>body{{font:14px system-ui,sans-serif;background:#f3f5f7;color:#202a35;margin:0}}main{{max-width:1500px;margin:auto;padding:20px}}section{{background:#fff;border:1px solid #cbd3dc;padding:14px;margin:12px 0;overflow:auto}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #dce2e8;padding:8px;text-align:center;vertical-align:top}}th{{background:#eaf1ef}}.complete,.viable{{background:#dff1e7}}.active,.partial{{background:#fff0c9}}.missing,.not_viable{{background:#f7dada}}small{{color:#4b5966}}</style></head>
<body><main><h1>Agent A × Agent B model coverage</h1><p>Green is finalized, amber is active or partial, and red is planned but missing. Counts are completed/planned configurations; active progress is not promoted to completed evidence.</p>
<section><h2>General 2 × 6 comparison matrix</h2><table><thead><tr><th>Agent A</th>{headers}</tr></thead><tbody>{''.join(body)}</tbody></table></section>
<section><h2>Agent model memory requirements</h2><p>All 12 canonical Agent A and Agent B combinations are ordered by combined RAM requirement from lowest to highest. Values are registered planning estimates for simultaneously resident models and exclude speech, evaluation, and operating-system overhead.{viability_note}</p><table><thead><tr><th>Agent A</th><th>Agent A model</th><th>Slot</th><th>Agent B model</th><th>Agent B size</th><th>Agent A RAM (GiB)</th><th>Agent B RAM (GiB)</th><th>Combined RAM (GiB)</th>{viability_header}</tr></thead><tbody>{''.join(viability_rows)}</tbody></table></section>
<section><h2>Control baseline progress</h2><table><thead><tr><th>Agent A</th><th>Agent B</th><th>Observed/planned</th><th>Task success</th><th>Runtime failures</th><th>Active run</th></tr></thead><tbody>{control_rows}</tbody></table></section>
</main></body></html>"""


def _atomic_csv(path, rows):
    rows = list(rows)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = Path(f"{path}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    temporary.replace(path)


def _html_report(summary, matrix_rows, case_rows):
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
    case_body = "".join(
        f'<tr class="{row["integrity_status"]}">'
        f'<td>{html.escape(row["test_case_key"])}</td>'
        f'<td>{html.escape(row["persona_key"])}</td>'
        f'<td>{row["network_seed"]}</td>'
        f'<td>{html.escape(row["constraint_order"])}</td>'
        f'<td>{row["satisfied_stage_count"]}/{row["stage_count"]}</td>'
        f'<td>{row["constraint_route_change_count"]}/'
        f'{row["required_constraint_route_change_count"]}</td>'
        f'<td>{row["minimum_viable_option_count"]}</td>'
        f'<td>{row["minimum_suboptimal_option_count"]}</td>'
        f'<td>{row["completed_configuration_count"]}/'
        f'{row["planned_configuration_count"]}</td>'
        f'<td>{row["integrity_status"]}</td></tr>'
        for row in case_rows
    )
    case_summary = summary["case_coverage"]
    case_section = (
        "<section><h2>Task-case integrity</h2>"
        f"<p>{case_summary['passing_treatment_count']} of "
        f"{case_summary['treatment_count']} planned case/persona/seed treatments pass "
        "all staged-route requirements. Constraint coverage: "
        f"{html.escape(', '.join(case_summary['covered_constraint_keys']) or 'none')}."
        "</p><table><thead><tr><th>Test case</th><th>Persona</th><th>Seed</th>"
        "<th>Progressive constraints</th><th>Stages valid</th>"
        "<th>Constraint route changes</th><th>Min viable routes</th>"
        "<th>Min alternatives</th><th>Completed/planned</th><th>Integrity</th>"
        f"</tr></thead><tbody>{case_body}</tbody></table></section>"
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Experiment coverage</title>
<style>body{{font:14px system-ui,sans-serif;background:#f3f5f7;color:#202a35;margin:0}}main{{max-width:1500px;margin:auto;padding:20px}}section{{background:#fff;border:1px solid #cbd3dc;padding:14px;margin:12px 0;overflow:auto}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #dce2e8;padding:6px;text-align:center;white-space:nowrap}}th{{background:#eaf1ef}}.complete,.pass{{background:#dff1e7}}.partial{{background:#fff0c9}}.missing,.fail{{background:#f7dada}}</style></head>
<body><main><h1>CoopNavigationSDS experiment coverage</h1>
<p>{summary['completed_planned_configuration_count']} of {summary['planned_configuration_count']} planned configurations completed ({summary['coverage_percentage']:.3f}%). {summary['completed_run_count']} finalized runs indexed.</p>
{case_section}{''.join(sections)}</main></body></html>"""


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
    case_rows = _case_coverage_rows(coverage)
    case_summary = _case_coverage_summary(case_rows)
    active_runs = _active_run_rows(results_root)
    agent_model_rows, control_runs = _agent_model_combination_rows(coverage, active_runs)
    global_planned_count = sum(bool(row["planned"]) for row in coverage)
    global_completed_planned = sum(bool(row["planned"] and row["completed_count"]) for row in coverage)
    selected_slots = {slot for slot, _size in AGENT_MODEL_SLOTS}
    thesis_scope_rows = [
        row for row in coverage
        if row.get("planned")
        and row.get("matrix_family") in AGENT_MODEL_MATRIX_FAMILIES
        and row.get("agent_b_model_slot") in selected_slots
    ]
    planned_count = len(thesis_scope_rows)
    completed_planned = sum(bool(row.get("completed_count")) for row in thesis_scope_rows)
    summary = {
        "schema_version": COVERAGE_SCHEMA_VERSION,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_file_count": len(job_paths),
        "completed_run_count": len(runs),
        "planned_configuration_count": planned_count,
        "completed_planned_configuration_count": completed_planned,
        "coverage_scope": "selected_thesis_agent_b_slots",
        "selected_agent_b_model_slots": sorted(selected_slots),
        "global_planned_configuration_count": global_planned_count,
        "global_completed_planned_configuration_count": global_completed_planned,
        "global_coverage_percentage": round(100.0 * global_completed_planned / global_planned_count, 6) if global_planned_count else 0.0,
        "observed_unplanned_configuration_count": sum(row["status"] == "observed_unplanned" for row in coverage),
        "coverage_percentage": round(100.0 * completed_planned / planned_count, 6) if planned_count else 0.0,
        "case_coverage": case_summary,
        "files": {
            "conditions": "experiment_coverage_conditions.csv",
            "runs": "experiment_coverage_runs.csv",
            "matrix": "experiment_coverage_matrix.csv",
            "report": "experiment_coverage.html",
            "agent_model_matrix": "agent_model_combination_coverage.csv",
            "agent_model_report": "agent_model_combination_coverage.html",
            "case_coverage": "experiment_case_coverage.csv",
        },
    }
    paths = {
        "conditions": results_root / summary["files"]["conditions"],
        "runs": results_root / summary["files"]["runs"],
        "matrix": results_root / summary["files"]["matrix"],
        "summary": results_root / "experiment_coverage_summary.json",
        "report": results_root / summary["files"]["report"],
        "agent_model_matrix": results_root / summary["files"]["agent_model_matrix"],
        "agent_model_report": results_root / summary["files"]["agent_model_report"],
        "case_coverage": results_root / summary["files"]["case_coverage"],
    }
    _atomic_csv(paths["conditions"], coverage)
    _atomic_csv(paths["runs"], runs)
    _atomic_csv(paths["matrix"], matrices)
    _atomic_csv(paths["agent_model_matrix"], agent_model_rows)
    _atomic_csv(paths["case_coverage"], case_rows)
    temporary_summary = Path(f"{paths['summary']}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    temporary_summary.replace(paths["summary"])
    temporary_report = Path(f"{paths['report']}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary_report.write_text(_html_report(summary, matrices, case_rows), encoding="utf-8")
    temporary_report.replace(paths["report"])
    temporary_agent_report = Path(f"{paths['agent_model_report']}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary_agent_report.write_text(
        _agent_model_html(agent_model_rows, control_runs),
        encoding="utf-8",
    )
    temporary_agent_report.replace(paths["agent_model_report"])
    return paths
