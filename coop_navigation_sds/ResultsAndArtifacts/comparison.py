"""Combine completed experiment runs into portable comparison tables and charts."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from coop_navigation_sds.NaturalLanguageGeneration.models import model_memory_requirement_gb


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
CORRELATION_MINIMUM_SAMPLES = 5
CONDITION_ANALYSIS_METRICS = (
    ("route_validity", "task_outcome_route_validity"),
    ("constraint_satisfaction", "task_outcome_constraint_satisfaction_rate"),
    ("word_error_rate", "asr_wer"),
    ("entity_error_rate", "asr_entity_error_rate"),
    ("asr_station_f1", "asr_station_f1"),
    ("repair_success_rate", "dialogue_management_repair_success_rate"),
    ("repair_count", "whole_dialogue_repair_count"),
    ("clarification_count", "whole_dialogue_clarification_count"),
    ("correction_turn_rate", "whole_dialogue_correction_turn_rate"),
    ("route_revision_count", "whole_dialogue_route_revision_count"),
    ("candidate_route_count", "whole_dialogue_candidate_count"),
    ("turns_to_success", "task_outcome_turns_to_success"),
    ("duration_regret", "task_outcome_duration_regret"),
    ("grounded_proposal_score", "agent_b_grounded_proposal_score"),
    ("task_focus_score", "whole_dialogue_task_focus_score"),
    ("goal_progress_auc", "whole_dialogue_goal_progress_auc"),
    ("dialogue_cost", "whole_dialogue_dialogue_cost"),
    ("failure_localization_score", "whole_dialogue_failure_localization_score"),
)
SPEECH_BAND_RANK = {"floor": 0, "challenging": 1, "nominal": 2, "ceiling": 3}
SPEECH_BAND_SCORE_TOLERANCE = 0.02
AGENT_B_SIZE_ORDER = {"small": 0, "medium": 1, "large": 2, "hosted": 3, "custom": 4, "": 9}
MODEL_COMPARISON_DIMENSIONS = (
    "agent_a_type",
    "scenario_key",
    "persona_key",
    "speech_pattern_key",
    "run_type",
    "tts_engine",
    "asr_engine",
    "asr_beam_size",
    "network_seed",
    "agent_a_audio_persona",
    "agent_b_audio_persona",
    "speech_performance_band",
    "model_param_key",
    "objective_mode",
    "iteration",
    "transfer_tolerance",
    "dialogue_stagnation_limit",
)
CANONICAL_AGENT_B_MODEL_NAMES = {
    "tinyllama/tinyllama-1.1b-chat-v1.0": "TinyLlama-1.1B-Chat-v1.0",
    "tinyllama-1.1b-chat-v1.0": "TinyLlama-1.1B-Chat-v1.0",
    "qwen/qwen2.5-0.5b-instruct": "Qwen2.5-0.5B-Instruct",
    "qwen2.5-0.5b-instruct": "Qwen2.5-0.5B-Instruct",
    "qwen/qwen2.5-1.5b-instruct": "Qwen2.5-1.5B-Instruct",
    "qwen2.5-1.5b-instruct": "Qwen2.5-1.5B-Instruct",
    "small2-qwen2.5-1.5b": "Qwen2.5-1.5B-Instruct",
    "microsoft/phi-3-mini-4k-instruct": "Phi-3-mini-4k-instruct",
    "phi-3-mini-4k-instruct": "Phi-3-mini-4k-instruct",
    "mid2-phi3-mini": "Phi-3-mini-4k-instruct",
    "qwen/qwen2.5-7b-instruct": "Qwen2.5-7B-Instruct",
    "qwen2.5-7b-instruct": "Qwen2.5-7B-Instruct",
    "mistralai/mistral-7b-instruct-v0.3": "Mistral-7B-Instruct-v0.3",
    "mistral-7b-instruct-v0.3": "Mistral-7B-Instruct-v0.3",
}
CURRENT_ANALYSIS_EXCLUDED_AGENT_B_MODELS = {"Mistral-7B-Instruct-v0.3"}
OBSOLETE_PIPELINE_FAILURE_TYPES = {"NameError", "ValueError"}
ACTIVE_AGENT_B_MODEL_SLOTS = {
    "TinyLlama-1.1B-Chat-v1.0": "small1",
    "Qwen2.5-0.5B-Instruct": "small2",
    "Qwen2.5-1.5B-Instruct": "medium1",
    "Phi-3-mini-4k-instruct": "medium2",
    "Qwen2.5-7B-Instruct": "large1",
}
ACTIVE_AGENT_B_SLOT_ORDER = ("small1", "small2", "medium1", "medium2", "large1")
MATCHED_AGENT_B_CONDITION_DIMENSIONS = (
    "agent_a_type",
    "test_case_key",
    "scenario_key",
    "persona_key",
    "speech_pattern_key",
    "run_type",
    "configured_tts_engine",
    "configured_asr_engine",
    "asr_beam_size",
    "network_seed",
    "agent_a_audio_persona",
    "agent_b_audio_persona",
    "speech_performance_band",
    "speech_performance_rank",
    "model_param_key",
    "objective_mode",
    "iteration",
    "transfer_tolerance",
    "dialogue_stagnation_limit",
    "matrix_family",
    "experiment_platform",
)
ARCHIVED_RESULT_DIRECTORY_PREFIXES = ("_archive",)
ARCHIVED_RESULT_DIRECTORY_NAMES = {"_safety", "_transfer"}


def _is_archived_result_path(path):
    """Return whether ``path`` belongs to a non-active result archive."""
    return any(
        part in ARCHIVED_RESULT_DIRECTORY_NAMES
        or any(part.startswith(prefix) for prefix in ARCHIVED_RESULT_DIRECTORY_PREFIXES)
        for part in Path(path).parts
    )


def discover_run_directories(paths):
    """Return unique standard run directories found at or below ``paths``."""
    discovered = set()
    for value in paths:
        root = Path(value).expanduser().resolve()
        candidates = [root] if root.is_dir() else []
        if root.is_dir():
            candidates.extend(
                path.parent
                for path in root.rglob("run_summary.json")
                if not _is_archived_result_path(path)
            )
        for candidate in candidates:
            if _is_archived_result_path(candidate):
                continue
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


def _comparison_condition_key(row):
    """Return a stable key for comparing identical conditions across Agent B models."""
    aliases = {
        "tts_engine": ("configured_tts_engine",),
        "asr_engine": ("configured_asr_engine",),
        "asr_beam_size": ("asr_search_width",),
        "network_seed": ("experiment_seed",),
    }
    payload = {}
    for key in MODEL_COMPARISON_DIMENSIONS:
        value = row.get(key, "")
        for alias in aliases.get(key, ()):
            if value not in (None, ""):
                break
            value = row.get(alias, "")
        payload[key] = str(value or "")
    digest = hashlib.sha1(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    return f"MC-{digest}"


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


def discover_evidence_run_directories(paths):
    """Discover finalized and interrupted batch directories without changing them."""
    discovered = set(discover_run_directories(paths))
    for value in paths:
        root = Path(value).expanduser().resolve()
        if not root.is_dir():
            continue
        if not _is_archived_result_path(root) and (root / "experiment_job.json").is_file():
            discovered.add(root)
        discovered.update(
            path.parent
            for path in root.rglob("experiment_job.json")
            if not _is_archived_result_path(path)
        )
    return sorted(discovered, key=lambda path: str(path).casefold())


def _source_inventory(roots, excluded_directory):
    """Hash source evidence while excluding the derived analysis destination."""
    excluded_directory = Path(excluded_directory).resolve()
    files = {}
    for root in roots:
        root = Path(root).resolve()
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if (
                not path.is_file()
                or excluded_directory in path.parents
                or _is_archived_result_path(path)
            ):
                continue
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            files[str(path)] = {"size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    return files


def _event_rows(path):
    rows = []
    try:
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    rows.append(row.get("event") if isinstance(row.get("event"), dict) else row)
    except (OSError, json.JSONDecodeError):
        return rows
    return rows


def _summary_sections(text):
    """Parse the runtime's human summary without inferring absent values."""
    sections = {}
    section = ""
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip().casefold()
            sections.setdefault(section, {})
        elif section and ":" in line:
            key, value = line.split(":", 1)
            sections[section][key.strip().casefold()] = value.strip()
    return sections


def _text_number(value):
    match = re.search(r"-?\d+(?:\.\d+)?", str(value or ""))
    return float(match.group()) if match else None


def _text_boolean(value):
    normalized = str(value or "").strip().casefold()
    if normalized in {"true", "yes", "1"}:
        return True
    if normalized in {"false", "no", "0"}:
        return False
    return None


def _normalized_model_name(value):
    text = str(value or "").replace("\\", "/").rstrip("/")
    if not text:
        return ""
    normalized = text.casefold()
    basename = text.rsplit("/", 1)[-1]
    return (
        CANONICAL_AGENT_B_MODEL_NAMES.get(normalized)
        or CANONICAL_AGENT_B_MODEL_NAMES.get(basename.casefold())
        or basename
    )


def _active_agent_b_model_slot(row):
    """Return the active thesis slot for selected Agent B models, excluding archived models."""
    return ACTIVE_AGENT_B_MODEL_SLOTS.get(_normalized_model_name(row.get("agent_b_model", "")), "")


def _current_analysis_condition_rows(rows):
    """Return rows suitable for active thesis analysis without editing raw evidence."""
    current = []
    for row in rows:
        model = _normalized_model_name(row.get("agent_b_model"))
        failure_type = str(row.get("pipeline_failure_type") or "").strip()
        if model in CURRENT_ANALYSIS_EXCLUDED_AGENT_B_MODELS:
            continue
        if failure_type in OBSOLETE_PIPELINE_FAILURE_TYPES:
            continue
        current.append(row)
    return current


def _current_analysis_related_rows(rows, current_conditions):
    valid = {
        (str(row.get("source_run", "")), str(row.get("condition_id", "")))
        for row in current_conditions
    }
    return [
        row for row in rows
        if (str(row.get("source_run", "")), str(row.get("condition_id", ""))) in valid
    ]


def _compact_text(value, limit=38):
    """Return a short visible label while keeping full values in HTML titles."""
    text = str(value or "")
    if len(text) <= limit:
        return text
    return f"{text[: max(0, limit - 1)]}..."


def _compact_model_label(value):
    name = _normalized_model_name(value)
    name = re.sub(r"[-_]?Instruct(?:-\w+)?$", "", name, flags=re.IGNORECASE)
    name = name.replace("-Chat-v1.0", "").replace("-Chat", "")
    return _compact_text(name, 30)


def _compact_run_label(value):
    text = str(value or "").replace("\\", "/").rstrip("/")
    parts = [part for part in text.split("/") if part]
    return _compact_text("/".join(parts[-3:]) if len(parts) >= 3 else text, 42)


def _compact_phase_label(value):
    return str(value or "").replace("_", " ")


def _compact_metric_label(value, limit=28):
    return _compact_text(str(value or "").replace("_", " "), limit)


def _planned_conditions(run_directory):
    breakdown = Path(run_directory) / "condition_configuration_breakdown.csv"
    return _read_csv(breakdown) if breakdown.is_file() else []


def _condition_sessions(run_directory):
    """Load one immutable runtime event stream per observed condition."""
    sessions = {}
    paths = sorted(Path(run_directory).glob("batch-*.jsonl"))
    runtime_events = Path(run_directory) / "runtime_events.jsonl"
    if runtime_events.is_file():
        paths.append(runtime_events)
    for path in paths:
        events = _event_rows(path)
        start = next((row for row in events if row.get("name") == "batch.condition.start"), None)
        if start is None:
            continue
        condition_id = str(start.get("payload", {}).get("condition_id") or "")
        if condition_id:
            sessions[condition_id] = {"path": path, "events": events}
    return sessions


def _condition_failures(run_directory):
    """Load captured condition failures without altering source evidence."""
    path = Path(run_directory) / "condition_failures.jsonl"
    if not path.is_file():
        return {}
    failures = {}
    for row in _read_jsonl(path):
        condition_id = str(row.get("condition_id") or "")
        if condition_id:
            failures[condition_id] = row
    return failures


def _condition_state(session):
    if session is None:
        return "not_started"
    names = {row.get("name") for row in session["events"]}
    return "completed" if "batch.condition.end" in names and "metrics" in names else "interrupted"


def _condition_runtime_view(run_id, run_directory, planned, session, failure=None):
    condition_id = str(planned.get("condition_id") or "")
    state = "interrupted" if failure is not None and session is None else _condition_state(session)
    events = session["events"] if session else []
    summary_event = next((row for row in reversed(events) if row.get("name") == "metrics"), None)
    sections = _summary_sections((summary_event or {}).get("payload", {}).get("metrics"))
    run_section = sections.get("run", {})
    task = sections.get("task", {})
    comparison = sections.get("comparison", {})
    execution = sections.get("execution", {})
    constraint_match = re.search(r"(\d+)\s*/\s*(\d+)", task.get("constraints satisfied", ""))
    row = {
        "source_run": run_id,
        "source_path": str(run_directory),
        "condition_id": condition_id,
        "condition_state": state,
        "run_type": planned.get("run_type", ""),
        "pair_id": planned.get("pair_id", ""),
        "model_comparison_condition_key": planned.get("model_comparison_condition_key", ""),
        "scenario_key": planned.get("scenario_key", ""),
        "persona_key": planned.get("persona_key", ""),
        "speech_pattern_key": planned.get("speech_pattern_key", ""),
        "speech_performance_band": planned.get("speech_performance_band", ""),
        "agent_a_type": planned.get("agent_a_type", ""),
        "agent_a_audio_persona": planned.get("agent_a_audio_persona", ""),
        "agent_b_audio_persona": planned.get("agent_b_audio_persona", ""),
        "agent_b_model": _normalized_model_name(planned.get("agent_b_model")),
        "agent_b_llm_size": planned.get("agent_b_llm_size", ""),
        "agent_b_model_role": planned.get("agent_b_model_role", ""),
        "model_param_key": planned.get("model_param_key", ""),
        "objective_mode": planned.get("objective_mode", ""),
        "iteration": planned.get("iteration", ""),
        "tts_engine": planned.get("configured_tts_engine", ""),
        "asr_engine": planned.get("configured_asr_engine", ""),
        "asr_beam_size": planned.get("asr_beam_size", ""),
        "network_seed": planned.get("network_seed", ""),
        "transfer_tolerance": planned.get("transfer_tolerance", ""),
        "dialogue_stagnation_limit": planned.get("dialogue_stagnation_limit", ""),
        "outcome": run_section.get("outcome") if state == "completed" else None,
        "task_success": _text_boolean(run_section.get("task success")) if state == "completed" else None,
        "stop_reason": run_section.get("stop reason") if state == "completed" else None,
        "route_valid": _text_boolean(task.get("route valid")) if state == "completed" else None,
        "destination_reached": _text_boolean(task.get("destination reached")) if state == "completed" else None,
        "route_duration_min": _text_number(task.get("duration")) if state == "completed" else None,
        "duration_limit_min": _text_number(task.get("duration limit")) if state == "completed" else None,
        "optimal_duration_min": _text_number(comparison.get("optimal duration")) if state == "completed" else None,
        "duration_gap_min": _text_number(comparison.get("duration gap")) if state == "completed" else None,
        "constraints_satisfied": int(constraint_match.group(1)) if constraint_match and state == "completed" else None,
        "constraints_total": int(constraint_match.group(2)) if constraint_match and state == "completed" else None,
        "candidates_compared": _text_number(comparison.get("candidates compared")) if state == "completed" else None,
        "route_revisions": _text_number(comparison.get("revisions")) if state == "completed" else None,
        "turn_count": _text_number(execution.get("turns")) if state == "completed" else None,
        "runtime_sec": _text_number(execution.get("runtime")) if state == "completed" else _number((failure or {}).get("runtime_sec")),
        "selected_route": task.get("selected route") if state == "completed" else None,
        "optimal_route": comparison.get("optimal route") if state == "completed" else None,
        "source_event_file": str(session["path"]) if session else None,
    }
    if failure is not None:
        failure_detail = failure.get("failure") if isinstance(failure.get("failure"), dict) else {}
        row["pipeline_failure_type"] = failure_detail.get("exception_type") or failure.get("pipeline_failure_type") or "captured_failure"
        row["failure_message"] = failure_detail.get("message") or failure.get("error_message") or ""
    row["model_comparison_condition_key"] = row["model_comparison_condition_key"] or _comparison_condition_key(row)
    return row


def _phase_comparison_rows(condition, session):
    if session is None:
        return []
    events = session["events"]
    timing = [row.get("payload", {}) for row in events if row.get("name") == "telemetry.phase_timing"]
    speech = [row.get("payload", {}) for row in events if row.get("name") == "telemetry.speech"]
    nlu = [row.get("payload", {}) for row in events if row.get("name") == "telemetry.nlu"]
    candidates = [row for row in events if row.get("name") == "candidate"]
    phase_fields = (
        ("natural_language_generation", "natural_language_generation_sec"),
        ("text_to_speech", "text_to_speech_processing_sec"),
        ("automatic_speech_recognition", "automatic_speech_recognition_processing_sec"),
        ("natural_language_understanding", "natural_language_understanding_sec"),
        ("dialogue_management", "dialogue_management_sec"),
        ("speech_pipeline", "speech_pipeline_wall_sec"),
    )
    rows = []
    for order, (phase, field) in enumerate(phase_fields, 1):
        values = [_number(item.get(field)) for item in timing]
        values = [value for value in values if value is not None]
        row = {
            "source_run": condition["source_run"],
            "condition_id": condition["condition_id"],
            "model_comparison_condition_key": condition["model_comparison_condition_key"],
            "condition_state": condition["condition_state"],
            "agent_a_type": condition["agent_a_type"],
            "agent_b_model": condition["agent_b_model"],
            "agent_b_llm_size": condition["agent_b_llm_size"],
            "tts_engine": condition["tts_engine"],
            "asr_engine": condition["asr_engine"],
            "run_type": condition["run_type"],
            "scenario_key": condition["scenario_key"],
            "persona_key": condition["persona_key"],
            "speech_pattern_key": condition["speech_pattern_key"],
            "agent_a_audio_persona": condition.get("agent_a_audio_persona", ""),
            "agent_b_audio_persona": condition.get("agent_b_audio_persona", ""),
            "phase_order": order,
            "phase": phase,
            "observation_count": len(values),
            "total_processing_sec": sum(values) if values else None,
            "mean_processing_sec": _mean(values),
            "maximum_processing_sec": max(values) if values else None,
            "pipeline_success_rate": None,
            "error_count": None,
            "misinterpreted_token_count": None,
            "correction_count": None,
            "valid_route_parse_rate": None,
            "candidate_count": None,
            "evidence_source": "runtime event telemetry; no retrospective substitution",
        }
        if phase in {"text_to_speech", "automatic_speech_recognition", "speech_pipeline"}:
            row["pipeline_success_rate"] = _mean([float(bool(item.get("pipeline_ok"))) for item in speech])
            row["error_count"] = sum(not bool(item.get("pipeline_ok")) for item in speech)
        if phase == "automatic_speech_recognition":
            row["misinterpreted_token_count"] = sum(len(item.get("misinterpreted_tokens") or ()) for item in speech)
            row["correction_count"] = sum(len(item.get("transcript_corrections") or ()) for item in speech)
        if phase == "natural_language_understanding":
            parsed = [item for item in nlu if item.get("route_valid") is not None]
            row["valid_route_parse_rate"] = _mean([float(bool(item.get("route_valid"))) for item in parsed])
        if phase == "dialogue_management":
            row["candidate_count"] = len(candidates)
        rows.append(row)
    return rows


def _state_agreement(snapshot_a, snapshot_b):
    keys = ("start_station", "destination_station", "start_time_min")
    a_values = snapshot_a.get("task_variables", {})
    b_values = snapshot_b.get("task_variables", {})
    known = [key for key in keys if a_values.get(key) is not None or b_values.get(key) is not None]
    return _mean([float(a_values.get(key) == b_values.get(key)) for key in known]) if known else None


def _dialogue_state_row(condition, session):
    if session is None:
        return {
            "source_run": condition["source_run"], "condition_id": condition["condition_id"],
            "condition_state": condition["condition_state"], "run_type": condition["run_type"],
            "scenario_key": condition["scenario_key"], "persona_key": condition["persona_key"],
        }
    events = session["events"]
    messages_a = [row for row in events if row.get("name") == "message.agent a"]
    messages_b = [row for row in events if row.get("name") == "message.agent b"]
    candidates = [row.get("payload", {}).get("candidate", {}) for row in events if row.get("name") == "candidate"]
    memories = [row.get("payload", {}) for row in events if row.get("name") == "telemetry.memory"]
    final = memories[-1].get("snapshots", {}) if memories else {}
    agent_a = final.get("Agent A", {})
    agent_b = final.get("Agent B", {})
    route_a = agent_a.get("current_route") or []
    route_b = agent_b.get("current_route") or []
    constraints_a = set(agent_a.get("active_constraints") or ())
    constraints_b = set(agent_b.get("active_constraints") or ())
    focuses = [
        (item.get("snapshots", {}).get("Agent A", {}).get("pending_focus"),
         item.get("snapshots", {}).get("Agent B", {}).get("pending_focus"))
        for item in memories
    ]
    return {
        "source_run": condition["source_run"],
        "condition_id": condition["condition_id"],
        "condition_state": condition["condition_state"],
        "run_type": condition["run_type"],
        "scenario_key": condition["scenario_key"],
        "persona_key": condition["persona_key"],
        "message_count": len(messages_a) + len(messages_b),
        "agent_a_turn_count": len(messages_a),
        "agent_b_turn_count": len(messages_b),
        "memory_snapshot_count": len(memories),
        "candidate_count": len(candidates),
        "unique_route_count": len({tuple(item.get("route") or ()) for item in candidates if item.get("route")}),
        "state_task_variable_agreement": _state_agreement(agent_a, agent_b),
        "active_constraint_agreement": float(constraints_a == constraints_b),
        "current_route_agreement": float(route_a == route_b),
        "agent_a_constraint_count": len(constraints_a),
        "agent_b_constraint_count": len(constraints_b),
        "agent_a_clarification_count": agent_a.get("clarification_request_count"),
        "agent_b_clarification_count": agent_b.get("clarification_request_count"),
        "final_agent_a_focus": agent_a.get("pending_focus"),
        "final_agent_b_focus": agent_b.get("pending_focus"),
        "focus_disagreement_count": sum(left != right for left, right in focuses),
    }


def _conversation_rows(condition, session):
    if session is None:
        return []
    events = session["events"]
    speech = {
        int(row.get("payload", {}).get("turn")): row.get("payload", {})
        for row in events if row.get("name") == "telemetry.speech" and row.get("payload", {}).get("turn") is not None
    }
    nlu = {
        int(row.get("payload", {}).get("turn")): row.get("payload", {})
        for row in events if row.get("name") == "telemetry.nlu" and row.get("payload", {}).get("turn") is not None
    }
    rows = []
    for turn in sorted(speech):
        item = speech[turn]
        semantic = nlu.get(turn, {})
        rows.append({
            "source_run": condition["source_run"],
            "condition_id": condition["condition_id"],
            "condition_state": condition["condition_state"],
            "run_type": condition["run_type"],
            "turn": turn,
            "speaker": item.get("speaker"),
            "listener": "Agent B" if item.get("speaker") == "Agent A" else "Agent A",
            "intended_text": item.get("generated_text"),
            "tts_text": item.get("outgoing_text"),
            "asr_raw_text": item.get("raw_asr_transcript"),
            "understood_text": item.get("agent_input_transcript"),
            "misinterpretation_count": len(item.get("misinterpreted_tokens") or ()),
            "correction_count": len(item.get("transcript_corrections") or ()),
            "pipeline_ok": item.get("pipeline_ok"),
            "tts_latency_sec": item.get("tts_latency_sec"),
            "asr_latency_sec": item.get("asr_latency_sec"),
            "pipeline_latency_sec": item.get("pipeline_latency_sec"),
            "nlu_route_valid": semantic.get("route_valid"),
            "nlu_route_reaches_goal": semantic.get("route_reaches_goal"),
        })
    return rows


def _configuration_group_rows(condition_rows):
    dimensions = (
        "agent_a_type", "agent_b_model", "agent_b_llm_size", "tts_engine", "asr_engine",
        "scenario_key", "persona_key", "speech_pattern_key", "run_type", "asr_beam_size",
        "network_seed", "agent_a_audio_persona", "agent_b_audio_persona",
        "speech_performance_band", "transfer_tolerance", "dialogue_stagnation_limit",
    )
    groups = defaultdict(list)
    for row in condition_rows:
        groups[tuple(row.get(key) for key in dimensions)].append(row)
    output = []
    for identity, rows in groups.items():
        completed = [row for row in rows if row["condition_state"] == "completed"]
        observed = [row for row in rows if row["condition_state"] != "not_started"]
        successful = [row for row in completed if row.get("task_success") is True]
        output.append({
            **dict(zip(dimensions, identity)),
            "planned_condition_count": len(rows),
            "observed_condition_count": len(observed),
            "completed_condition_count": len(completed),
            "interrupted_condition_count": sum(row["condition_state"] == "interrupted" for row in rows),
            "not_started_condition_count": sum(row["condition_state"] == "not_started" for row in rows),
            "successful_condition_count": len(successful),
            "task_success_rate_completed": len(successful) / len(completed) if completed else None,
            "source_runs": _joined_values(rows, "source_run"),
        })
    return sorted(output, key=lambda row: tuple(str(row.get(key, "")).casefold() for key in dimensions))


def _numeric_values(rows, key):
    values = []
    for row in rows:
        value = _number(row.get(key))
        if value is not None:
            values.append(value)
    return values


def _rate(rows, key, expected=True):
    known = [row for row in rows if row.get(key) is not None]
    return (
        sum(row.get(key) is expected for row in known) / len(known)
        if known else None
    )


def _agent_b_sort_key(row):
    return (
        AGENT_B_SIZE_ORDER.get(str(row.get("agent_b_llm_size") or "").casefold(), 8),
        str(row.get("agent_b_model") or "").casefold(),
        str(row.get("run_type") or "").casefold(),
        str(row.get("scenario_key") or "").casefold(),
        str(row.get("persona_key") or "").casefold(),
    )


def _agent_b_model_summary_rows(condition_rows, phase_rows, dialogue_rows):
    groups = defaultdict(list)
    for row in condition_rows:
        groups[(row.get("agent_b_llm_size"), row.get("agent_b_model"), row.get("agent_a_type"))].append(row)
    phase_groups = defaultdict(list)
    for row in phase_rows:
        phase_groups[(row.get("agent_b_llm_size"), row.get("agent_b_model"), row.get("agent_a_type"))].append(row)
    dialogue_groups = defaultdict(list)
    for row in dialogue_rows:
        matching = next(
            (
                condition for condition in condition_rows
                if condition.get("condition_id") == row.get("condition_id")
            ),
            {},
        )
        dialogue_groups[(matching.get("agent_b_llm_size"), matching.get("agent_b_model"), matching.get("agent_a_type"))].append(row)
    output = []
    for (size, model, agent_a), rows in groups.items():
        completed = [row for row in rows if row.get("condition_state") == "completed"]
        phases = phase_groups.get((size, model, agent_a), [])
        dialogues = dialogue_groups.get((size, model, agent_a), [])
        output.append({
            "agent_b_llm_size": size,
            "agent_b_model": model,
            "agent_a_type": agent_a,
            "agent_b_estimated_memory_gb": model_memory_requirement_gb(model),
            "planned_conditions": len(rows),
            "completed_conditions": len(completed),
            "interrupted_conditions": sum(row.get("condition_state") == "interrupted" for row in rows),
            "not_started_conditions": sum(row.get("condition_state") == "not_started" for row in rows),
            "task_success_rate": _rate(completed, "task_success", True),
            "route_validity_rate": _rate(completed, "route_valid", True),
            "mean_duration_gap_min": _mean(_numeric_values(completed, "duration_gap_min")),
            "mean_turn_count": _mean(_numeric_values(completed, "turn_count")),
            "mean_phase_processing_sec": _mean(_numeric_values(phases, "mean_processing_sec")),
            "total_asr_misinterpretations": sum(
                _number(row.get("misinterpreted_token_count")) or 0
                for row in phases if row.get("phase") == "automatic_speech_recognition"
            ),
            "total_transcript_corrections": sum(
                _number(row.get("correction_count")) or 0
                for row in phases if row.get("phase") == "automatic_speech_recognition"
            ),
            "mean_dialogue_state_agreement": _mean(_numeric_values(dialogues, "state_task_variable_agreement")),
            "mean_route_agreement": _mean(_numeric_values(dialogues, "current_route_agreement")),
            "evidence_source": "condition, phase, and dialogue summaries grouped by Agent B model",
        })
    return sorted(output, key=_agent_b_sort_key)


def _program_execution_summary_rows(condition_rows, run_directories=None):
    """Summarize program/runtime completion separately from task satisfaction."""
    run_summary_groups = defaultdict(lambda: {
        "run_count": 0,
        "expected": 0,
        "completed": 0,
        "failures": 0,
    })
    for run_directory in run_directories or ():
        run_directory = Path(run_directory)
        summary_path = run_directory / "run_summary.json"
        conditions_path = run_directory / "conditions.jsonl"
        if not summary_path.is_file() or not conditions_path.is_file():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            first_condition = next(iter(_read_jsonl(conditions_path)), {})
        except Exception:
            continue
        identity = (
            first_condition.get("agent_b_llm_size", ""),
            _normalized_model_name(first_condition.get("agent_b_model", "")),
            first_condition.get("agent_a_type", ""),
        )
        expected = int(_number(summary.get("condition_count")) or 0)
        completed = int(_number(summary.get("completed_condition_count")) or 0)
        failures = int(_number(summary.get("failed_condition_count")) or 0)
        group = run_summary_groups[identity]
        group["run_count"] += 1
        group["expected"] += expected or max(completed + failures, 1)
        group["completed"] += completed
        group["failures"] += failures
    groups = defaultdict(list)
    for row in condition_rows:
        groups[(row.get("agent_b_llm_size"), row.get("agent_b_model"), row.get("agent_a_type"))].append(row)
    output = []
    for identity in sorted(
        set(groups) | set(run_summary_groups),
        key=lambda item: (
            AGENT_B_SIZE_ORDER.get(str(item[0] or "").casefold(), 8),
            str(item[1] or "").casefold(),
            str(item[2] or "").casefold(),
        ),
    ):
        size, model, agent_a = identity
        rows = groups.get(identity, [])
        summary_group = run_summary_groups.get(identity)
        if rows and any(str(row.get("condition_state") or "") for row in rows):
            expected = len(rows)
            completed = [
                row for row in rows
                if str(row.get("execution_status") or row.get("condition_state") or "").casefold() == "completed"
            ]
            captured_failures = sum(
                str(row.get("condition_state") or "").casefold() == "interrupted"
                for row in rows
            )
            missing = sum(
                str(row.get("condition_state") or "").casefold() == "not_started"
                for row in rows
            )
            finalized = expected - missing
            completed_count = len(completed)
        elif summary_group:
            expected = summary_group["expected"]
            finalized = expected
            completed = rows
            completed_count = summary_group["completed"]
            captured_failures = summary_group["failures"]
            missing = max(expected - completed_count - captured_failures, 0)
        else:
            expected = len(rows)
            finalized = len(rows)
            completed = rows
            completed_count = len(rows)
            captured_failures = 0
            missing = 0
        runtimes = _numeric_values(completed, "runtime_sec")
        output.append({
            "agent_b_llm_size": size,
            "agent_b_model": model,
            "agent_a_type": agent_a,
            "expected_conditions": expected,
            "observed_conditions": finalized,
            "finalized_result_folders": finalized,
            "finalized_rate": finalized / expected if expected else None,
            "program_completed_conditions": completed_count,
            "program_completion_rate": completed_count / expected if expected else None,
            "captured_program_failures": captured_failures,
            "missing_conditions": missing,
            "mean_completed_runtime_sec": _mean(runtimes),
            "max_completed_runtime_sec": max(runtimes) if runtimes else None,
            "text_only_conditions": sum(row.get("run_type") == "text_only" for row in rows),
            "audio_variant_conditions": sum(row.get("run_type") == "audio_variant" for row in rows),
            "program_failure_types": ";".join(
                f"{failure}:{count}"
                for failure, count in sorted(Counter(
                    row.get("pipeline_failure_type") or row.get("failure_phase") or "unknown"
                    for row in rows
                    if str(row.get("execution_status") or row.get("condition_state") or "").casefold() != "completed"
                ).items())
            ),
            "interpretation": "program execution only; task satisfaction and route success are reported in task tables",
        })
    return sorted(output, key=_agent_b_sort_key)


def _task_success_summary_rows(condition_rows):
    dimensions = (
        "agent_a_type", "agent_b_model", "agent_b_llm_size", "scenario_key",
        "persona_key", "speech_pattern_key", "run_type", "tts_engine", "asr_engine",
    )
    groups = defaultdict(list)
    for row in condition_rows:
        groups[tuple(row.get(key) for key in dimensions)].append(row)
    output = []
    for identity, rows in groups.items():
        completed = [row for row in rows if row.get("condition_state") == "completed"]
        observed = [row for row in rows if row.get("condition_state") != "not_started"]
        gaps = _numeric_values(completed, "duration_gap_min")
        turns = _numeric_values(completed, "turn_count")
        runtime = _numeric_values(completed, "runtime_sec")
        output.append({
            **dict(zip(dimensions, identity)),
            "planned_conditions": len(rows),
            "observed_conditions": len(observed),
            "completed_conditions": len(completed),
            "interrupted_conditions": sum(row.get("condition_state") == "interrupted" for row in rows),
            "not_started_conditions": sum(row.get("condition_state") == "not_started" for row in rows),
            "task_success_rate": _rate(completed, "task_success", True),
            "route_validity_rate": _rate(completed, "route_valid", True),
            "destination_reached_rate": _rate(completed, "destination_reached", True),
            "mean_duration_gap_min": _mean(gaps),
            "mean_turn_count": _mean(turns),
            "mean_runtime_sec": _mean(runtime),
            "mean_constraints_satisfied": _mean(_numeric_values(completed, "constraints_satisfied")),
            "mean_constraints_total": _mean(_numeric_values(completed, "constraints_total")),
            "evidence_source": "task_outcome_comparison.csv grouped without changing source results",
        })
    return sorted(output, key=_agent_b_sort_key)


def _model_configuration_matrix_rows(condition_rows):
    """Pivot matching non-model conditions so Agent B models are directly comparable."""
    grouped = defaultdict(dict)
    model_order = []
    for row in condition_rows:
        model = str(row.get("agent_b_model") or "unknown")
        if model not in model_order:
            model_order.append(model)
        key = row.get("model_comparison_condition_key") or _comparison_condition_key(row)
        grouped[key][model] = row
    model_order.sort(key=lambda model: (
        AGENT_B_SIZE_ORDER.get(str(next(
            (
                row.get("agent_b_llm_size")
                for rows in grouped.values()
                for candidate, row in rows.items()
                if candidate == model
            ),
            "",
        )).casefold(), 9),
        model.casefold(),
    ))
    output = []
    for key, by_model in sorted(grouped.items()):
        exemplar = next(iter(by_model.values()))
        row = {
            "model_comparison_condition_key": key,
            **{
                dimension: exemplar.get(dimension, "")
                for dimension in MODEL_COMPARISON_DIMENSIONS
            },
            "model_count": len(by_model),
            "completed_model_count": sum(
                item.get("condition_state") == "completed"
                for item in by_model.values()
            ),
            "successful_model_count": sum(
                item.get("task_success") is True
                for item in by_model.values()
            ),
        }
        for model in model_order:
            item = by_model.get(model, {})
            prefix = f"model::{model}"
            row[f"{prefix}::condition_state"] = item.get("condition_state", "")
            row[f"{prefix}::task_success"] = item.get("task_success", "")
            row[f"{prefix}::route_valid"] = item.get("route_valid", "")
            row[f"{prefix}::constraint_satisfaction"] = item.get("constraint_satisfaction")
            if row[f"{prefix}::constraint_satisfaction"] in (None, ""):
                row[f"{prefix}::constraint_satisfaction"] = (
                    None
                    if item.get("constraints_satisfied") is None or item.get("constraints_total") in (None, 0)
                    else float(item.get("constraints_satisfied")) / float(item.get("constraints_total"))
                )
            row[f"{prefix}::word_error_rate"] = (
                item.get("word_error_rate")
                if item.get("word_error_rate") not in (None, "")
                else item.get("word_error_rate", "")
            )
            row[f"{prefix}::turn_count"] = item.get("turn_count", "")
            row[f"{prefix}::failure_phase"] = item.get("failure_phase") or item.get("pipeline_failure_type") or ""
            row[f"{prefix}::source_run"] = item.get("source_run", "")
        output.append(row)
    return output


def _write_model_configuration_matrix_html(path, rows):
    """Write a compact visual matrix for model-by-configuration inspection."""
    rows = list(rows)
    model_names = []
    for row in rows:
        for key in row:
            if key.startswith("model::") and key.endswith("::task_success"):
                model = key.removeprefix("model::").removesuffix("::task_success")
                if model not in model_names:
                    model_names.append(model)
    model_names.sort(key=str.casefold)
    head = "".join(f"<th>{html.escape(model)}</th>" for model in model_names)
    body = []
    for row in rows:
        config = "<br>".join(html.escape(str(row.get(key, ""))) for key in (
            "scenario_key", "persona_key", "speech_pattern_key", "run_type",
            "tts_engine", "asr_engine", "asr_beam_size", "network_seed",
            "agent_a_audio_persona", "agent_b_audio_persona",
        ))
        cells = [f"<th>{config}</th>"]
        for model in model_names:
            prefix = f"model::{model}"
            state = row.get(f"{prefix}::condition_state", "")
            success = row.get(f"{prefix}::task_success", "")
            route = row.get(f"{prefix}::route_valid", "")
            constraints = _number(row.get(f"{prefix}::constraint_satisfaction"))
            wer = _number(row.get(f"{prefix}::word_error_rate"))
            turns = row.get(f"{prefix}::turn_count", "")
            failure = row.get(f"{prefix}::failure_phase", "")
            css = (
                "success" if success is True or str(success).casefold() == "true"
                else "failure" if success is False or str(success).casefold() == "false"
                else "partial" if state
                else "missing"
            )
            detail = (
                f"state={state}; route={route}; constraints="
                f"{'' if constraints is None else f'{constraints:.2f}'}; "
                f"wer={'' if wer is None else f'{wer:.2f}'}; turns={turns}; failure={failure}"
            )
            label = (
                "success" if css == "success"
                else "failed" if css == "failure"
                else state or "missing"
            )
            cells.append(
                f'<td class="{css}" title="{html.escape(detail)}">'
                f"{html.escape(label)}<br><small>{html.escape(detail)}</small></td>"
            )
        body.append("<tr>" + "".join(cells) + "</tr>")
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Agent B model by configuration matrix</title>
<style>body{{font:14px system-ui,sans-serif;background:#f3f5f7;color:#202a35;margin:0;padding:18px}}table{{border-collapse:separate;border-spacing:0;background:white}}th,td{{border:1px solid #dce2e8;padding:6px 8px;text-align:left;vertical-align:top;white-space:nowrap}}thead th{{position:sticky;top:0;background:#244b5a;color:white}}tbody th{{position:sticky;left:0;background:#eaf1ef;z-index:1}}.success{{background:#dff1e7}}.failure{{background:#f7dada}}.partial{{background:#fff0c9}}.missing{{background:#eef1f4;color:#5d6975}}small{{color:#44515d}}</style></head>
<body><h1>Agent B model by configuration matrix</h1><p>Derived view. Rows are identical non-model experimental conditions; columns are Agent B models. Raw run files remain authoritative.</p><table><thead><tr><th>Configuration</th>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></body></html>"""
    Path(path).write_text(document, encoding="utf-8")


def _phase_summary_rows(phase_rows):
    dimensions = (
        "agent_a_type", "agent_b_model", "agent_b_llm_size", "run_type",
        "tts_engine", "asr_engine", "phase_order", "phase",
    )
    groups = defaultdict(list)
    for row in phase_rows:
        groups[tuple(row.get(key) for key in dimensions)].append(row)
    output = []
    for identity, rows in groups.items():
        output.append({
            **dict(zip(dimensions, identity)),
            "condition_phase_rows": len(rows),
            "observed_phase_rows": sum(_number(row.get("observation_count")) not in (None, 0) for row in rows),
            "mean_observation_count": _mean(_numeric_values(rows, "observation_count")),
            "mean_processing_sec": _mean(_numeric_values(rows, "mean_processing_sec")),
            "maximum_processing_sec": max(_numeric_values(rows, "maximum_processing_sec") or [None]),
            "mean_pipeline_success_rate": _mean(_numeric_values(rows, "pipeline_success_rate")),
            "total_error_count": sum(_numeric_values(rows, "error_count")),
            "total_misinterpreted_tokens": sum(_numeric_values(rows, "misinterpreted_token_count")),
            "total_corrections": sum(_numeric_values(rows, "correction_count")),
            "mean_valid_route_parse_rate": _mean(_numeric_values(rows, "valid_route_parse_rate")),
            "total_candidates": sum(_numeric_values(rows, "candidate_count")),
            "evidence_source": "phase_metric_comparison.csv grouped by model/run type/phase",
        })
    return sorted(output, key=lambda row: (
        *_agent_b_sort_key(row),
        _number(row.get("phase_order")) or 0,
    ))


def _dialogue_summary_rows(dialogue_rows):
    dimensions = ("source_run", "run_type", "scenario_key", "persona_key")
    groups = defaultdict(list)
    for row in dialogue_rows:
        groups[tuple(row.get(key) for key in dimensions)].append(row)
    output = []
    for identity, rows in groups.items():
        output.append({
            **dict(zip(dimensions, identity)),
            "condition_count": len(rows),
            "completed_condition_count": sum(row.get("condition_state") == "completed" for row in rows),
            "mean_message_count": _mean(_numeric_values(rows, "message_count")),
            "mean_agent_a_turn_count": _mean(_numeric_values(rows, "agent_a_turn_count")),
            "mean_agent_b_turn_count": _mean(_numeric_values(rows, "agent_b_turn_count")),
            "mean_memory_snapshot_count": _mean(_numeric_values(rows, "memory_snapshot_count")),
            "mean_candidate_count": _mean(_numeric_values(rows, "candidate_count")),
            "mean_unique_route_count": _mean(_numeric_values(rows, "unique_route_count")),
            "mean_task_state_agreement": _mean(_numeric_values(rows, "state_task_variable_agreement")),
            "mean_constraint_agreement": _mean(_numeric_values(rows, "active_constraint_agreement")),
            "mean_route_agreement": _mean(_numeric_values(rows, "current_route_agreement")),
            "total_focus_disagreements": sum(_numeric_values(rows, "focus_disagreement_count")),
            "evidence_source": "dialogue_state_comparison.csv grouped by run/scenario/persona",
        })
    return sorted(output, key=lambda row: tuple(str(row.get(key, "")).casefold() for key in dimensions))


def _run_inventory_row(run_directory, planned, sessions, condition_rows):
    completed = sum(row["condition_state"] == "completed" for row in condition_rows)
    interrupted = sum(row["condition_state"] == "interrupted" for row in condition_rows)
    not_started = sum(row["condition_state"] == "not_started" for row in condition_rows)
    finalized = (Path(run_directory) / "run_summary.json").is_file()
    if finalized:
        state = "finalized"
    elif interrupted:
        state = "interrupted"
    elif sessions:
        state = "partial"
    else:
        state = "preflight_only"
    first = planned[0] if planned else {}
    successes = sum(row.get("task_success") is True for row in condition_rows)
    known_outcomes = sum(row.get("task_success") is not None for row in condition_rows)
    return {
        "source_run": Path(run_directory).name,
        "source_path": str(run_directory),
        "run_state": state,
        "agent_a_type": first.get("agent_a_type", ""),
        "agent_b_model": _normalized_model_name(first.get("agent_b_model")),
        "agent_b_llm_size": first.get("agent_b_llm_size", ""),
        "tts_engines": _joined_values(planned, "configured_tts_engine"),
        "asr_engines": _joined_values(planned, "configured_asr_engine"),
        "planned_condition_count": len(planned),
        "observed_condition_count": len(sessions),
        "completed_condition_count": completed,
        "interrupted_condition_count": interrupted,
        "not_started_condition_count": not_started,
        "known_outcome_count": known_outcomes,
        "successful_condition_count": successes,
        "task_success_rate_completed": successes / known_outcomes if known_outcomes else None,
        "finalized_artifacts_available": finalized,
        "interpretation": (
            "canonical finalized result" if finalized else
            "runtime evidence ended during a condition" if state == "interrupted" else
            "some condition evidence exists but canonical finalization is absent" if state == "partial" else
            "configuration was expanded; no condition runtime evidence was found"
        ),
    }


def _html_table(rows, columns, *, limit=None):
    selected = list(rows if limit is None else rows[:limit])
    header = "".join(f"<th>{html.escape(label)}</th>" for _key, label in columns)
    body = []
    for row in selected:
        state = str(row.get("run_state") or row.get("condition_state") or "")
        cells = []
        for key, _label in columns:
            value = row.get(key)
            if isinstance(value, float):
                value = f"{value:.4f}"
            cells.append(f"<td>{html.escape('' if value is None else str(value))}</td>")
        body.append(f'<tr class="{html.escape(state)}">{"".join(cells)}</tr>')
    if not body:
        body.append(f'<tr><td colspan="{len(columns)}">No evidence available.</td></tr>')
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _write_analysis_guide(path, generated_files):
    lines = [
        "# Result Analysis Guide",
        "",
        "This folder contains derived analysis tables only. The generator reads raw",
        "run evidence, writes normalized comparison views, hashes the source evidence",
        "before and after generation, and aborts if source evidence changes.",
        "",
        "## Files",
        "",
        "| File | Purpose |",
        "| --- | --- |",
        "| `run_inventory.csv` | One row per discovered run folder, including lifecycle state and completed/observed counts. |",
        "| `program_execution_summary.csv` | Program execution completion by Agent B model, independent of task success or route satisfaction. |",
        "| `configuration_groups.csv` | Planned and observed coverage grouped by controlled configuration factors. |",
        "| `configuration_conditions.csv` | Exact generated job conditions with staged-validity audit. |",
        "| `configuration_condition_overview.html` | Human-readable overview of planned configuration conditions. |",
        "| `task_outcome_comparison.csv` | One row per planned condition; unfinished outcomes remain blank. |",
        "| `task_success_by_configuration.csv` | Aggregated task success, route validity, duration gap, turns, and runtime by configuration. |",
        "| `model_configuration_matrix.csv` | Wide matrix joining identical non-model conditions across Agent B models. |",
        "| `model_configuration_matrix.html` | Color-coded inspection view of the same model-by-configuration matrix. |",
        "| `agent_b_model_summary.csv` | One row per Agent B model and Agent A pairing, sorted by Agent B size and model. |",
        "| `phase_metric_comparison.csv` | One row per condition and dialogue-system phase with telemetry counts and timings. |",
        "| `phase_summary_by_model.csv` | Phase-level aggregates by Agent A, Agent B, run type, TTS, ASR, and phase. |",
        "| `../comparison/metric_outcome_correlations.csv` | Descriptive Pearson correlations between pre-outcome metrics and task outcomes for finalized runs. |",
        "| `../comparison/metric_outcome_correlations.html` | Phase-wise readable correlation overview. |",
        "| `dialogue_state_comparison.csv` | One row per condition with memory, route, candidate, and agreement evidence. |",
        "| `dialogue_state_summary.csv` | Dialogue-state aggregates by run, scenario, persona, and run type. |",
        "| `conversation_turns.csv` | Turn-level intended text, TTS text, raw ASR, understood text, and correction counts. |",
        "| `analysis_manifest.json` | Source-evidence hash, generated file list, and read-only integrity check. |",
        "",
        "## Interpretation Rules",
        "",
        "- `completed` means the condition reached a canonical end event.",
        "- `interrupted` means runtime evidence exists but no successful condition end was recorded.",
        "- `not_started` means the condition was planned but has no observed session evidence.",
        "- Blank numeric outcomes are missing evidence, not zero.",
        "- HTML colors are exploratory visual markers only; statistical analysis should use the CSV files.",
        "",
        "## Generated Files",
        "",
        *[f"- `{name}`" for name in generated_files],
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _write_evidence_overview(path, inventory, configurations, outcomes, phases, dialogues, task_summary, model_summary, program_execution, phase_summary, dialogue_summary):
    run_table = _html_table(inventory, (
        ("source_run", "Run"), ("run_state", "State"), ("agent_a_type", "Agent A"),
        ("agent_b_model", "Agent B"), ("planned_condition_count", "Planned"),
        ("completed_condition_count", "Completed"), ("interrupted_condition_count", "Interrupted"),
        ("task_success_rate_completed", "Success rate"),
    ))
    config_table = _html_table(configurations, (
        ("agent_b_model", "Agent B"), ("scenario_key", "Scenario"), ("persona_key", "Persona"),
        ("speech_pattern_key", "Speech"), ("run_type", "Run type"),
        ("planned_condition_count", "Planned"), ("completed_condition_count", "Completed"),
    ), limit=80)
    outcome_table = _html_table(outcomes, (
        ("condition_id", "Condition"), ("condition_state", "State"), ("scenario_key", "Scenario"),
        ("run_type", "Run type"), ("task_success", "Success"), ("route_valid", "Route valid"),
        ("constraints_satisfied", "Constraints"), ("constraints_total", "Total"),
        ("route_duration_min", "Duration"), ("duration_gap_min", "Gap"), ("turn_count", "Turns"),
    ), limit=120)
    phase_table = _html_table(phases, (
        ("condition_id", "Condition"), ("phase", "Phase"), ("observation_count", "N"),
        ("mean_processing_sec", "Mean sec"), ("pipeline_success_rate", "Pipeline success"),
        ("error_count", "Errors"), ("misinterpreted_token_count", "Misinterpretations"),
        ("correction_count", "Corrections"),
    ), limit=160)
    dialogue_table = _html_table(dialogues, (
        ("condition_id", "Condition"), ("condition_state", "State"), ("message_count", "Messages"),
        ("candidate_count", "Candidates"), ("unique_route_count", "Unique routes"),
        ("state_task_variable_agreement", "Task-state agreement"),
        ("active_constraint_agreement", "Constraint agreement"),
        ("current_route_agreement", "Route agreement"), ("focus_disagreement_count", "Focus disagreements"),
    ), limit=120)
    task_summary_table = _html_table(task_summary, (
        ("agent_a_type", "Agent A"), ("agent_b_model", "Agent B"), ("scenario_key", "Scenario"),
        ("persona_key", "Persona"), ("run_type", "Run type"), ("planned_conditions", "Planned"),
        ("completed_conditions", "Completed"), ("task_success_rate", "Success"),
        ("route_validity_rate", "Route valid"), ("mean_duration_gap_min", "Mean gap"),
        ("mean_turn_count", "Mean turns"),
    ), limit=120)
    model_summary_table = _html_table(model_summary, (
        ("agent_b_llm_size", "Size"), ("agent_b_model", "Agent B"), ("agent_a_type", "Agent A"),
        ("agent_b_estimated_memory_gb", "Agent B RAM"), ("planned_conditions", "Planned"),
        ("completed_conditions", "Completed"), ("task_success_rate", "Success"),
        ("route_validity_rate", "Route valid"), ("mean_duration_gap_min", "Mean gap"),
        ("mean_phase_processing_sec", "Phase sec"), ("total_asr_misinterpretations", "ASR misheard"),
        ("mean_dialogue_state_agreement", "State agreement"),
    ), limit=120)
    program_execution_table = _html_table(program_execution, (
        ("agent_b_llm_size", "Size"), ("agent_b_model", "Agent B"), ("agent_a_type", "Agent A"),
        ("expected_conditions", "Expected"), ("observed_conditions", "Observed"),
        ("finalized_rate", "Finalized rate"), ("program_completed_conditions", "Program completed"),
        ("program_completion_rate", "Program completion"), ("captured_program_failures", "Captured failures"),
        ("missing_conditions", "Missing"), ("mean_completed_runtime_sec", "Mean runtime sec"),
        ("max_completed_runtime_sec", "Max runtime sec"), ("program_failure_types", "Failure types"),
    ), limit=120)
    phase_summary_table = _html_table(phase_summary, (
        ("agent_b_model", "Agent B"), ("run_type", "Run type"), ("phase", "Phase"),
        ("condition_phase_rows", "Rows"), ("mean_processing_sec", "Mean sec"),
        ("mean_pipeline_success_rate", "Pipeline success"), ("total_error_count", "Errors"),
        ("total_misinterpreted_tokens", "Misinterpretations"), ("total_corrections", "Corrections"),
        ("mean_valid_route_parse_rate", "Route parse"),
    ), limit=160)
    dialogue_summary_table = _html_table(dialogue_summary, (
        ("source_run", "Run"), ("scenario_key", "Scenario"), ("persona_key", "Persona"),
        ("run_type", "Run type"), ("mean_message_count", "Messages"),
        ("mean_candidate_count", "Candidates"), ("mean_unique_route_count", "Unique routes"),
        ("mean_task_state_agreement", "Task agreement"), ("mean_route_agreement", "Route agreement"),
        ("total_focus_disagreements", "Focus disagreements"),
    ), limit=120)
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Result evidence comparison</title>
<style>
body{{font:14px system-ui,sans-serif;margin:0;background:#f4f6f8;color:#1d2935}}main{{max-width:1800px;margin:auto;padding:20px}}
h1,h2{{margin:0 0 10px}}section{{background:#fff;border:1px solid #cad3dc;margin:12px 0;padding:14px;overflow:auto}}
nav a{{margin-right:16px;color:#15536d;font-weight:650}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
table{{border-collapse:collapse;width:100%}}th,td{{border-bottom:1px solid #dce2e8;padding:6px 8px;text-align:left;white-space:nowrap}}
th{{position:sticky;top:0;background:#244b5a;color:#fff}}.finalized,.completed{{background:#e1f2e8}}.partial{{background:#fff1c9}}
.interrupted{{background:#f8dddd}}.preflight_only,.not_started{{background:#edf0f3;color:#56616c}}code{{background:#edf1f4;padding:2px 4px}}
</style></head><body><main><h1>Result evidence comparison</h1>
<p>Derived representation only. Source logs, audio, configuration, and outcomes are not modified. Blank values mean unavailable evidence, not zero or failure.</p>
<nav><a href="analysis_guide.md">Guide</a><a href="configuration_condition_overview.html">Planned conditions</a><a href="run_inventory.csv">Runs</a><a href="program_execution_summary.csv">Program execution</a><a href="configuration_groups.csv">Configurations</a><a href="task_outcome_comparison.csv">Tasks</a><a href="../comparison/run_outcome_band_summary.html">Outcome bands</a><a href="task_success_by_configuration.csv">Task summary</a><a href="model_configuration_matrix.html">Model matrix</a><a href="../comparison/metric_value_heatmap.html">Metric heatmap</a><a href="../comparison/metric_outcome_correlations.html">Correlations</a><a href="agent_b_model_summary.csv">Agent B summary</a><a href="phase_metric_comparison.csv">Phase metrics</a><a href="phase_summary_by_model.csv">Phase summary</a><a href="dialogue_state_comparison.csv">Dialogue state</a><a href="dialogue_state_summary.csv">Dialogue summary</a><a href="conversation_turns.csv">Turns</a><a href="analysis_manifest.json">Integrity manifest</a></nav>
<section><h2>Run lifecycle</h2>{run_table}</section><section><h2>Configuration groups</h2>{config_table}</section>
<section><h2>Program execution, independent of task satisfaction</h2><p>This table counts whether the software finalized and completed each configured condition. It does not judge route quality, constraint satisfaction, or dialogue success.</p>{program_execution_table}</section>
<section><h2>Task and success</h2>{outcome_table}</section><section><h2>Outcome bands and indicating metrics</h2><p>Open <a href="../comparison/run_outcome_band_summary.html">run_outcome_band_summary.html</a> for successful, semi-successful, unsuccessful, and incomplete groups, and <a href="../comparison/metric_value_heatmap.html">metric_value_heatmap.html</a> for phase metric values colored by declared range and cross-run extremeness.</p></section><section><h2>Task summary by configuration</h2>{task_summary_table}</section>
<section><h2>Model-by-configuration matrix</h2><p>Open <a href="model_configuration_matrix.html">model_configuration_matrix.html</a> for the full color-coded pivot table.</p></section>
<section><h2>Agent B model summary</h2>{model_summary_table}</section>
<section><h2>Phase metrics</h2>{phase_table}</section><section><h2>Phase summary by model</h2>{phase_summary_table}</section>
<section><h2>Conversation and dialogue state</h2>{dialogue_table}</section><section><h2>Dialogue summary</h2>{dialogue_summary_table}</section></main></body></html>"""
    Path(path).write_text(document, encoding="utf-8")


def write_evidence_comparison(inputs, output_directory):
    """Create comparable derived views for every discovered run lifecycle state."""
    roots = [Path(value).expanduser().resolve() for value in inputs]
    output = Path(output_directory).expanduser().resolve()
    runs = discover_evidence_run_directories(roots)
    if not runs:
        raise ValueError("No result directories containing experiment_job.json or canonical run artifacts were found.")
    before = _source_inventory(roots, output)
    output.mkdir(parents=True, exist_ok=True)
    inventory_rows = []
    condition_rows = []
    phase_rows = []
    dialogue_rows = []
    conversation_rows = []
    for run_directory in runs:
        planned = _planned_conditions(run_directory)
        sessions = _condition_sessions(run_directory)
        failures = _condition_failures(run_directory)
        if not planned and (run_directory / "conditions.jsonl").is_file():
            planned = _read_jsonl(run_directory / "conditions.jsonl")
        run_id = run_directory.name
        run_conditions = []
        for planned_row in planned:
            condition_id = str(planned_row.get("condition_id") or "")
            condition = _condition_runtime_view(
                run_id, run_directory, planned_row, sessions.get(condition_id), failures.get(condition_id)
            )
            run_conditions.append(condition)
            session = sessions.get(condition["condition_id"])
            phase_rows.extend(_phase_comparison_rows(condition, session))
            dialogue_rows.append(_dialogue_state_row(condition, session))
            conversation_rows.extend(_conversation_rows(condition, session))
        condition_rows.extend(run_conditions)
        inventory_rows.append(_run_inventory_row(run_directory, planned, sessions, run_conditions))
    configuration_rows = _configuration_group_rows(condition_rows)
    task_summary_rows = _task_success_summary_rows(condition_rows)
    model_configuration_rows = _model_configuration_matrix_rows(condition_rows)
    model_summary_rows = _agent_b_model_summary_rows(condition_rows, phase_rows, dialogue_rows)
    program_execution_rows = _program_execution_summary_rows(condition_rows, runs)
    phase_summary_rows = _phase_summary_rows(phase_rows)
    dialogue_summary_rows = _dialogue_summary_rows(dialogue_rows)
    current_condition_rows = _current_analysis_condition_rows(condition_rows)
    current_phase_rows = _current_analysis_related_rows(phase_rows, current_condition_rows)
    current_dialogue_rows = _current_analysis_related_rows(dialogue_rows, current_condition_rows)
    current_conversation_rows = _current_analysis_related_rows(conversation_rows, current_condition_rows)
    current_program_execution_rows = _program_execution_summary_rows(current_condition_rows)
    current_task_summary_rows = _task_success_summary_rows(current_condition_rows)
    current_model_configuration_rows = _model_configuration_matrix_rows(current_condition_rows)
    current_model_summary_rows = _agent_b_model_summary_rows(
        current_condition_rows,
        current_phase_rows,
        current_dialogue_rows,
    )
    current_phase_summary_rows = _phase_summary_rows(current_phase_rows)
    current_dialogue_summary_rows = _dialogue_summary_rows(current_dialogue_rows)
    paths = {
        "overview": output / "comparison_overview.html",
        "guide": output / "analysis_guide.md",
        "run_inventory": output / "run_inventory.csv",
        "program_execution": output / "program_execution_summary.csv",
        "current_program_execution": output / "current_program_execution_summary.csv",
        "configurations": output / "configuration_groups.csv",
        "outcomes": output / "task_outcome_comparison.csv",
        "current_outcomes": output / "current_task_outcome_comparison.csv",
        "task_summary": output / "task_success_by_configuration.csv",
        "current_task_summary": output / "current_task_success_by_configuration.csv",
        "model_configuration_matrix": output / "model_configuration_matrix.csv",
        "current_model_configuration_matrix": output / "current_model_configuration_matrix.csv",
        "model_configuration_matrix_report": output / "model_configuration_matrix.html",
        "model_summary": output / "agent_b_model_summary.csv",
        "current_model_summary": output / "current_agent_b_model_summary.csv",
        "phases": output / "phase_metric_comparison.csv",
        "current_phases": output / "current_phase_metric_comparison.csv",
        "phase_summary": output / "phase_summary_by_model.csv",
        "current_phase_summary": output / "current_phase_summary_by_model.csv",
        "dialogue": output / "dialogue_state_comparison.csv",
        "current_dialogue": output / "current_dialogue_state_comparison.csv",
        "dialogue_summary": output / "dialogue_state_summary.csv",
        "current_dialogue_summary": output / "current_dialogue_state_summary.csv",
        "turns": output / "conversation_turns.csv",
        "current_turns": output / "current_conversation_turns.csv",
        "manifest": output / "analysis_manifest.json",
    }
    for key, rows in (
        ("run_inventory", inventory_rows), ("configurations", configuration_rows),
        ("program_execution", program_execution_rows),
        ("outcomes", condition_rows), ("task_summary", task_summary_rows),
        ("model_configuration_matrix", model_configuration_rows),
        ("model_summary", model_summary_rows),
        ("phases", phase_rows), ("phase_summary", phase_summary_rows),
        ("dialogue", dialogue_rows), ("dialogue_summary", dialogue_summary_rows),
        ("turns", conversation_rows),
        ("current_program_execution", current_program_execution_rows),
        ("current_outcomes", current_condition_rows),
        ("current_task_summary", current_task_summary_rows),
        ("current_model_configuration_matrix", current_model_configuration_rows),
        ("current_model_summary", current_model_summary_rows),
        ("current_phases", current_phase_rows),
        ("current_phase_summary", current_phase_summary_rows),
        ("current_dialogue", current_dialogue_rows),
        ("current_dialogue_summary", current_dialogue_summary_rows),
        ("current_turns", current_conversation_rows),
    ):
        _write_csv(paths[key], rows)
    _write_model_configuration_matrix_html(
        paths["model_configuration_matrix_report"],
        model_configuration_rows,
    )
    generated_file_names = [path.name for key, path in paths.items() if key != "manifest"]
    _write_analysis_guide(paths["guide"], generated_file_names)
    _write_evidence_overview(
        paths["overview"], inventory_rows, configuration_rows, condition_rows,
        phase_rows, dialogue_rows, task_summary_rows, model_summary_rows, program_execution_rows, phase_summary_rows,
        dialogue_summary_rows,
    )
    after = _source_inventory(roots, output)
    if before != after:
        raise RuntimeError("Source result evidence changed while derived analysis was generated.")
    aggregate = hashlib.sha256(
        "".join(f"{path}:{item['sha256']}\n" for path, item in sorted(before.items())).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "read-only derivation from canonical and partial runtime evidence",
        "source_evidence_unchanged": True,
        "source_file_count": len(before),
        "source_evidence_sha256": aggregate,
        "discovered_run_count": len(runs),
        "planned_condition_count": len(condition_rows),
        "observed_condition_count": sum(row["condition_state"] != "not_started" for row in condition_rows),
        "completed_condition_count": sum(row["condition_state"] == "completed" for row in condition_rows),
        "program_execution_summary_count": len(program_execution_rows),
        "current_analysis_filter": {
            "excluded_agent_b_models": sorted(CURRENT_ANALYSIS_EXCLUDED_AGENT_B_MODELS),
            "excluded_pipeline_failure_types": sorted(OBSOLETE_PIPELINE_FAILURE_TYPES),
            "current_condition_count": len(current_condition_rows),
        },
        "generated_files": generated_file_names,
        "source_files": before,
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    return paths


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
        condition_state = condition.get("condition_state") or condition.get("execution_status") or "completed"
        row = {
            "source_run": run_id,
            "source_path": condition.get("source_path", ""),
            "condition_id": condition_id,
            "condition_state": condition_state,
            "model_comparison_condition_key": (
                condition.get("model_comparison_condition_key")
                or _comparison_condition_key(condition)
            ),
            "pair_id": condition.get("pair_id", ""),
            "outcome": "success" if success else "failure",
            "task_success": success,
            "execution_status": condition.get("execution_status", condition_state),
            "failure_phase": text_lookup.get((run_id, condition_id, "whole_dialogue_failure_phase"), ""),
            "pipeline_failure_type": condition.get("pipeline_failure_type", ""),
            "test_case_key": condition.get("test_case_key", ""),
            "scenario_key": condition.get("scenario_key", ""),
            "persona_key": condition.get("persona_key", ""),
            "run_type": condition.get("run_type", ""),
            "speech_pattern_key": condition.get("speech_pattern_key", ""),
            "speech_performance_band": condition.get("speech_performance_band", ""),
            "speech_performance_rank": condition.get("speech_performance_rank", ""),
            "agent_a_type": condition.get("agent_a_type", ""),
            "agent_a_audio_persona": condition.get("agent_a_audio_persona", ""),
            "agent_b_audio_persona": condition.get("agent_b_audio_persona", ""),
            "agent_b_model": _normalized_model_name(condition.get("agent_b_model", "")),
            "agent_b_llm_size": condition.get("agent_b_llm_size", ""),
            "agent_b_model_role": condition.get("agent_b_model_role", ""),
            "agent_b_model_slot": _active_agent_b_model_slot(condition),
            "model_param_key": condition.get("model_param_key", ""),
            "objective_mode": condition.get("objective_mode", ""),
            "configured_tts_engine": condition.get("configured_tts_engine", condition.get("tts_engine", "")),
            "configured_asr_engine": condition.get("configured_asr_engine", condition.get("asr_engine", "")),
            "tts_engine": condition.get("configured_tts_engine", condition.get("tts_engine", "")),
            "asr_engine": condition.get("configured_asr_engine", condition.get("asr_engine", "")),
            "asr_search_width": condition.get("asr_search_width", ""),
            "asr_beam_size": condition.get("asr_search_width", condition.get("asr_beam_size", "")),
            "network_seed": condition.get("network_seed", condition.get("experiment_seed", "")),
            "transfer_tolerance": condition.get("transfer_tolerance", ""),
            "dialogue_stagnation_limit": condition.get("dialogue_stagnation_limit", ""),
            "matrix_family": condition.get("matrix_family", ""),
            "experiment_platform": condition.get("experiment_platform", ""),
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
        normalized_conditions = []
        for row in condition_rows:
            normalized = dict(row)
            normalized["model_comparison_condition_key"] = (
                normalized.get("model_comparison_condition_key")
                or _comparison_condition_key(normalized)
            )
            normalized_conditions.append(normalized)
        by_condition = {str(row.get("condition_id")): row for row in normalized_conditions}
        for row in normalized_conditions:
            conditions.append({"source_run": run_id, "source_path": str(run_dir), **row})
        for row in _read_csv(run_dir / "metrics_long.csv"):
            condition = by_condition.get(str(row.get("condition_id")), {})
            metrics.append({
                "source_run": run_id,
                "source_path": str(run_dir),
                "model_comparison_condition_key": condition.get("model_comparison_condition_key", ""),
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
        "source_run", "model_comparison_condition_key", "matrix_family", "experiment_platform",
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


def _safe_filename(value):
    text = str(value or "unassigned").strip().lower()
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in text)
    return "_".join(part for part in cleaned.split("_") if part) or "unassigned"


def build_phase_condition_metric_rows(conditions, metrics):
    """Join metric rows to condition outcomes for phase-wise cross-run inspection."""
    def display(value):
        return "" if value is None else value

    condition_by_key = {}
    first_condition_by_run = {}
    for condition in conditions:
        run_id = str(condition.get("source_run") or condition.get("result_run_id") or "")
        condition_id = str(condition.get("condition_id") or "")
        if run_id and condition_id:
            condition_by_key[(run_id, condition_id)] = condition
        first_condition_by_run.setdefault(run_id, condition)

    rows = []
    for metric in metrics:
        if str(metric.get("available", "true")).casefold() not in {"true", "1", "yes"}:
            continue
        run_id = str(metric.get("source_run") or metric.get("result_run_id") or "")
        condition_id = str(metric.get("condition_id") or "")
        condition = condition_by_key.get((run_id, condition_id), first_condition_by_run.get(run_id, {}))
        value = metric.get("value_numeric")
        phase = str(metric.get("phase") or "unassigned")
        rows.append({
            "phase_order": _phase_order(metric.get("phase_order")),
            "phase": phase,
            "agent_b_llm_size": condition.get("agent_b_llm_size") or "",
            "agent_a_type": condition.get("agent_a_type") or "",
            "agent_b_model": condition.get("agent_b_model") or "",
            "source_run": run_id,
            "condition_id": condition_id,
            "run_type": metric.get("run_type") or condition.get("run_type") or "",
            "scenario_key": condition.get("scenario_key") or "",
            "persona_key": condition.get("persona_key") or "",
            "speech_pattern_key": condition.get("speech_pattern_key") or "",
            "tts_engine": (
                condition.get("configured_tts_engine")
                or condition.get("tts_engine")
                or metric.get("configured_tts_engine")
                or ""
            ),
            "asr_engine": (
                condition.get("configured_asr_engine")
                or condition.get("asr_engine")
                or metric.get("configured_asr_engine")
                or ""
            ),
            "task_success": display(condition.get("task_success")),
            "route_valid": display(condition.get("route_valid")),
            "constraint_satisfaction": display(condition.get("constraint_satisfaction")),
            "failure_phase": condition.get("failure_phase") or condition.get("pipeline_failure_type") or "",
            "metric_key": metric.get("metric_key") or "",
            "metric_label": metric.get("metric_label") or metric.get("metric_key") or "",
            "value": value,
            "unit": metric.get("unit") or "",
            "range_min": metric.get("range_min") or "",
            "range_max": metric.get("range_max") or "",
            "higher_is_better": metric.get("higher_is_better") or "",
            "formula": metric.get("formula") or "",
            "operands_json": metric.get("operands_json") or "",
        })
    size_rank = {"small": 0, "medium": 1, "large": 2, "hosted": 3}
    return sorted(rows, key=lambda row: (
        row["phase_order"],
        row["phase"],
        size_rank.get(str(row["agent_b_llm_size"]).casefold(), 99),
        str(row["agent_b_model"]).casefold(),
        str(row["source_run"]).casefold(),
        str(row["condition_id"]).casefold(),
        str(row["metric_key"]).casefold(),
    ))


def write_phase_condition_metric_files(conditions, metrics, output_directory):
    """Write one comparison-ready metric evidence file per dialogue phase."""
    output = Path(output_directory)
    phase_dir = output / "phase_metrics"
    phase_dir.mkdir(parents=True, exist_ok=True)
    for obsolete in phase_dir.glob("*.csv"):
        obsolete.unlink()

    rows = build_phase_condition_metric_rows(conditions, metrics)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["phase"]].append(row)

    index_rows = []
    for phase, phase_rows in sorted(
        grouped.items(),
        key=lambda item: (min(row["phase_order"] for row in item[1]), item[0]),
    ):
        filename = f"{_safe_filename(phase)}.csv"
        _write_csv(phase_dir / filename, phase_rows)
        index_rows.append({
            "phase_order": min(row["phase_order"] for row in phase_rows),
            "phase": phase,
            "file": f"phase_metrics/{filename}",
            "row_count": len(phase_rows),
            "condition_count": len({(row["source_run"], row["condition_id"]) for row in phase_rows}),
            "metric_count": len({row["metric_key"] for row in phase_rows}),
        })
    index_path = output / "phase_metric_files.csv"
    _write_csv(index_path, index_rows)
    return index_path


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


def _metric_score(value, specification):
    """Return direction-adjusted 0..1 score for a metric value."""
    minimum = specification.get("range_min")
    maximum = specification.get("range_max")
    if value is None or minimum is None or maximum is None:
        return None
    position = 0.5 if maximum == minimum else (float(value) - minimum) / (maximum - minimum)
    position = min(1.0, max(0.0, position))
    return position if specification.get("higher_is_better", True) else 1.0 - position


def _value_from_row(row, *keys):
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return ""


def _outcome_band(row):
    """Classify task result for comparison without changing source evidence."""
    state = str(row.get("condition_state") or row.get("execution_status") or "").casefold()
    if state and state not in {"completed", "finalized"}:
        return "execution_incomplete"
    if _truthy(row.get("task_success")):
        return "successful"
    route_valid = _truthy(row.get("route_valid"))
    constraint_satisfaction = _number(row.get("constraint_satisfaction"))
    evaluation = _number(row.get("automatic_eval_score"))
    if route_valid or (constraint_satisfaction is not None and constraint_satisfaction > 0) or (
        evaluation is not None and evaluation >= 0.5
    ):
        return "semi_successful"
    return "unsuccessful"


def _metric_context_key(row):
    return (
        str(row.get("phase") or "unassigned"),
        str(row.get("metric_key") or "unnamed_metric"),
        str(row.get("run_type") or ""),
    )


def _observed_metric_statistics(metrics):
    grouped = defaultdict(list)
    for row in metrics:
        if str(row.get("available", "true")).casefold() not in {"true", "1", "yes"}:
            continue
        value = _number(row.get("value_numeric"))
        if value is None:
            continue
        grouped[_metric_context_key(row)].append(value)
    statistics_by_key = {}
    for key, values in grouped.items():
        values = sorted(values)
        median = statistics.median(values)
        deviations = [abs(value - median) for value in values]
        mad = statistics.median(deviations) if deviations else None
        statistics_by_key[key] = {
            "observed_minimum": values[0],
            "observed_maximum": values[-1],
            "observed_median": median,
            "observed_mad": mad,
            "sample_count": len(values),
            "values": values,
        }
    return statistics_by_key


def _percentile_rank(sorted_values, value):
    if not sorted_values:
        return None
    below_or_equal = sum(1 for item in sorted_values if item <= value)
    return below_or_equal / len(sorted_values)


def _heatmap_color(score, outlier_intensity=0.0):
    """Return red/amber/green color; stronger outliers are darker."""
    if score is None:
        return "#EEF1F4"
    base = _metric_cell_color(
        score,
        {"range_min": 0.0, "range_max": 1.0, "higher_is_better": True},
    )
    match = re.fullmatch(r"#([0-9A-Fa-f]{6})", base)
    if not match:
        return base
    channels = [int(match.group(1)[index:index + 2], 16) for index in (0, 2, 4)]
    intensity = min(1.0, max(0.0, float(outlier_intensity or 0.0)))
    # Normal values stay light; increasingly extreme observed deviations become darker.
    factor = 1.0 - 0.38 * intensity
    channels = [round(channel * factor) for channel in channels]
    return "#" + "".join(f"{channel:02X}" for channel in channels)


def build_outcome_band_summary(condition_analysis):
    """Summarize successful, semi-successful, unsuccessful, and incomplete conditions."""
    dimensions = (
        "agent_a_type", "agent_b_llm_size", "agent_b_model", "agent_b_model_role",
        "run_type", "speech_performance_band", "configured_tts_engine",
        "configured_asr_engine", "asr_search_width",
    )
    grouped = defaultdict(list)
    for row in condition_analysis:
        enriched = dict(row)
        enriched["outcome_band"] = _outcome_band(row)
        grouped[tuple(enriched.get(key, "") for key in (*dimensions, "outcome_band"))].append(enriched)
    output = []
    for identity, rows in grouped.items():
        values = dict(zip((*dimensions, "outcome_band"), identity))
        output.append({
            **values,
            "condition_count": len(rows),
            "task_success_count": sum(_truthy(row.get("task_success")) for row in rows),
            "route_valid_count": sum(_truthy(row.get("route_valid")) for row in rows),
            "mean_constraint_satisfaction": _mean(_numeric_values(rows, "constraint_satisfaction")),
            "mean_automatic_eval_score": _mean(_numeric_values(rows, "automatic_eval_score")),
            "mean_word_error_rate": _mean(_numeric_values(rows, "word_error_rate")),
            "mean_entity_error_rate": _mean(_numeric_values(rows, "entity_error_rate")),
            "mean_repair_success_rate": _mean(_numeric_values(rows, "repair_success_rate")),
            "mean_grounded_proposal_score": _mean(_numeric_values(rows, "grounded_proposal_score")),
            "mean_turn_count": _mean(_numeric_values(rows, "turn_count")),
            "mean_runtime_sec": _mean(_numeric_values(rows, "runtime_sec")),
            "failure_phase_counts": ";".join(
                f"{phase}:{count}"
                for phase, count in sorted(Counter(
                    row.get("failure_phase") or row.get("pipeline_failure_type") or "none"
                    for row in rows
                ).items())
            ),
            "interpretation": "derived outcome band; semi_successful means partial task progress without full task success",
        })
    band_order = {"successful": 0, "semi_successful": 1, "unsuccessful": 2, "execution_incomplete": 3}
    return sorted(output, key=lambda row: (
        AGENT_B_SIZE_ORDER.get(str(row.get("agent_b_llm_size", "")).casefold(), 9),
        str(row.get("agent_b_model", "")).casefold(),
        band_order.get(row.get("outcome_band"), 9),
        str(row.get("run_type", "")).casefold(),
    ))


def _matched_agent_b_condition_key(row):
    """Return a model-independent key for exact Agent B comparison."""
    payload = {
        key: str(row.get(key, "") or "")
        for key in MATCHED_AGENT_B_CONDITION_DIMENSIONS
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _deduplicated_matched_rows(condition_analysis):
    """Keep one retained row per Agent A, active Agent B slot, and condition."""
    rows = {}
    for row in condition_analysis:
        slot = row.get("agent_b_model_slot") or _active_agent_b_model_slot(row)
        if slot not in ACTIVE_AGENT_B_SLOT_ORDER:
            continue
        key = (str(row.get("agent_a_type") or ""), slot, _matched_agent_b_condition_key(row))
        priority = (
            bool(row.get("task_success")),
            str(row.get("condition_state") or "") == "completed",
            bool(row.get("route_valid")),
            str(row.get("source_run") or ""),
        )
        previous = rows.get(key)
        if previous is None:
            rows[key] = row
            continue
        previous_priority = (
            bool(previous.get("task_success")),
            str(previous.get("condition_state") or "") == "completed",
            bool(previous.get("route_valid")),
            str(previous.get("source_run") or ""),
        )
        if priority > previous_priority:
            rows[key] = row
    return rows


def build_matched_success_diagnostics(condition_analysis):
    """Summarize successful cases only where all active Agent B models cover a condition."""
    deduplicated = _deduplicated_matched_rows(condition_analysis)
    slots_by_condition = defaultdict(set)
    for agent_a, slot, comparable_key in deduplicated:
        slots_by_condition[(agent_a, comparable_key)].add(slot)
    matched_keys = {
        key for key, slots in slots_by_condition.items()
        if all(slot in slots for slot in ACTIVE_AGENT_B_SLOT_ORDER)
    }
    rows = []
    for agent_a in sorted({agent for agent, _ in matched_keys}):
        comparable_keys = sorted(key for key in matched_keys if key[0] == agent_a)
        for slot in ACTIVE_AGENT_B_SLOT_ORDER:
            matched_rows = []
            successful_rows = []
            for _, comparable_key in comparable_keys:
                row = deduplicated.get((agent_a, slot, comparable_key))
                if not row:
                    continue
                matched_rows.append(row)
                if bool(row.get("task_success")):
                    successful_rows.append(row)
            if not matched_rows:
                continue
            repair_counts = _numeric_values(successful_rows, "repair_count")
            rows.append({
                "agent_a_type": agent_a,
                "agent_b_model_slot": slot,
                "agent_b_model": _joined_values(matched_rows, "agent_b_model"),
                "agent_b_llm_size": _joined_values(matched_rows, "agent_b_llm_size"),
                "matched_case_count": len(matched_rows),
                "successful_matched_case_count": len(successful_rows),
                "matched_success_rate": len(successful_rows) / len(matched_rows) if matched_rows else None,
                "mean_turns_successful": _mean(_numeric_values(successful_rows, "turn_count")),
                "mean_turns_to_success": _mean(_numeric_values(successful_rows, "turns_to_success")),
                "mean_repair_turns_successful": _mean(repair_counts),
                "successful_runs_with_repair_rate": (
                    sum(value > 0 for value in repair_counts) / len(repair_counts)
                    if repair_counts else None
                ),
                "mean_repair_success_rate": _mean(_numeric_values(successful_rows, "repair_success_rate")),
                "mean_clarification_count": _mean(_numeric_values(successful_rows, "clarification_count")),
                "mean_correction_turn_rate": _mean(_numeric_values(successful_rows, "correction_turn_rate")),
                "mean_route_revision_count": _mean(_numeric_values(successful_rows, "route_revision_count")),
                "mean_candidate_route_count": _mean(_numeric_values(successful_rows, "candidate_route_count")),
                "mean_duration_regret_min": _mean(_numeric_values(successful_rows, "duration_regret")),
                "mean_constraint_satisfaction": _mean(_numeric_values(successful_rows, "constraint_satisfaction")),
                "mean_task_focus_score": _mean(_numeric_values(successful_rows, "task_focus_score")),
                "mean_asr_word_error_rate": _mean(_numeric_values(successful_rows, "word_error_rate")),
                "mean_asr_station_f1": _mean(_numeric_values(successful_rows, "asr_station_f1")),
                "mean_goal_progress_auc": _mean(_numeric_values(successful_rows, "goal_progress_auc")),
                "interpretation": (
                    "deduplicated matched Agent B comparison; metrics are means over successful "
                    "matched cases only, while matched_case_count is the all-model denominator"
                ),
            })
    return sorted(rows, key=lambda row: (
        str(row.get("agent_a_type") or "").casefold(),
        ACTIVE_AGENT_B_SLOT_ORDER.index(row["agent_b_model_slot"]),
    ))


def build_metric_value_heatmap_summary(condition_analysis, metrics):
    """Create compact phase/metric/outcome summaries for visual comparison."""
    conditions = {
        (str(row.get("source_run", "")), str(row.get("condition_id", ""))): row
        for row in condition_analysis
    }
    observed = _observed_metric_statistics(metrics)
    grouped = defaultdict(list)
    for row in metrics:
        if str(row.get("available", "true")).casefold() not in {"true", "1", "yes"}:
            continue
        value = _number(row.get("value_numeric"))
        if value is None:
            continue
        condition = conditions.get((str(row.get("source_run", "")), str(row.get("condition_id", ""))), {})
        if not condition:
            continue
        range_min = _number(row.get("range_min"))
        range_max = _number(row.get("range_max"))
        stats = observed.get(_metric_context_key(row), {})
        if range_min is None:
            range_min = stats.get("observed_minimum")
        if range_max is None:
            range_max = stats.get("observed_maximum")
        higher_is_better = str(row.get("higher_is_better", "")).casefold() in {"true", "1", "yes"}
        specification = {
            "range_min": range_min,
            "range_max": range_max,
            "higher_is_better": higher_is_better,
        }
        score = _metric_score(value, specification)
        mad = stats.get("observed_mad")
        median = stats.get("observed_median")
        modified_z = None
        if mad and mad > 0 and median is not None:
            modified_z = 0.67448975 * (value - median) / mad
        percentile = _percentile_rank(stats.get("values", ()), value)
        grouped[(
            _phase_order(row.get("phase_order")),
            str(row.get("phase") or "unassigned"),
            str(row.get("metric_key") or "unnamed_metric"),
            str(row.get("metric_label") or row.get("metric_key") or ""),
            str(row.get("run_type") or condition.get("run_type") or ""),
            str(condition.get("agent_b_llm_size") or ""),
            str(condition.get("agent_b_model") or ""),
            _outcome_band(condition),
        )].append({
            "value": value,
            "score": score,
            "modified_z": modified_z,
            "absolute_modified_z": abs(modified_z) if modified_z is not None else None,
            "percentile": percentile,
            "range_min": range_min,
            "range_max": range_max,
            "higher_is_better": higher_is_better,
        })
    output = []
    for identity, rows in grouped.items():
        (
            phase_order, phase, metric_key, metric_label, run_type,
            size, model, outcome_band,
        ) = identity
        scores = [row["score"] for row in rows if row["score"] is not None]
        absolute_z = [row["absolute_modified_z"] for row in rows if row["absolute_modified_z"] is not None]
        mean_score = _mean(scores)
        max_abs_z = max(absolute_z) if absolute_z else None
        outlier_count = sum((row["absolute_modified_z"] or 0.0) >= OUTLIER_MODIFIED_Z_THRESHOLD for row in rows)
        output.append({
            "phase_order": phase_order,
            "phase": phase,
            "metric_key": metric_key,
            "metric_label": metric_label or metric_key,
            "run_type": run_type,
            "agent_b_llm_size": size,
            "agent_b_model": model,
            "outcome_band": outcome_band,
            "observation_count": len(rows),
            "mean_value": _mean([row["value"] for row in rows]),
            "observed_minimum": min(row["value"] for row in rows),
            "observed_maximum": max(row["value"] for row in rows),
            "mean_direction_adjusted_score": mean_score,
            "mean_observed_percentile": _mean([row["percentile"] for row in rows if row["percentile"] is not None]),
            "maximum_absolute_modified_z_score": max_abs_z,
            "outlier_count": outlier_count,
            "outlier_rate": outlier_count / len(rows) if rows else None,
            "cell_color": _heatmap_color(mean_score, min(1.0, (max_abs_z or 0.0) / 8.0)),
            "range_min": rows[0]["range_min"],
            "range_max": rows[0]["range_max"],
            "higher_is_better": rows[0]["higher_is_better"],
            "interpretation": (
                "green is favorable by declared metric direction; red is unfavorable; "
                "darker cells are more extreme relative to other runs of the same phase/metric/run type"
            ),
        })
    band_order = {"successful": 0, "semi_successful": 1, "unsuccessful": 2, "execution_incomplete": 3}
    return sorted(output, key=lambda row: (
        row["phase_order"],
        row["phase"],
        row["metric_key"],
        AGENT_B_SIZE_ORDER.get(str(row["agent_b_llm_size"]).casefold(), 9),
        str(row["agent_b_model"]).casefold(),
        band_order.get(row["outcome_band"], 9),
        str(row["run_type"]).casefold(),
    ))


def write_outcome_band_summary_html(path, rows):
    """Write success/semi-success/failure overview focused on interpretable outcomes."""
    band_class = {
        "successful": "successful",
        "semi_successful": "semi",
        "unsuccessful": "unsuccessful",
        "execution_incomplete": "incomplete",
    }
    body = []
    for row in rows:
        css = band_class.get(row.get("outcome_band"), "")
        transport = " / ".join(
            str(row.get(key, "") or "-")
            for key in ("configured_tts_engine", "configured_asr_engine")
        )
        failure_phases = _compact_text(row.get("failure_phase_counts", ""), 56)
        body.append(
            f'<tr class="{css}">'
            f'<td><strong>{html.escape(str(row.get("outcome_band", "")))}</strong></td>'
            f'<td>{html.escape(str(row.get("agent_b_llm_size", "")))}</td>'
            f'<td title="{html.escape(str(row.get("agent_b_model", "")))}">{html.escape(_compact_model_label(row.get("agent_b_model", "")))}</td>'
            f'<td>{html.escape(str(row.get("run_type", "")))}</td>'
            f'<td>{html.escape(str(row.get("speech_performance_band", "")))}</td>'
            f'<td>{html.escape(str(row.get("condition_count", "")))}</td>'
            f'<td>{html.escape(str(row.get("route_valid_count", "")))}</td>'
            f'<td>{_display_number(row.get("mean_constraint_satisfaction"), percentage=True)}</td>'
            f'<td>{_display_number(row.get("mean_automatic_eval_score"), percentage=True)}</td>'
            f'<td>{_display_number(row.get("mean_word_error_rate"), percentage=True)} / {_display_number(row.get("mean_entity_error_rate"), percentage=True)}</td>'
            f'<td>{html.escape(transport)}</td>'
            f'<td>{_display_number(row.get("mean_turn_count"), decimals=1)}</td>'
            f'<td title="{html.escape(str(row.get("failure_phase_counts", "")))}">{html.escape(failure_phases)}</td>'
            "</tr>"
        )
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Outcome band summary</title><style>
body{{font:13px system-ui,sans-serif;background:#f3f5f7;color:#202a35;margin:0;padding:12px}}
section{{background:white;border:1px solid #cbd3dc;padding:10px;overflow:auto}}h1{{font-size:18px;margin:0 0 4px}}p{{margin:4px 0 8px;max-width:980px}}table{{border-collapse:separate;border-spacing:0;width:100%}}
th,td{{border-right:1px solid #dce2e8;border-bottom:1px solid #dce2e8;padding:4px 6px;text-align:left;white-space:nowrap;vertical-align:top}}
th{{position:sticky;top:0;background:#244b5a;color:white}}.successful{{background:#dff1e7}}.semi{{background:#fff0c9}}.unsuccessful{{background:#f7dada}}.incomplete{{background:#eef1f4;color:#5d6975}}
</style></head><body><section><h1>Successful, semi-successful, and unsuccessful runs</h1>
<p>Derived view only. Semi-successful means partial task progress without full task success. Full values remain in the CSV and hover titles.</p>
<table><thead><tr><th>Outcome</th><th>Size</th><th>Agent B</th><th>Run</th><th>Speech</th><th>N</th><th>Valid</th><th>Constraints</th><th>Eval</th><th>WER / Entity</th><th>TTS / ASR</th><th>Turns</th><th>Failure phases</th></tr></thead><tbody>{''.join(body) or '<tr><td colspan="13">No rows.</td></tr>'}</tbody></table>
</section></body></html>"""
    Path(path).write_text(document, encoding="utf-8")


def write_metric_value_heatmap_html(path, rows):
    """Write compact metric heatmap with range and cross-run extremeness coloring."""
    body = []
    for row in rows:
        title = (
            f'range={row.get("range_min")}..{row.get("range_max")}; '
            f'preferred={"higher" if row.get("higher_is_better") else "lower"}; '
            f'max abs modified z={_display_number(row.get("maximum_absolute_modified_z_score"), decimals=2)}'
        )
        body.append(
            f'<tr class="{html.escape(str(row.get("outcome_band", "")))}">'
            f'<td>{html.escape(str(row.get("phase_order", "")))}</td>'
            f'<td title="{html.escape(str(row.get("phase", "")))}">{html.escape(_compact_phase_label(row.get("phase", "")))}</td>'
            f'<td title="{html.escape(str(row.get("metric_key", "")))}">{html.escape(_compact_metric_label(row.get("metric_label", "")))}</td>'
            f'<td>{html.escape(str(row.get("agent_b_llm_size", "")))}</td>'
            f'<td title="{html.escape(str(row.get("agent_b_model", "")))}">{html.escape(_compact_model_label(row.get("agent_b_model", "")))}</td>'
            f'<td>{html.escape(str(row.get("run_type", "")))}</td>'
            f'<td>{html.escape(str(row.get("outcome_band", "")))}</td>'
            f'<td>{html.escape(str(row.get("observation_count", "")))}</td>'
            f'<td style="background:{html.escape(str(row.get("cell_color", "#EEF1F4")))}" title="{html.escape(title)}">'
            f'{_display_number(row.get("mean_value"), decimals=4)} ({_display_number(row.get("mean_direction_adjusted_score"), percentage=True)})</td>'
            f'<td>{html.escape(str(row.get("outlier_count", "")))}</td>'
            "</tr>"
        )
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Metric value heatmap</title><style>
body{{font:12px system-ui,sans-serif;background:#f3f5f7;color:#202a35;margin:0;padding:12px}}
h1{{font-size:18px;margin:0 0 4px}}p{{max-width:1100px;margin:4px 0 8px}}section{{background:white;border:1px solid #cbd3dc;padding:10px;overflow:auto}}table{{border-collapse:separate;border-spacing:0;width:100%}}
th,td{{border-right:1px solid #dce2e8;border-bottom:1px solid #dce2e8;padding:4px 5px;text-align:left;white-space:nowrap;vertical-align:top}}
th{{position:sticky;top:0;background:#244b5a;color:white}}small{{color:#202a35}}.successful td:first-child{{border-left:5px solid #14804a}}.semi_successful td:first-child{{border-left:5px solid #c78300}}.unsuccessful td:first-child{{border-left:5px solid #b42318}}.execution_incomplete td:first-child{{border-left:5px solid #687485}}
</style></head><body><section><h1>Phase metric heatmap by outcome band</h1>
<p>Green is favorable, red unfavorable. Darker cells are more extreme within comparable phase/metric/run-type groups. Use CSV for plots and tests.</p>
<table><thead><tr><th>#</th><th>Phase</th><th>Metric</th><th>Size</th><th>Agent B</th><th>Run type</th><th>Outcome</th><th>N</th><th>Value (favorable)</th><th>Outliers</th></tr></thead><tbody>{''.join(body) or '<tr><td colspan="10">No rows.</td></tr>'}</tbody></table>
</section></body></html>"""
    Path(path).write_text(document, encoding="utf-8")


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
        "agent_b_llm_size", "agent_a_type", "agent_b_model", "source_run",
        "task_outcome_status", "condition_count", "task_success_rate", "metric_outlier_count",
    )
    fixed_labels = {
        "agent_b_llm_size": "Size", "agent_a_type": "Agent A", "agent_b_model": "Agent B",
        "source_run": "Run", "task_outcome_status": "Outcome", "condition_count": "N",
        "task_success_rate": "Success", "metric_outlier_count": "Outliers",
    }
    by_phase = defaultdict(list)
    for column in metric_columns:
        by_phase[specifications[column]["phase"]].append(column)
    ordered_phases = list(by_phase.items())
    fixed_header_html = "".join(
        f'<th rowspan="2">{html.escape(fixed_labels[column])}</th>' for column in fixed_columns
    )
    phase_header_html = "".join(
        f'<th colspan="{len(columns)}" class="phase-header">{index}. {html.escape(phase)}</th>'
        for index, (phase, columns) in enumerate(ordered_phases, start=1)
    )
    metric_header_html = "".join(
        f'<th title="{html.escape(specifications[column]["metric_key"])}">'
        f'{html.escape(_compact_metric_label(specifications[column]["metric_label"], 18))}</th>'
        for _phase, columns in ordered_phases for column in columns
    )
    body = []
    for row in rows:
        status = str(row["task_outcome_status"])
        cells = []
        for column in fixed_columns:
            value = row.get(column)
            display = f"{100 * value:.1f}%" if column == "task_success_rate" and value is not None else value
            if column == "agent_b_model":
                display = _compact_model_label(value)
            elif column == "source_run":
                display = _compact_run_label(value)
            title = html.escape(str(value if value is not None else ""))
            cells.append(f'<td class="{html.escape(status)}" title="{title}">{html.escape(str(display))}</td>')
        for column in metric_columns:
            value = row.get(column)
            specification = specifications[column]
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
            badge = f'<span class="outlier-badge">!{len(outlier_rows)}</span>' if outlier_rows else ""
            title = ""
            if outlier_rows:
                title = (
                    f'{specification["metric_label"]}: {_display_number(value, decimals=4)}; '
                    "outliers=" + ", ".join(sorted(alignments))
                )
            title_attribute = f' title="{html.escape(title)}"' if title else ""
            cells.append(
                f'<td class="metric-cell {outlier_class}" style="background:{_metric_cell_color(value, specification)}" '
                f'{title_attribute}>{display}{badge}</td>'
            )
        body.append("<tr>" + "".join(cells) + "</tr>")
    empty = '<tr><td colspan="6">No finalized runs were found.</td></tr>'
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Run phase metric matrix</title><style>
body{{font:12px system-ui,sans-serif;color:#202A35;background:#F3F5F7;margin:0;padding:12px}}
table{{border-collapse:separate;border-spacing:0;background:white;box-shadow:0 1px 4px #BBC3CC}}
th,td{{border-right:1px solid #DCE2E8;border-bottom:1px solid #DCE2E8;padding:3px 5px;white-space:nowrap}}
thead tr:first-child th{{position:sticky;top:0;background:#244B5A;color:white;text-align:left;z-index:3}}
thead tr:nth-child(2) th{{position:sticky;top:24px;background:#376879;color:white;text-align:left;z-index:2;max-width:92px;overflow:hidden;text-overflow:ellipsis}}
.phase-header{{text-align:center!important;background:#173D4B!important}}
.all_successful{{background:#DFF1E7}}.mixed{{background:#FFF0C9}}.all_failed{{background:#F7DADA}}
.metric-cell{{position:relative;padding-right:6px;text-align:right}}.outlier-badge{{display:inline-block;font-size:10px;font-weight:700;margin-left:3px}}
.failure-outlier{{box-shadow:inset 0 0 0 3px #B42318}}.failure-outlier .outlier-badge{{color:#8A1710}}
.success-outlier{{box-shadow:inset 0 0 0 3px #14804A}}.success-outlier .outlier-badge{{color:#0D6338}}
.contradicting-outlier,.unknown-outlier{{box-shadow:inset 0 0 0 3px #6F4BA8}}.contradicting-outlier .outlier-badge,.unknown-outlier .outlier-badge{{color:#563884}}
a{{color:#174F69}}nav{{margin-bottom:8px}}nav a{{margin-right:12px;font-weight:650}}h1{{font-size:18px;margin:0 0 4px}}p{{margin:4px 0 8px;max-width:1200px}}
</style></head><body><h1>Completed runs by phase metric</h1>
<nav><a href="run_phase_metric_matrix.csv">Matrix CSV</a><a href="combined_metrics_long.csv">Metric evidence</a><a href="condition_analysis.csv">Conditions</a><a href="metric_outcome_correlations.html">Correlations</a><a href="metric_outliers.csv">Outliers</a></nav>
<p>One row per finalized run. Metric cells keep numeric values only; color encodes declared metric direction and range. <strong>!n</strong> marks robust outliers.</p>
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


def _pearson_correlation(left, right):
    pairs = [
        (float(x), float(y))
        for x, y in zip(left, right)
        if x is not None and y is not None
    ]
    if len(pairs) < CORRELATION_MINIMUM_SAMPLES:
        return None
    xs, ys = zip(*pairs)
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    denominator_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denominator_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if denominator_x == 0 or denominator_y == 0:
        return None
    return numerator / (denominator_x * denominator_y)


def summarize_metric_outcome_correlations(metrics, conditions):
    """Correlate pre-outcome metric values with observed task outcomes."""
    outcomes = {
        (str(row.get("source_run", "")), str(row.get("condition_id", ""))): row
        for row in conditions
    }
    grouped = defaultdict(list)
    metadata = {}
    for row in metrics:
        phase = str(row.get("phase", ""))
        if phase in OUTLIER_EXCLUDED_PHASES:
            continue
        if str(row.get("available", "true")).casefold() not in {"true", "1", "yes"}:
            continue
        value = _number(row.get("value_numeric"))
        if value is None:
            continue
        outcome = outcomes.get(
            (str(row.get("source_run", "")), str(row.get("condition_id", ""))),
            {},
        )
        if not outcome:
            continue
        key = (
            phase,
            str(row.get("metric_key", "")),
            str(row.get("run_type", "")),
        )
        grouped[key].append({
            "metric_value": value,
            "task_success": 1.0 if outcome.get("task_success") is True else 0.0,
            "route_validity": _number(outcome.get("route_validity")),
            "constraint_satisfaction": _number(outcome.get("constraint_satisfaction")),
        })
        metadata[key] = {
            "phase_order": _phase_order(row.get("phase_order")),
            "phase": phase,
            "metric_key": str(row.get("metric_key", "")),
            "metric_label": str(row.get("metric_label") or row.get("metric_key") or ""),
            "run_type": str(row.get("run_type", "")),
            "higher_is_better": str(row.get("higher_is_better", "")).casefold() in {"true", "1", "yes"},
        }

    rows = []
    for key, values in grouped.items():
        metric_values = [row["metric_value"] for row in values]
        base = {
            **metadata[key],
            "sample_count": len(values),
            "metric_mean": statistics.fmean(metric_values),
            "metric_minimum": min(metric_values),
            "metric_maximum": max(metric_values),
        }
        for outcome_key, label in (
            ("task_success", "Task success"),
            ("route_validity", "Route validity"),
            ("constraint_satisfaction", "Constraint satisfaction"),
        ):
            outcome_values = [row[outcome_key] for row in values]
            correlation = _pearson_correlation(metric_values, outcome_values)
            rows.append({
                **base,
                "outcome": outcome_key,
                "outcome_label": label,
                "outcome_mean": _mean(outcome_values),
                "pearson_r": correlation,
                "absolute_pearson_r": abs(correlation) if correlation is not None else None,
                "interpretation": (
                    "descriptive association; not causal and not validated without held-out data"
                    if correlation is not None
                    else f"unavailable; requires at least {CORRELATION_MINIMUM_SAMPLES} varying paired observations"
                ),
            })
    return sorted(rows, key=lambda row: (
        row["phase_order"],
        row["phase"],
        -1 if row["absolute_pearson_r"] is None else -row["absolute_pearson_r"],
        row["metric_key"],
        row["outcome"],
    ))


def write_metric_correlation_html(path, rows):
    """Write a compact phase-wise correlation overview."""
    rows = list(rows)
    phases = defaultdict(list)
    for row in rows:
        phases[row["phase"]].append(row)
    sections = []
    for phase_index, (phase, phase_rows) in enumerate(
        sorted(
            phases.items(),
            key=lambda item: (min(row["phase_order"] for row in item[1]), item[0]),
        ),
        start=1,
    ):
        body = []
        for row in sorted(
            phase_rows,
            key=lambda item: (
                -1 if item["absolute_pearson_r"] is None else -item["absolute_pearson_r"],
                item["metric_key"],
                item["outcome"],
            ),
        )[:80]:
            value = row["pearson_r"]
            strength = "" if value is None else f"{value:.3f}"
            color_value = None if value is None else abs(value)
            color = _metric_cell_color(
                color_value,
                {"range_min": 0.0, "range_max": 1.0, "higher_is_better": True},
            )
            body.append(
                "<tr>"
                f"<td>{html.escape(row['metric_label'])}</td>"
                f"<td>{html.escape(row['run_type'])}</td>"
                f"<td>{html.escape(row['outcome_label'])}</td>"
                f"<td>{row['sample_count']}</td>"
                f"<td style=\"background:{color}\">{strength}</td>"
                f"<td>{html.escape(row['interpretation'])}</td>"
                "</tr>"
            )
        sections.append(
            f"<section><h2>{phase_index}. {html.escape(phase)}</h2>"
            "<table><thead><tr><th>Metric</th><th>Run type</th><th>Outcome</th>"
            "<th>N</th><th>Pearson r</th><th>Interpretation</th></tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table></section>"
        )
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Metric outcome correlations</title>
<style>body{{font:14px system-ui,sans-serif;color:#202A35;background:#F3F5F7;margin:0;padding:20px}}main{{max-width:1500px;margin:auto}}section{{background:white;border:1px solid #CBD3DC;margin:12px 0;padding:12px;overflow:auto}}table{{border-collapse:collapse;width:100%}}th,td{{border-bottom:1px solid #DCE2E8;padding:6px 8px;text-align:left;white-space:nowrap}}th{{background:#244B5A;color:white}}p{{max-width:1000px}}</style></head>
<body><main><h1>Metric outcome correlations</h1>
<p>Descriptive Pearson correlations between pre-outcome metric values and task outcomes. Correlations summarize association only; they do not prove causation, and sparse or non-varying samples remain unavailable.</p>
{''.join(sections) or '<p>No correlations available.</p>'}</main></body></html>"""
    Path(path).write_text(document, encoding="utf-8")


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
    metric_correlations=(),
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

    phase_headers = "".join(
        f"<th>{index}. {html.escape(phase)}</th>"
        for index, phase in enumerate(phases, start=1)
    )
    correlation_rows = "".join(
        f'<tr><td>{html.escape(str(row.get("phase", "")))}</td>'
        f'<td>{html.escape(str(row.get("metric_label", row.get("metric_key", ""))))}</td>'
        f'<td>{html.escape(str(row.get("run_type", "")))}</td>'
        f'<td>{html.escape(str(row.get("outcome_label", "")))}</td>'
        f'<td>{row.get("sample_count", "")}</td>'
        f'<td>{_display_number(row.get("pearson_r"), decimals=3)}</td>'
        f'<td>{html.escape(str(row.get("interpretation", "")))}</td></tr>'
        for row in sorted(
            metric_correlations,
            key=lambda item: (
                -1 if item.get("absolute_pearson_r") is None else -float(item["absolute_pearson_r"]),
                str(item.get("phase", "")),
                str(item.get("metric_key", "")),
            ),
        )[:40]
    ) or '<tr><td colspan="7">No metric-outcome correlations available.</td></tr>'
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
<nav><a href="#runs">Runs</a><a href="#phases">Phases</a><a href="#correlations">Correlations</a><a href="#conditions">Conditions</a><a href="#outliers">Outliers</a></nav>
<h1>Experiment analysis overview</h1><p>Analysis-first index over canonical result tables. Raw metrics and protocol evidence remain authoritative.</p>
<div class="summary"><div><strong>{len(run_analysis)}</strong><small>finalized runs</small></div><div><strong>{total_conditions}</strong><small>conditions</small></div><div><strong>{successful}</strong><small>successful</small></div><div><strong>{failed}</strong><small>failed</small></div></div>
<section id="runs"><h2>Run comparison</h2><p>Matched audio/text delta is audio success minus its paired text control. A dash means the comparison is unavailable.</p>
<table><thead><tr><th>Run</th><th>Size</th><th>Agent A</th><th>Agent B</th><th>TTS</th><th>ASR</th><th>N</th><th>Success</th><th>Audio</th><th>Text</th><th>Audio delta</th><th>Route valid</th><th>Constraints</th><th>Audio WER</th><th>Repair</th><th>Turns</th><th>Runtime s</th><th>Failure phases</th><th>Evidence</th></tr></thead><tbody>{''.join(run_rows)}</tbody></table></section>
<section id="phases"><h2>Phase scorecard</h2><p>Equal-weighted descriptive mean over available metrics with declared ranges. Coverage is shown in every cell; compare component metrics before drawing conclusions.</p>
<table><thead><tr><th>Run</th>{phase_headers}</tr></thead><tbody>{''.join(phase_rows)}</tbody></table></section>
<section id="correlations"><h2>Metric-outcome correlations</h2><p>Top descriptive Pearson correlations between pre-outcome metrics and task outcomes. These are associations for inspection, not validated predictors.</p>
<table><thead><tr><th>Phase</th><th>Metric</th><th>Run type</th><th>Outcome</th><th>N</th><th>Pearson r</th><th>Interpretation</th></tr></thead><tbody>{correlation_rows}</tbody></table></section>
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
    metric_correlations = summarize_metric_outcome_correlations(metrics, conditions)
    condition_analysis = build_condition_analysis_rows(conditions, metrics)
    program_execution_rows = _program_execution_summary_rows(condition_analysis, run_directories)
    current_condition_analysis = _current_analysis_condition_rows(condition_analysis)
    current_program_execution_rows = _program_execution_summary_rows(current_condition_analysis)
    model_configuration_rows = _model_configuration_matrix_rows(condition_analysis)
    performance_bands = build_performance_band_summary(condition_analysis)
    outcome_band_summary = build_outcome_band_summary(condition_analysis)
    metric_value_heatmap = build_metric_value_heatmap_summary(condition_analysis, metrics)
    matched_success_diagnostics = build_matched_success_diagnostics(current_condition_analysis)
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
        "metric_correlations": output / "metric_outcome_correlations.csv",
        "metric_correlation_report": output / "metric_outcome_correlations.html",
        "run_metric_matrix": output / "run_phase_metric_matrix.csv",
        "run_metric_matrix_report": output / "run_phase_metric_matrix.html",
        "condition_analysis": output / "condition_analysis.csv",
        "current_condition_analysis": output / "current_condition_analysis.csv",
        "program_execution_summary": output / "program_execution_summary.csv",
        "current_program_execution_summary": output / "current_program_execution_summary.csv",
        "model_configuration_matrix": output / "model_configuration_matrix.csv",
        "model_configuration_matrix_report": output / "model_configuration_matrix.html",
        "performance_band_summary": output / "performance_band_summary.csv",
        "outcome_band_summary": output / "run_outcome_band_summary.csv",
        "outcome_band_summary_report": output / "run_outcome_band_summary.html",
        "metric_value_heatmap": output / "metric_value_heatmap.csv",
        "metric_value_heatmap_report": output / "metric_value_heatmap.html",
        "matched_success_diagnostics": output / "matched_success_diagnostics.csv",
        "phase_metric_files": output / "phase_metric_files.csv",
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
    _write_csv(paths["metric_correlations"], metric_correlations)
    _write_csv(paths["run_metric_matrix"], run_metric_matrix)
    _write_csv(paths["condition_analysis"], condition_analysis)
    _write_csv(paths["current_condition_analysis"], current_condition_analysis)
    _write_csv(paths["program_execution_summary"], program_execution_rows)
    _write_csv(paths["current_program_execution_summary"], current_program_execution_rows)
    _write_csv(paths["model_configuration_matrix"], model_configuration_rows)
    _write_csv(paths["performance_band_summary"], performance_bands)
    _write_csv(paths["outcome_band_summary"], outcome_band_summary)
    _write_csv(paths["metric_value_heatmap"], metric_value_heatmap)
    _write_csv(paths["matched_success_diagnostics"], matched_success_diagnostics)
    write_phase_condition_metric_files(conditions, metrics, output)
    _write_model_configuration_matrix_html(
        paths["model_configuration_matrix_report"],
        model_configuration_rows,
    )
    write_outcome_band_summary_html(
        paths["outcome_band_summary_report"],
        outcome_band_summary,
    )
    write_metric_value_heatmap_html(
        paths["metric_value_heatmap_report"],
        metric_value_heatmap,
    )
    write_run_phase_metric_matrix_html(
        paths["run_metric_matrix_report"],
        run_metric_matrix,
        run_metric_specifications,
        run_metric_outliers,
    )
    write_metric_correlation_html(paths["metric_correlation_report"], metric_correlations)
    for run_directory in run_directories:
        write_run_analysis_outputs(run_directory)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare standard CoopNavigationSDS result folders.")
    parser.add_argument("runs", nargs="+", help="Run folders or result roots to discover recursively.")
    parser.add_argument("--output", required=True, help="Directory for combined CSV files and the HTML report.")
    parser.add_argument(
        "--include-partial",
        action="store_true",
        help="Represent finalized, partial, interrupted, and preflight-only runs without modifying evidence.",
    )
    args = parser.parse_args(argv)
    paths = (
        write_evidence_comparison(args.runs, args.output)
        if args.include_partial
        else compare_runs(args.runs, args.output)
    )
    for label, path in paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
