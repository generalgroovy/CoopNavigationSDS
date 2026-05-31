"""Research artifact writers for model, network, and metric analysis data."""
from __future__ import annotations

from dataclasses import asdict
import csv
from datetime import datetime
import json
from pathlib import Path
import re

from minillama.evaluation.xlsx_export import write_metrics_xlsx
from minillama.evaluation.metrics import METRIC_FAMILY_SPECS, MetricComputer, phase_key_from_title
from minillama.model.network_overview import build_network_overview
from minillama.model.network_picture import write_network_svg


def safe_artifact_name(value):
    """Return a stable filesystem name for research artifacts."""
    text = str(value or "run").strip().lower()
    text = re.sub(r"[^a-z0-9_.-]+", "_", text)
    return text.strip("._") or "run"


def execution_run_id(label=None, timestamp=None):
    """Return a systematic identifier for one complete program execution."""
    if timestamp is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    elif hasattr(timestamp, "strftime"):
        stamp = timestamp.strftime("%Y%m%d_%H%M%S")
    else:
        stamp = safe_artifact_name(timestamp)
    return f"{stamp}_{safe_artifact_name(label or 'run')}"


def create_execution_run_dir(base_dir, label=None, timestamp=None):
    """Create and return a unique output folder for one execution run."""
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    run_id = execution_run_id(label, timestamp)
    candidate = base_dir / run_id
    suffix = 2
    while candidate.exists():
        candidate = base_dir / f"{run_id}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def run_scoped_path(run_dir, configured_path, default_name):
    """Resolve relative output paths inside an execution run directory."""
    if not configured_path:
        return Path(run_dir) / default_name
    path = Path(configured_path)
    return path if path.is_absolute() else Path(run_dir) / path


def write_metrics_csv(metrics, path):
    """Write flat metric rows as CSV."""
    records = list(metrics)
    if not records:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [record.as_dict() for record in records]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_metrics_file(metrics, path):
    """Write metrics as XLSX or CSV based on the requested file extension."""
    path = Path(path)
    records = list(metrics)
    if path.suffix.lower() == ".xlsx":
        write_metrics_xlsx(records, path)
    else:
        write_metrics_csv(records, path)


def write_metric_phase_logs(metrics, output_dir):
    """Write one JSONL file per metric phase for parsable analysis pipelines."""
    records = list(metrics)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    handles = {}
    try:
        for record in records:
            identifiers = {
                "condition_id": record.condition_id,
                "test_case_key": record.test_case_key,
                "persona_key": record.persona_key,
                "scenario_key": record.scenario_key,
                "speech_pattern_key": record.speech_pattern_key,
                "model_name": record.model_name,
                "model_param_key": record.model_param_key,
            }
            for phase, phase_metrics in sorted(record.metric_families.items()):
                handle = handles.get(phase)
                if handle is None:
                    handle = (output_dir / f"{phase}.jsonl").open("w", encoding="utf-8")
                    handles[phase] = handle
                handle.write(json.dumps({**identifiers, "phase": phase, "metrics": phase_metrics}, ensure_ascii=True) + "\n")
        summary_path = output_dir / "summary.jsonl"
        with summary_path.open("w", encoding="utf-8") as handle:
            for record in records:
                row = record.as_dict()
                row.pop("metric_families", None)
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
        catalog = {
            phase_key_from_title(family["title"]): {
                "title": family["title"],
                "metrics": [
                    {"key": key, "label": label}
                    for key, label in family["metrics"]
                ],
            }
            for family in METRIC_FAMILY_SPECS
        }
        (output_dir / "metric_catalog.json").write_text(json.dumps(catalog, indent=2, ensure_ascii=True), encoding="utf-8")
    finally:
        for handle in handles.values():
            handle.close()


def write_network_research_artifacts(current_time_min, output_dir, picture_dir=None):
    """Write network data JSON and SVG graph artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    overview = build_network_overview(current_time_min)
    network_json = output_dir / "network_overview.json"
    network_json.write_text(
        json.dumps(
            {
                "line_count": overview.line_count,
                "station_count": overview.station_count,
                "segment_count": overview.segment_count,
                "lines": [asdict(line) for line in overview.lines],
                "stations": [asdict(station) for station in overview.stations],
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    picture_root = Path(picture_dir) if picture_dir is not None else output_dir / "network_graphs"
    graph_path = write_network_svg(picture_root / "network_graph.svg")
    return {"network_json": network_json, "network_graph": graph_path}


def write_conversation_protocol(result, output_dir):
    """Write detailed, parsable protocol data for one completed conversation."""
    output_dir = Path(output_dir)
    condition_dir = output_dir / safe_artifact_name(result.condition_id)
    condition_dir.mkdir(parents=True, exist_ok=True)

    turns = [
        {
            "turn_index": index,
            "speaker": speaker,
            "utterance": utterance,
        }
        for index, (speaker, utterance) in enumerate(result.conversation, start=1)
    ]
    speech_turns = list(result.extra.get("speech_turns", []))
    timing_turns = list(result.extra.get("timing_turns", []))
    nlu_turns = list(result.extra.get("nlu_turns", []))
    agent_turn_segments = list(result.extra.get("agent_turn_segments", []))
    runtime_events = list(result.extra.get("runtime_events", []))
    candidate_events = list(result.extra.get("candidate_events", []))
    agent_timing_summary = result.extra.get("agent_timing_summary", {})
    verification = verify_conversation_protocol(
        result,
        turns,
        speech_turns,
        timing_turns,
        nlu_turns,
        agent_turn_segments,
        agent_timing_summary,
        runtime_events,
    )
    summary = {
        "condition_id": result.condition_id,
        "test_case_key": result.test_case_key,
        "persona_key": result.persona_key,
        "scenario_key": result.scenario_key,
        "speech_pattern_key": result.speech_pattern_key,
        "model_name": result.model_name,
        "message_count": len(turns),
        "route": result.route,
        "route_steps": result.route_steps,
        "route_valid": result.route_valid,
        "route_reaches_goal": result.route_reaches_goal,
        "route_correct": result.route_correct,
        "route_duration_min": result.route_duration_min,
        "runtime_sec": result.runtime_sec,
        "extra": result.extra,
        "verification": verification,
    }

    paths = {
        "summary": condition_dir / "summary.json",
        "turns": condition_dir / "turns.jsonl",
        "speech": condition_dir / "speech_pipeline.jsonl",
        "timing": condition_dir / "timing.jsonl",
        "agent_turn_segments": condition_dir / "agent_turn_segments.jsonl",
        "agent_a_segments": condition_dir / "agent_a_turn_segments.jsonl",
        "agent_b_segments": condition_dir / "agent_b_turn_segments.jsonl",
        "agent_timing_summary": condition_dir / "agent_timing_summary.json",
        "semantic": condition_dir / "semantic_parsing.jsonl",
        "runtime_events": condition_dir / "runtime_events.jsonl",
        "candidate_routes": condition_dir / "candidate_routes.jsonl",
        "retrospective_summary": condition_dir / "retrospective_summary.txt",
        "verification": condition_dir / "verification.json",
    }
    paths["summary"].write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    write_jsonl(paths["turns"], turns)
    write_jsonl(paths["speech"], speech_turns)
    write_jsonl(paths["timing"], timing_turns)
    write_jsonl(paths["agent_turn_segments"], agent_turn_segments)
    write_jsonl(paths["agent_a_segments"], [row for row in agent_turn_segments if row.get("speaker") == "Agent A"])
    write_jsonl(paths["agent_b_segments"], [row for row in agent_turn_segments if row.get("speaker") == "Agent B"])
    paths["agent_timing_summary"].write_text(json.dumps(agent_timing_summary, indent=2, ensure_ascii=True), encoding="utf-8")
    write_jsonl(paths["semantic"], nlu_turns)
    write_jsonl(paths["runtime_events"], runtime_events)
    write_jsonl(paths["candidate_routes"], candidate_events)
    paths["retrospective_summary"].write_text(result.metrics_text or "", encoding="utf-8")
    paths["verification"].write_text(json.dumps(verification, indent=2, ensure_ascii=True), encoding="utf-8")
    return paths


def write_single_run_research_outputs(result, scenario, output_dir, *, run_dir=None):
    """Write protocol files plus compiled metric files for one interactive run."""
    run_dir = Path(run_dir) if run_dir is not None else create_execution_run_dir(
        output_dir,
        label=f"single_{result.condition_id}",
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    protocol_root = run_dir / "conversation_protocol"
    protocol_paths = write_conversation_protocol(result, protocol_root)
    metric = MetricComputer().compute(result, scenario)
    compiled_dir = run_dir / "compiled_metrics"
    metrics_file = compiled_dir / "metrics.xlsx"
    phase_log_dir = compiled_dir / "metrics_by_phase"
    retrospective_json = compiled_dir / "retrospective_metrics.json"
    write_metrics_file([metric], metrics_file)
    write_metric_phase_logs([metric], phase_log_dir)
    network_paths = write_network_research_artifacts(
        scenario["start_time_min"],
        run_dir / "network",
        picture_dir=run_dir / "network_graphs",
    )
    retrospective_json.write_text(json.dumps(metric.as_dict(), indent=2, ensure_ascii=True), encoding="utf-8")
    manifest = {
        "run_id": run_dir.name,
        "condition_id": result.condition_id,
        "test_case_key": result.test_case_key,
        "persona_key": result.persona_key,
        "scenario_key": result.scenario_key,
        "speech_pattern_key": result.speech_pattern_key,
        "model_name": result.model_name,
        "output_layout": {
            "conversation_protocol": str(protocol_root),
            "compiled_metrics": str(compiled_dir),
            "network": str(network_paths["network_json"].parent),
            "network_graphs": str(network_paths["network_graph"].parent),
        },
    }
    run_manifest = run_dir / "run_manifest.json"
    run_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    return {
        "run_dir": run_dir,
        "run_manifest": run_manifest,
        "protocol": protocol_paths,
        "metrics_file": metrics_file,
        "phase_log_dir": phase_log_dir,
        "retrospective_json": retrospective_json,
        "network_json": network_paths["network_json"],
        "network_graph": network_paths["network_graph"],
    }


def write_conversation_protocols(results, output_dir):
    """Write protocol artifacts for all completed conversations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [write_conversation_protocol(result, output_dir) for result in results]
    index_rows = [
        {
            "condition_id": result.condition_id,
            "test_case_key": result.test_case_key,
            "persona_key": result.persona_key,
            "scenario_key": result.scenario_key,
            "message_count": len(result.conversation),
            "route_correct": result.route_correct,
            "folder": safe_artifact_name(result.condition_id),
        }
        for result in results
    ]
    write_jsonl(output_dir / "index.jsonl", index_rows)
    return paths


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def verify_conversation_protocol(result, turns, speech_turns, timing_turns, nlu_turns, agent_turn_segments=None, agent_timing_summary=None, runtime_events=None):
    """Return validation checks for research-grade protocol completeness."""
    agent_turn_segments = agent_turn_segments or []
    agent_timing_summary = agent_timing_summary or {}
    runtime_events = runtime_events or []
    checks = {
        "conversation_has_turns": bool(turns),
        "message_count_matches_result": len(turns) == result.extra.get("messages", len(result.conversation)),
        "route_flags_are_boolean": all(isinstance(value, bool) for value in (result.route_valid, result.route_reaches_goal, result.route_correct)),
        "speech_turns_have_transcripts": all(
            {"generated_text", "outgoing_text", "incoming_transcript"}.issubset(turn)
            for turn in speech_turns
        ),
        "speech_turns_have_pipeline_status": all(
            {"mode", "pipeline_ok"}.issubset(turn)
            for turn in speech_turns
        ),
        "timing_turns_have_latency": all("turn_latency_sec" in turn for turn in timing_turns),
        "timing_turns_have_elapsed": all(
            "turn_elapsed_sec" in turn or "turn_latency_sec" in turn
            for turn in timing_turns
        ),
        "semantic_turns_have_parse_flags": all("route_valid" in turn and "route_reaches_goal" in turn for turn in nlu_turns),
        "agent_segments_have_timing": all(
            {"turn", "speaker", "turn_elapsed_sec"}.issubset(turn)
            for turn in agent_turn_segments
        ),
        "agent_timing_summary_present": bool(agent_timing_summary) if agent_turn_segments else True,
        "runtime_events_present": bool(runtime_events),
        "preflight_viability_present": bool(result.extra.get("preflight_viability")),
        "retrospective_outcome_present": result.extra.get("conversation_outcome") in {"satisfied", "semi_satisfied", "unsatisfied"},
    }
    return {
        "verified": all(checks.values()),
        "checks": checks,
        "counts": {
            "turns": len(turns),
            "speech_turns": len(speech_turns),
            "timing_turns": len(timing_turns),
            "semantic_turns": len(nlu_turns),
            "agent_turn_segments": len(agent_turn_segments),
            "runtime_events": len(runtime_events),
        },
    }


def write_experiment_manifest(
    conditions,
    output_dir,
    *,
    num_turns,
    speech_engine,
    speech_scope,
    agent_b_plugin,
    tts_engine=None,
    asr_engine=None,
):
    """Write a scientific-method manifest describing the experiment design."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "condition_id": condition.condition_id,
            "test_case_key": condition.test_case_key,
            "persona_key": condition.persona_key,
            "scenario_key": condition.scenario_key,
            "speech_pattern_key": condition.speech_pattern_key,
            "tts_engine": tts_engine or speech_engine,
            "asr_engine": asr_engine or speech_engine,
            "model_param_key": condition.model_param_key,
            "objective_mode": getattr(condition, "objective_mode", None),
            "iteration": condition.iteration,
        }
        for condition in conditions
    ]
    manifest = {
        "research_objective": "Measure whether two cooperative route-planning agents converge on valid, constraint-fitting routes through natural speech dialog.",
        "hypotheses": [
            "Valid route proposals should appear before secondary optimization.",
            "Persona constraints should change route comparisons across time, fullness, transfers, and delay risk.",
            "Speech-enabled runs should preserve route semantics while adding measurable automatic speech recognition, text-to-speech, and runtime phases.",
        ],
        "independent_variables": [
            "test_case_key",
            "scenario_key",
            "persona_key",
            "speech_pattern_key",
            "tts_engine",
            "asr_engine",
            "model_param_key",
            "objective_mode",
            "agent_b_plugin",
        ],
        "dependent_metrics": [
            "route_valid",
            "route_reaches_goal",
            "route_duration_min",
            "candidate_route_count",
            "route_revision_count",
            "automatic_eval_score",
            "asr_word_error_rate",
            "runtime_response_latency_sec",
        ],
        "controls": {
            "num_turns": num_turns,
            "speech_engine": speech_engine,
            "tts_engine": tts_engine or speech_engine,
            "asr_engine": asr_engine or speech_engine,
            "speech_scope": speech_scope,
            "agent_b_plugin": agent_b_plugin,
        },
        "conditions": rows,
    }
    path = output_dir / "experiment_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    return path
