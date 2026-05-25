"""Research artifact writers for model, network, and metric analysis data."""
from __future__ import annotations

from dataclasses import asdict
import csv
import json
from pathlib import Path
import re

from minillama.evaluation.xlsx_export import write_metrics_xlsx
from minillama.evaluation.metrics import MetricComputer
from minillama.model.network_overview import build_network_overview
from minillama.model.network_picture import write_network_svg


def safe_artifact_name(value):
    """Return a stable filesystem name for research artifacts."""
    text = str(value or "run").strip().lower()
    text = re.sub(r"[^a-z0-9_.-]+", "_", text)
    return text.strip("._") or "run"


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
    verification = verify_conversation_protocol(result, turns, speech_turns, timing_turns, nlu_turns)
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
        "semantic": condition_dir / "semantic_parsing.jsonl",
        "metric_snapshots": condition_dir / "metric_snapshots.jsonl",
        "metrics": condition_dir / "metrics.txt",
        "verification": condition_dir / "verification.json",
    }
    paths["summary"].write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    write_jsonl(paths["turns"], turns)
    write_jsonl(paths["speech"], speech_turns)
    write_jsonl(paths["timing"], timing_turns)
    write_jsonl(paths["semantic"], nlu_turns)
    write_jsonl(paths["metric_snapshots"], result.extra.get("metric_snapshots", []))
    paths["metrics"].write_text(result.metrics_text or "", encoding="utf-8")
    paths["verification"].write_text(json.dumps(verification, indent=2, ensure_ascii=True), encoding="utf-8")
    return paths


def write_single_run_research_outputs(result, scenario, output_dir):
    """Write protocol files plus compiled metric files for one interactive run."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_paths = write_conversation_protocol(result, output_dir)
    metric = MetricComputer().compute(result, scenario)
    compiled_dir = output_dir / safe_artifact_name(result.condition_id) / "compiled_metrics"
    metrics_file = compiled_dir / "metrics.xlsx"
    phase_log_dir = compiled_dir / "metrics_by_phase"
    write_metrics_file([metric], metrics_file)
    write_metric_phase_logs([metric], phase_log_dir)
    return {
        "protocol": protocol_paths,
        "metrics_file": metrics_file,
        "phase_log_dir": phase_log_dir,
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


def verify_conversation_protocol(result, turns, speech_turns, timing_turns, nlu_turns):
    """Return validation checks for research-grade protocol completeness."""
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
        "semantic_turns_have_parse_flags": all("route_valid" in turn and "route_reaches_goal" in turn for turn in nlu_turns),
        "metric_snapshots_present": bool(result.extra.get("metric_snapshots")),
    }
    return {
        "verified": all(checks.values()),
        "checks": checks,
        "counts": {
            "turns": len(turns),
            "speech_turns": len(speech_turns),
            "timing_turns": len(timing_turns),
            "semantic_turns": len(nlu_turns),
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
