"""Research artifact writers for model, network, and metric analysis data."""
from __future__ import annotations

from dataclasses import asdict
import csv
from datetime import datetime
import json
from pathlib import Path
import wave

from coop_navigation_sds.Configuration.schema import (
    RAW_TRACE_COLLECTIONS,
    RESULT_SCHEMA_VERSION,
    TRACE_SCHEMA_VERSION,
    RunArtifactPaths,
    safe_artifact_name,
)
from coop_navigation_sds.Configuration.run_identity import naming_scheme_document, write_naming_scheme
from coop_navigation_sds.ResultsAndArtifacts.xlsx import write_metrics_xlsx
from coop_navigation_sds.ResultsAndArtifacts.metric_tables import write_metric_long_exports
from coop_navigation_sds.EvaluationMetrics.metrics import (
    METRIC_FAMILY_SPECS,
    MetricComputer,
    failure_indicator_analysis,
)
from coop_navigation_sds.EvaluationMetrics.catalog import (
    global_metric_key,
    metric_local_name,
    metric_metadata,
    phase_key,
)
from coop_navigation_sds.TransportNetwork.overview import build_network_overview
from coop_navigation_sds.TransportNetwork.picture import write_network_svg
from coop_navigation_sds.TransportNetwork.routes import route_path_text_from_steps, route_step_details


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


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
    write_naming_scheme(base_dir)
    run_id = execution_run_id(label, timestamp)
    candidate = base_dir / run_id
    suffix = 2
    while candidate.exists():
        candidate = base_dir / f"{run_id}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _metric_export_context(output_dir, scope):
    return {
        "result_scope": scope,
        "result_run_id": Path(output_dir).name,
    }


def write_metrics_csv(metrics, path, context=None):
    """Write flat metric rows as CSV."""
    records = list(metrics)
    if not records:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if context is None:
        rows = [record.as_dict() for record in records]
    else:
        context_values = _metric_export_context(path.parent, "run")
        context_values.update(dict(context or {}))
        rows = [{**context_values, **record.as_dict()} for record in records]
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_metrics_file(metrics, path, context=None):
    """Write metrics as XLSX or CSV based on the requested file extension."""
    path = Path(path)
    records = list(metrics)
    if path.suffix.lower() == ".xlsx":
        write_metrics_xlsx(records, path, context=context)
    else:
        write_metrics_csv(records, path, context=context)


def write_metric_phase_logs(metrics, output_dir, *, result_scope="run"):
    """Write calculated phase metrics to one JSONL file and a matching catalog."""
    records = list(metrics)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    export_context = _metric_export_context(output_dir, result_scope)
    metric_path = output_dir / "metrics_by_phase.jsonl"
    emitted_by_phase = {}
    with metric_path.open("w", encoding="utf-8") as handle:
        for record in records:
            identifiers = {
                **export_context,
                "condition_id": record.condition_id,
                "test_case_key": record.test_case_key,
                "persona_key": record.persona_key,
                "scenario_key": record.scenario_key,
                "speech_pattern_key": record.speech_pattern_key,
                "agent_a_audio_persona": getattr(record, "agent_a_audio_persona", "unknown"),
                "agent_b_audio_persona": getattr(record, "agent_b_audio_persona", "unknown"),
                "model_name": record.model_name,
                "model_param_key": record.model_param_key,
            }
            known_order = [phase_key(family) for family in METRIC_FAMILY_SPECS]
            ordered_phases = [
                phase for phase in known_order if phase in record.metric_families
            ] + [
                phase for phase in record.metric_families if phase not in known_order
            ]
            for phase in ordered_phases:
                phase_metrics = record.metric_families[phase]
                if not phase_metrics:
                    continue
                emitted_by_phase.setdefault(phase, set()).update(phase_metrics)
                calculations = {
                    name: getattr(record, "metric_calculations", {}).get(global_metric_key(phase, name), {})
                    for name in phase_metrics
                }
                handle.write(json.dumps({
                    **identifiers,
                    "phase": phase,
                    "metrics": phase_metrics,
                    "calculations": calculations,
                }, ensure_ascii=True) + "\n")
    catalog = {
        phase_key(family): {
                "order": family["order"],
                "title": family["title"],
                "description": family["description"],
                "metrics": [
                    {
                        **metric_metadata(key, phase_key(family)),
                        "label": label,
                    }
                    for key, label in family["metrics"]
                    if metric_local_name(key) in emitted_by_phase.get(phase_key(family), set())
                ],
            }
            for family in METRIC_FAMILY_SPECS
            if emitted_by_phase.get(phase_key(family))
        }
    catalog_path = output_dir / "metric_catalog.json"
    catalog_path.write_text(json.dumps(catalog, indent=2, ensure_ascii=True), encoding="utf-8")
    return {
        "phase_metrics": metric_path,
        "catalog": catalog_path,
        **write_metric_long_exports(records, output_dir, context=export_context),
    }


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
    picture_root = Path(picture_dir) if picture_dir is not None else output_dir
    graph_path = write_network_svg(picture_root / "network_graph.svg")
    return {"network_json": network_json, "network_graph": graph_path}


def write_conversation_transcript(path, result, turns, speech_turns, timing_turns, phase_timings=None):
    """Write the full human-readable dialog transcript for one conversation."""
    timing_by_turn = {turn.get("turn"): turn for turn in timing_turns if isinstance(turn, dict)}
    phase_timing_by_turn = {
        turn.get("turn"): turn
        for turn in (phase_timings or [])
        if isinstance(turn, dict)
    }
    lines = [
        "CoopNavigationSDS Conversation Transcript",
        f"Condition: {result.condition_id}",
        f"Test case: {result.test_case_key}",
        f"Persona: {result.persona_key}",
        f"Scenario: {result.scenario_key}",
        f"Speech pattern: {result.speech_pattern_key}",
        f"Agent A audio persona: {result.extra.get('agent_a_audio_persona', 'unknown')}",
        f"Agent B audio persona: {result.extra.get('agent_b_audio_persona', 'unknown')}",
        f"Model: {result.model_name}",
        f"Outcome: {result.extra.get('conversation_outcome', 'unknown')}",
        f"Route valid: {result.route_valid}",
        f"Route reaches goal: {result.route_reaches_goal}",
        f"Route correct: {result.route_correct}",
        f"Route duration minutes: {result.route_duration_min}",
        f"Runtime seconds: {result.runtime_sec}",
        "",
        "Conversation",
    ]
    for turn in turns:
        timing = timing_by_turn.get(turn["turn_index"], {})
        elapsed = timing.get("turn_elapsed_sec", timing.get("turn_latency_sec"))
        elapsed_text = f" [{elapsed:.3f} seconds]" if isinstance(elapsed, (int, float)) else ""
        lines.append(f"{turn['turn_index']:02d}. {turn['speaker']}{elapsed_text}: {turn['utterance']}")
        phases = phase_timing_by_turn.get(turn["turn_index"], {})
        if phases:
            lines.append(
                "    processing seconds: "
                f"generation={phases.get('natural_language_generation_sec')}, "
                f"synthesis={phases.get('text_to_speech_processing_sec')}, "
                f"recognition={phases.get('automatic_speech_recognition_processing_sec')}, "
                f"understanding={phases.get('natural_language_understanding_sec')}, "
                f"dialogue_management={phases.get('dialogue_management_sec')}"
            )
            lines.append(
                "    duration seconds: "
                f"audio={phases.get('audio_duration_sec')}, "
                f"speech_pipeline_wall={phases.get('speech_pipeline_wall_sec')}, "
                f"observed_turn={phases.get('observed_turn_sec')}, "
                f"accounted_processing={phases.get('accounted_processing_sec')}"
            )

    if speech_turns:
        lines.extend(["", "Speech Pipeline"])
        for index, turn in enumerate(speech_turns, start=1):
            speaker = turn.get("speaker", "unknown speaker")
            generated = turn.get("generated_text", "")
            outgoing = turn.get("outgoing_text", "")
            incoming = turn.get("incoming_transcript", "")
            raw = turn.get("raw_asr_transcript", incoming)
            misinterpretations = turn.get("misinterpreted_tokens", [])
            corrections = turn.get("transcript_corrections", [])
            audio = turn.get("audio") if isinstance(turn.get("audio"), dict) else {}
            audio_path = audio.get("path", "no audio file")
            lines.append(f"{index:02d}. {speaker}")
            lines.append(f"    generated: {generated}")
            lines.append(f"    outgoing speech text: {outgoing}")
            lines.append(f"    raw recognition: {raw}")
            lines.append(f"    misinterpreted tokens: {json.dumps(misinterpretations, ensure_ascii=True)}")
            lines.append(f"    transcript corrections: {json.dumps(corrections, ensure_ascii=True)}")
            lines.append(f"    understood transcript: {incoming}")
            lines.append(f"    audio file: {audio_path}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_conversation_wav(path, speech_turns, gap_sec=0.15):
    """Compile per-turn WAV artifacts into one conversation WAV when possible."""
    path = Path(path)
    sources = []
    for index, turn in enumerate(speech_turns, start=1):
        audio = turn.get("audio") if isinstance(turn, dict) and isinstance(turn.get("audio"), dict) else {}
        audio_path = audio.get("path")
        if audio_path:
            source_path = Path(audio_path)
            if source_path.suffix.lower() == ".wav" and source_path.exists():
                sources.append({
                    "turn_index": index,
                    "speaker": turn.get("speaker"),
                    "path": source_path,
                    "transcript_path": Path(audio["transcript_path"]) if audio.get("transcript_path") else None,
                })

    manifest = {
        "created": False,
        "path": str(path),
        "source_count": len(sources),
        "sources": [
            {
                "turn_index": source["turn_index"],
                "speaker": source["speaker"],
                "path": str(source["path"]),
                "transcript_path": str(source["transcript_path"]) if source.get("transcript_path") else None,
            }
            for source in sources
        ],
        "skipped": [],
        "removed_source_files": [],
        "duration_sec": 0.0,
    }
    if not sources:
        manifest["reason"] = "no_wav_turn_audio"
        return manifest

    path.parent.mkdir(parents=True, exist_ok=True)
    template = None
    for source in sources:
        try:
            with wave.open(str(source["path"]), "rb") as handle:
                template = {
                    "channels": handle.getnchannels(),
                    "sample_width": handle.getsampwidth(),
                    "frame_rate": handle.getframerate(),
                    "compression_type": handle.getcomptype(),
                    "compression_name": handle.getcompname(),
                }
                break
        except wave.Error as exc:
            manifest["skipped"].append({
                "path": str(source["path"]),
                "reason": f"cannot_read_wav: {exc}",
            })
    if template is None:
        manifest["reason"] = "no_readable_wav_turn_audio"
        return manifest

    frames_written = 0
    compatible_sources = 0
    with wave.open(str(path), "wb") as output:
        output.setnchannels(template["channels"])
        output.setsampwidth(template["sample_width"])
        output.setframerate(template["frame_rate"])
        output.setcomptype(template["compression_type"], template["compression_name"])
        silence_frames = int(max(gap_sec, 0.0) * template["frame_rate"])
        silence = b"\x00" * silence_frames * template["channels"] * template["sample_width"]
        for source in sources:
            try:
                with wave.open(str(source["path"]), "rb") as handle:
                    params = {
                        "channels": handle.getnchannels(),
                        "sample_width": handle.getsampwidth(),
                        "frame_rate": handle.getframerate(),
                        "compression_type": handle.getcomptype(),
                    }
                    if params != {key: template[key] for key in params}:
                        manifest["skipped"].append({
                            "path": str(source["path"]),
                            "reason": "wav_format_mismatch",
                            "format": params,
                        })
                        continue
                    frame_count = handle.getnframes()
                    output.writeframes(handle.readframes(frame_count))
                    frames_written += frame_count
                    compatible_sources += 1
                    if silence and source is not sources[-1]:
                        output.writeframes(silence)
                        frames_written += silence_frames
            except wave.Error as exc:
                manifest["skipped"].append({
                    "path": str(source["path"]),
                    "reason": f"cannot_read_wav: {exc}",
                })

    if compatible_sources == 0:
        try:
            path.unlink()
        except OSError:
            pass
        manifest["reason"] = "no_compatible_wav_turn_audio"
        return manifest
    manifest["created"] = True
    manifest["compatible_source_count"] = compatible_sources
    manifest["duration_sec"] = round(frames_written / float(template["frame_rate"]), 3)
    return manifest


def remove_compiled_turn_audio_sources(audio_manifest, roots):
    """Remove per-turn speech files after a complete conversation WAV is compiled."""
    if not audio_manifest.get("created"):
        return audio_manifest
    safe_roots = [Path(root).resolve() for root in roots if root]
    removed = []
    for source in audio_manifest.get("sources", []):
        for key in ("path", "transcript_path"):
            file_value = source.get(key)
            if not file_value:
                continue
            file_path = Path(file_value)
            try:
                resolved = file_path.resolve()
            except OSError:
                continue
            if not any(is_path_relative_to(resolved, root) for root in safe_roots):
                continue
            try:
                if file_path.exists():
                    file_path.unlink()
                    removed.append(str(file_path))
                    remove_empty_parent_dirs(file_path.parent, safe_roots)
            except OSError as exc:
                audio_manifest.setdefault("cleanup_errors", []).append({
                    "path": str(file_path),
                    "reason": str(exc),
                })
    audio_manifest["removed_source_files"] = removed
    return audio_manifest


def is_path_relative_to(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def remove_empty_parent_dirs(start, roots):
    """Remove empty generated audio folders without climbing outside safe roots."""
    current = Path(start)
    root_set = {Path(root).resolve() for root in roots}
    while True:
        try:
            resolved = current.resolve()
        except OSError:
            return
        if resolved in root_set or not any(is_path_relative_to(resolved, root) for root in root_set):
            return
        try:
            current.rmdir()
        except OSError:
            return
        current = current.parent


def build_combined_protocol(result, summary, turns, speech_turns, timing_turns, phase_timings, nlu_turns, agent_turn_segments, agent_timing_summary, runtime_events, candidate_events, verification, audio_manifest):
    """Return a single JSON document containing all protocol data for one run."""
    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "trace_collections": list(RAW_TRACE_COLLECTIONS),
        "summary": summary,
        "conversation": turns,
        "agent_memories": dict(result.extra.get("agent_memories", {})),
        "speech_pipeline": speech_turns,
        "timing": timing_turns,
        "phase_timing": phase_timings,
        "semantic_parsing": nlu_turns,
        "agent_turn_segments": agent_turn_segments,
        "agent_timing_summary": agent_timing_summary,
        "runtime_events": runtime_events,
        "candidate_routes": candidate_events,
        "prompt_audits": list(result.extra.get("prompt_audits", [])),
        "retrospective_summary": result.metrics_text or "",
        "verification": verification,
        "audio_manifest": audio_manifest,
    }


def build_metric_input_document(result, scenario):
    """Build the immutable evidence document used by retrospective metrics."""
    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "captured_before_metric_calculation": True,
        "condition_id": result.condition_id,
        "scenario": scenario,
        "result": asdict(result),
        "trace_collections": {
            "conversation": list(result.conversation),
            "agent_memories": dict(result.extra.get("agent_memories", {})),
            "speech_pipeline": list(result.extra.get("speech_turns", [])),
            "timing": list(result.extra.get("timing_turns", [])),
            "phase_timing": list(result.extra.get("phase_timings", [])),
            "semantic_parsing": list(result.extra.get("nlu_turns", [])),
            "runtime_events": list(result.extra.get("runtime_events", [])),
            "candidate_routes": list(result.extra.get("candidate_events", [])),
            "prompt_audits": list(result.extra.get("prompt_audits", [])),
        },
    }


def write_metric_inputs(result, scenario, path):
    """Persist raw run evidence before any derived metric is calculated."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            build_metric_input_document(result, scenario),
            indent=2,
            ensure_ascii=True,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    return path


def calculate_metrics_from_inputs(path):
    """Recalculate one metric record entirely from a stored evidence document."""
    from coop_navigation_sds.DialogManagement.result import DialogResult

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != TRACE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported metric-input schema {payload.get('schema_version')!r}; "
            f"expected {TRACE_SCHEMA_VERSION}."
        )
    result = DialogResult(**payload["result"])
    return MetricComputer().compute(result, payload["scenario"])


def write_batch_metric_inputs(results, path):
    """Persist every completed condition before calculating batch metrics."""
    documents = [
        build_metric_input_document(
            result,
            result.extra.get("resolved_scenario", {}),
        )
        for result in results
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": TRACE_SCHEMA_VERSION,
                "captured_before_metric_calculation": True,
                "condition_count": len(documents),
                "conditions": documents,
            },
            indent=2,
            ensure_ascii=True,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    return path


def calculate_batch_metrics_from_inputs(path):
    """Reconstruct batch metrics solely from persisted immutable evidence."""
    from coop_navigation_sds.DialogManagement.result import DialogResult

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != TRACE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported batch metric-input schema {payload.get('schema_version')!r}; "
            f"expected {TRACE_SCHEMA_VERSION}."
        )
    records = []
    for document in payload.get("conditions", []):
        result = DialogResult(**document["result"])
        record = MetricComputer().compute(result, document["scenario"])
        record.pair_id = str(result.extra.get("pair_id", ""))
        record.run_type = str(result.extra.get("run_type", "audio_variant"))
        records.append(record)
    return records


def build_retrospective_metrics_document(metrics, *, result_scope, input_inventory=None):
    """Return a common retrospective metric document for single and batch runs."""
    records = list(metrics)
    conditions = [
        {
            "condition_id": record.condition_id,
            "pair_id": getattr(record, "pair_id", ""),
            "run_type": getattr(record, "run_type", "audio_variant"),
            "test_case_key": record.test_case_key,
            "persona_key": record.persona_key,
            "scenario_key": record.scenario_key,
            "speech_pattern_key": record.speech_pattern_key,
            "agent_a_audio_persona": getattr(record, "agent_a_audio_persona", "unknown"),
            "agent_b_audio_persona": getattr(record, "agent_b_audio_persona", "unknown"),
            "model_name": record.model_name,
            "model_param_key": record.model_param_key,
            "metrics_by_phase": record.metric_families,
            "calculation_evidence": record.metric_calculations,
            "input_inventory": (input_inventory or {}).get(record.condition_id, {}),
        }
        for record in records
    ]
    document = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "result_scope": result_scope,
        "condition_count": len(conditions),
        "conditions": conditions,
    }
    if len(conditions) == 1:
        document.update({
            "condition_id": conditions[0]["condition_id"],
            "test_case_key": conditions[0]["test_case_key"],
            "scenario_key": conditions[0]["scenario_key"],
            "model_name": conditions[0]["model_name"],
            "metrics_by_phase": conditions[0]["metrics_by_phase"],
            "calculation_evidence": conditions[0]["calculation_evidence"],
            "input_inventory": conditions[0]["input_inventory"],
        })
    return document


def write_retrospective_metrics_json(metrics, path, *, result_scope, input_inventory=None):
    """Write common retrospective metric evidence for single or batch runs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_retrospective_metrics_document(
        metrics,
        result_scope=result_scope,
        input_inventory=input_inventory,
    )
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, default=_json_default),
        encoding="utf-8",
    )
    return path


def write_conversation_protocol(result, output_dir):
    """Write detailed, parsable protocol data for one completed conversation."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_prefix = safe_artifact_name(result.condition_id)

    turns = [
        {
            "turn_index": index,
            "speaker": speaker,
            "utterance": utterance,
        }
        for index, (speaker, utterance) in enumerate(result.conversation, start=1)
    ]
    speech_turns = list(result.extra.get("speech_turns", []))
    for turn in speech_turns:
        diagnostics = turn.get("diagnostics") if isinstance(turn.get("diagnostics"), dict) else {}
        turn.setdefault("raw_asr_transcript", diagnostics.get("raw_asr_transcript", turn.get("incoming_transcript", "")))
        turn.setdefault("misinterpreted_tokens", diagnostics.get("misinterpreted_tokens", []))
        turn.setdefault("transcript_corrections", diagnostics.get("transcript_corrections", []))
        turn.setdefault("agent_input_transcript", turn.get("incoming_transcript", ""))
    timing_turns = list(result.extra.get("timing_turns", []))
    phase_timings = list(result.extra.get("phase_timings", []))
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
    paths = RunArtifactPaths(output_dir, artifact_prefix).as_dict()
    write_conversation_transcript(
        paths["transcript_txt"],
        result,
        turns,
        speech_turns,
        timing_turns,
        phase_timings,
    )
    audio_manifest = write_conversation_wav(paths["conversation_wav"], speech_turns)
    audio_manifest = remove_compiled_turn_audio_sources(
        audio_manifest,
        [output_dir],
    )
    summary = {
        "condition_id": result.condition_id,
        "test_case_key": result.test_case_key,
        "persona_key": result.persona_key,
        "scenario_key": result.scenario_key,
        "speech_pattern_key": result.speech_pattern_key,
        "agent_a_audio_persona": result.extra.get("agent_a_audio_persona"),
        "agent_b_audio_persona": result.extra.get("agent_b_audio_persona"),
        "model_name": result.model_name,
        "message_count": len(turns),
        "route": result.route,
        "route_path": route_path_text_from_steps(result.route_steps),
        "route_steps": result.route_steps,
        "route_step_details": route_step_details(result.route_steps),
        "route_valid": result.route_valid,
        "route_reaches_goal": result.route_reaches_goal,
        "route_correct": result.route_correct,
        "route_duration_min": result.route_duration_min,
        "runtime_sec": result.runtime_sec,
        "extra": result.extra,
        "verification": verification,
        "compiled_artifacts": {
            "protocol_json": str(paths["protocol"]),
            "conversation_transcript_txt": str(paths["transcript_txt"]),
            "conversation_wav": str(paths["conversation_wav"]) if audio_manifest.get("created") else None,
        },
    }

    paths["protocol"].write_text(
        json.dumps(
            build_combined_protocol(
                result,
                summary,
                turns,
                speech_turns,
                timing_turns,
                phase_timings,
                nlu_turns,
                agent_turn_segments,
                agent_timing_summary,
                runtime_events,
                candidate_events,
                verification,
                audio_manifest,
            ),
            indent=2,
            ensure_ascii=True,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    return paths


def write_single_run_research_outputs(result, scenario, output_dir, *, run_dir=None):
    """Write protocol files plus compiled metric files for one interactive run."""
    run_dir = Path(run_dir) if run_dir is not None else create_execution_run_dir(
        output_dir,
        label=f"single_{result.condition_id}",
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact_prefix = safe_artifact_name(result.condition_id)
    canonical_paths = RunArtifactPaths(run_dir, artifact_prefix).as_dict()
    metric_inputs = write_metric_inputs(result, scenario, canonical_paths["metric_inputs"])
    metric = MetricComputer().compute(result, scenario)
    protocol_paths = write_conversation_protocol(result, run_dir)
    metrics_file = run_dir / "metrics.xlsx"
    phase_log_dir = run_dir
    retrospective_json = run_dir / "retrospective_metrics.json"
    export_context = _metric_export_context(run_dir, "single_run")
    write_metrics_file([metric], metrics_file, context=export_context)
    metric_exports = write_metric_phase_logs([metric], phase_log_dir, result_scope="single_run")
    network_paths = write_network_research_artifacts(
        scenario["start_time_min"],
        run_dir,
        picture_dir=run_dir,
    )
    write_retrospective_metrics_json(
        [metric],
        retrospective_json,
        result_scope="single_run",
        input_inventory={metric.condition_id: result.extra.get("metric_input_inventory", {})},
    )
    manifest = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "run_id": run_dir.name,
        "naming_scheme": naming_scheme_document(),
        "condition_id": result.condition_id,
        "test_case_key": result.test_case_key,
        "persona_key": result.persona_key,
        "scenario_key": result.scenario_key,
        "speech_pattern_key": result.speech_pattern_key,
        "agent_a_audio_persona": result.extra.get("agent_a_audio_persona"),
        "agent_b_audio_persona": result.extra.get("agent_b_audio_persona"),
        "resolved_audio_personas": result.extra.get("resolved_audio_personas", {}),
        "model_name": result.model_name,
        "model_backend": result.extra.get("model_backend", {}),
        "configuration": result.extra.get("resolved_run_config", {}),
        "configuration_provenance": result.extra.get("configuration_provenance", {}),
        "pipeline_contract": result.extra.get("pipeline_contract", {}),
        "runtime_environment": result.extra.get("runtime_environment", {}),
        "random_seed": result.extra.get("resolved_run_config", {}).get("network_seed"),
        "artifacts": {
            "metric_inputs": Path(metric_inputs).name,
            "protocol": Path(protocol_paths["protocol"]).name,
            "transcript": Path(protocol_paths["transcript_txt"]).name,
            "metrics": metrics_file.name,
            "metrics_by_phase": "metrics_by_phase.jsonl",
            "metric_catalog": "metric_catalog.json",
            "metrics_long_csv": Path(metric_exports["metric_long_csv"]).name,
            "metrics_long_jsonl": Path(metric_exports["metric_long_jsonl"]).name,
            "metrics_wide_csv": Path(metric_exports["metric_wide_csv"]).name,
            "metrics_wide_jsonl": Path(metric_exports["metric_wide_jsonl"]).name,
            "retrospective_metrics": retrospective_json.name,
        },
        "output_layout": {
            "layout": "flat",
            "naming_scheme": str(run_dir.parent / "naming_scheme.json"),
            "run_folder": str(run_dir),
            "network": str(network_paths["network_json"].parent),
            "network_graph": str(network_paths["network_graph"]),
        },
    }
    run_manifest = run_dir / "run_manifest.json"
    run_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    standard_summary = write_standard_run_summary(
        [result],
        [metric],
        run_dir,
        result_scope="single_run",
        manifest_path=run_manifest,
    )
    return {
        "run_dir": run_dir,
        "metric": metric,
        "run_manifest": run_manifest,
        "metric_inputs": metric_inputs,
        "protocol": protocol_paths,
        "metrics_file": metrics_file,
        "phase_log_dir": phase_log_dir,
        "retrospective_json": retrospective_json,
        "network_json": network_paths["network_json"],
        "network_graph": network_paths["network_graph"],
        "run_summary": standard_summary["summary"],
        "conditions": standard_summary["conditions"],
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
            "agent_a_audio_persona": result.extra.get("agent_a_audio_persona"),
            "agent_b_audio_persona": result.extra.get("agent_b_audio_persona"),
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


def write_standard_run_summary(
    results,
    metrics,
    output_dir,
    *,
    result_scope,
    manifest_path,
):
    """Write the common human/machine entry point for single and batch runs."""
    output_dir = Path(output_dir)
    results = list(results)
    metrics = list(metrics)
    metric_by_condition = {metric.condition_id: metric for metric in metrics}
    condition_rows = []
    for result in results:
        metric = metric_by_condition.get(result.condition_id)
        extras = result.extra or {}
        parameters = dict(extras.get("parameter_values") or {})
        condition_rows.append({
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "result_scope": result_scope,
            "result_run_id": output_dir.name,
            "condition_id": result.condition_id,
            "pair_id": extras.get("pair_id"),
            "run_type": extras.get("run_type", "audio_variant"),
            "test_case_key": result.test_case_key,
            "scenario_key": result.scenario_key,
            "persona_key": result.persona_key,
            "speech_pattern_key": result.speech_pattern_key,
            "agent_a_type": extras.get("agent_a_type") or extras.get("resolved_run_config", {}).get("agent_a_type"),
            "agent_a_audio_persona": extras.get("agent_a_audio_persona"),
            "agent_b_audio_persona": extras.get("agent_b_audio_persona"),
            "agent_b_plugin": extras.get("agent_b_plugin") or extras.get("resolved_run_config", {}).get("agent_b_plugin"),
            "agent_b_model": extras.get("agent_b_model") or result.model_name,
            "agent_b_llm_size": parameters.get("agent_b_llm_size"),
            "agent_b_model_role": parameters.get("agent_b_model_role"),
            "model_param_key": extras.get("model_param_key"),
            "objective_mode": extras.get("objective_mode"),
            "iteration": extras.get("iteration"),
            "tts_engine": extras.get("tts_engine") or extras.get("resolved_run_config", {}).get("tts_engine"),
            "asr_engine": extras.get("asr_engine") or extras.get("resolved_run_config", {}).get("asr_engine"),
            "configured_tts_engine": extras.get("configured_tts_engine") or extras.get("tts_engine"),
            "configured_asr_engine": extras.get("configured_asr_engine") or extras.get("asr_engine"),
            "asr_search_width": parameters.get("asr_beam_size"),
            "matrix_family": parameters.get("matrix_family"),
            "experiment_platform": parameters.get("experiment_platform"),
            "experiment_seed": parameters.get("experiment_seed", parameters.get("network_seed")),
            "repetition": parameters.get("repetition", extras.get("iteration")),
            "run_mode": parameters.get("run_mode"),
            "slurm_condition_index": parameters.get("slurm_condition_index"),
            "slurm_grid_name": parameters.get("slurm_grid_name"),
            "configuration_fingerprint": extras.get("configuration_provenance", {}).get("fingerprint_sha256"),
            "route_valid": bool(result.route_valid),
            "route_reaches_goal": bool(result.route_reaches_goal),
            "route_correct": bool(result.route_correct),
            "route_duration_min": result.route_duration_min,
            "turn_count": int(extras.get("messages", len(result.conversation))),
            "runtime_sec": result.runtime_sec,
            "automatic_eval_score": getattr(metric, "automatic_eval_score", None),
            "task_success": getattr(metric, "success", bool(result.route_correct)),
        })
    conditions_path = output_dir / "conditions.jsonl"
    write_jsonl(conditions_path, condition_rows)
    artifact_rows = []
    for path in sorted(item for item in output_dir.iterdir() if item.is_file()):
        artifact_rows.append({
            "path": path.name,
            "bytes": path.stat().st_size,
            "extension": path.suffix.lower(),
        })
    summary = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "result_scope": result_scope,
        "result_run_id": output_dir.name,
        "condition_count": len(condition_rows),
        "successful_condition_count": sum(bool(row["task_success"]) for row in condition_rows),
        "configuration_fingerprints": sorted({
            row["configuration_fingerprint"]
            for row in condition_rows
            if row["configuration_fingerprint"]
        }),
        "manifest": Path(manifest_path).name,
        "condition_table": conditions_path.name,
        "recommended_analysis_tables": [
            "metrics_long.csv",
            "metrics_wide.csv",
            conditions_path.name,
        ],
        "artifacts": artifact_rows,
    }
    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return {"summary": summary_path, "conditions": conditions_path}


def write_failure_indicator_report(metrics, path, minimum_per_outcome=3):
    """Persist leakage-controlled exploratory failure thresholds for a batch."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = failure_indicator_analysis(
        metrics,
        minimum_per_outcome=minimum_per_outcome,
    )
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=True, default=_json_default),
        encoding="utf-8",
    )
    return path


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
            {
                "generated_text", "outgoing_text", "raw_asr_transcript",
                "misinterpreted_tokens", "transcript_corrections",
                "incoming_transcript", "agent_input_transcript",
            }.issubset(turn)
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
    metrics_filename="metrics.xlsx",
    configuration=None,
):
    """Write a scientific-method manifest describing the experiment design."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_naming_scheme(output_dir)
    rows = [
        {
            "condition_id": condition.condition_id,
            "test_case_key": condition.test_case_key,
            "persona_key": condition.persona_key,
            "scenario_key": condition.scenario_key,
            "speech_pattern_key": condition.speech_pattern_key,
            "agent_a_audio_persona": getattr(condition, "agent_a_audio_persona", None),
            "agent_b_audio_persona": getattr(condition, "agent_b_audio_persona", None),
            "tts_engine": getattr(condition, "tts_engine", None) or tts_engine or speech_engine,
            "asr_engine": getattr(condition, "asr_engine", None) or asr_engine or speech_engine,
            "agent_b_model": getattr(condition, "agent_b_model", None),
            "pair_id": getattr(condition, "pair_id", None),
            "run_type": getattr(condition, "run_type", "audio_variant"),
            "model_param_key": condition.model_param_key,
            "objective_mode": getattr(condition, "objective_mode", None),
            "iteration": condition.iteration,
            "parameter_values": dict(getattr(condition, "parameter_values", {}) or {}),
        }
        for condition in conditions
    ]
    configuration = dict(configuration or {})
    manifest = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "naming_scheme": naming_scheme_document(),
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
            "agent_a_audio_persona",
            "agent_b_audio_persona",
            "tts_engine",
            "asr_engine",
            "model_param_key",
            "objective_mode",
            "agent_b_plugin",
            "agent_b_model",
            "parameter_values.profile_key",
            "parameter_values.agent_b_llm_size",
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
            "configuration": configuration,
        },
        "configuration_provenance": configuration.get("configuration_provenance", {}),
        "pipeline_contract": configuration.get("pipeline_contract", {}),
        "analysis_artifacts": {
            "wide_workbook": metrics_filename,
            "phase_jsonl": "metrics_by_phase.jsonl",
            "long_csv": "metrics_long.csv",
            "long_jsonl": "metrics_long.jsonl",
            "wide_csv": "metrics_wide.csv",
            "wide_jsonl": "metrics_wide.jsonl",
            "metric_catalog": "metric_catalog.json",
            "raw_metric_inputs": "metric_inputs.json",
            "retrospective_metrics": "retrospective_metrics.json",
            "failure_indicators": "failure_indicators.json",
        },
        "conditions": rows,
    }
    path = output_dir / "experiment_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    return path
