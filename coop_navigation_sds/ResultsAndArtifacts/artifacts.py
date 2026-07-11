"""Research artifact writers for model, network, and metric analysis data."""
from __future__ import annotations

from dataclasses import asdict
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import wave

from coop_navigation_sds.Configuration.schema import (
    CANONICAL_RESULT_FILES,
    RAW_TRACE_COLLECTIONS,
    RESULT_FILES,
    RESULT_SCHEMA_VERSION,
    TRACE_SCHEMA_VERSION,
    RunArtifactPaths,
    safe_artifact_name,
)
from coop_navigation_sds.Configuration.run_identity import naming_scheme_document, write_naming_scheme
from coop_navigation_sds.ResultsAndArtifacts.xlsx import write_metrics_xlsx
from coop_navigation_sds.ResultsAndArtifacts.metric_tables import write_metric_long_exports
from coop_navigation_sds.EvaluationMetrics.metrics import (
    MetricComputer,
    failure_indicator_analysis,
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
    while True:
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            candidate = base_dir / f"{run_id}_{suffix:02d}"
            suffix += 1


def _metric_export_context(output_dir, scope):
    return {
        "result_scope": scope,
        "result_run_id": Path(output_dir).name,
    }


def _artifact_role(path):
    """Classify a result artifact by its research function."""
    name = path.name
    if name in {
        RESULT_FILES["conditions"],
        RESULT_FILES["metric_inputs"], RESULT_FILES["protocols"],
        RESULT_FILES["transcripts"], RESULT_FILES["runtime_events"],
        RESULT_FILES["network_data"], RESULT_FILES["network_graph"],
    } or name.endswith("_conversation.wav"):
        return "evidence"
    if name in {RESULT_FILES["metrics_long"], RESULT_FILES["metrics_wide"]} or name.endswith(".xlsx"):
        return "metrics"
    if name.endswith("manifest.json") or name in {
        "experiment_job.json", "naming_scheme.json", "coverage_plan.json",
        "condition_configuration_breakdown.csv",
    }:
        return "configuration"
    if name.endswith(".html") or name in {
        RESULT_FILES["condition_analysis"], RESULT_FILES["run_analysis"],
        RESULT_FILES["phase_scorecard"], RESULT_FILES["performance_band_summary"],
    }:
        return "analysis"
    if name.endswith(".log") or "failure" in name or "warning" in name:
        return "diagnostic"
    return "support"


def build_artifact_inventory(output_dir):
    """Return a hashed, role-aware inventory for an otherwise flat run folder."""
    output_dir = Path(output_dir)
    excluded = {RESULT_FILES["summary"], RESULT_FILES["artifact_inventory"]}
    rows = []
    for path in sorted(item for item in output_dir.iterdir() if item.is_file()):
        if path.name in excluded:
            continue
        role = _artifact_role(path)
        rows.append({
            "path": path.name,
            "role": role,
            "canonical": path.name in CANONICAL_RESULT_FILES or path.name.endswith("_conversation.wav"),
            "regenerable": role == "analysis",
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        })
    return rows


def write_artifact_inventory(output_dir):
    """Write the integrity and provenance index for all run artifacts."""
    output_dir = Path(output_dir)
    rows = build_artifact_inventory(output_dir)
    payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "run_id": output_dir.name,
        "artifact_count": len(rows),
        "canonical_count": sum(bool(row["canonical"]) for row in rows),
        "artifacts": rows,
    }
    path = output_dir / RESULT_FILES["artifact_inventory"]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return path


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
    """Write canonical long and wide retrospective metric tables."""
    records = list(metrics)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    export_context = _metric_export_context(output_dir, result_scope)
    return write_metric_long_exports(records, output_dir, context=export_context)


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
                    "clean_path": Path(audio["clean_path"]) if audio.get("clean_path") else None,
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
                "clean_path": str(source["clean_path"]) if source.get("clean_path") else None,
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
        for key in ("path", "clean_path", "transcript_path"):
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


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def consolidate_completed_conversation_artifacts(output_dir, artifact_paths, *, remove_sources=True):
    """Combine per-condition protocol/TXT files without combining condition audio."""
    output_dir = Path(output_dir)
    combined_protocol = output_dir / "conversation_protocols.jsonl"
    combined_transcript = output_dir / "conversation_transcripts.txt"
    protocol_records = []
    transcript_lines = []
    compact_paths = []
    source_paths = []
    for record_index, paths in enumerate(artifact_paths, start=1):
        protocol_path = Path(paths["protocol"])
        transcript_path = Path(paths["transcript_txt"])
        if not protocol_path.is_file() or not transcript_path.is_file():
            raise RuntimeError(
                f"Cannot consolidate missing conversation artifacts: {protocol_path}, {transcript_path}"
            )
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        condition_id = str(protocol.get("summary", {}).get("condition_id") or protocol_path.stem)
        transcript = transcript_path.read_text(encoding="utf-8").rstrip()
        line_start = len(transcript_lines) + 1
        transcript_lines.extend((f"=== Conversation {record_index}: {condition_id} ===", *transcript.splitlines()))
        line_end = len(transcript_lines)
        transcript_lines.append("")
        protocol.setdefault("summary", {}).setdefault("compiled_artifacts", {}).update({
            "protocol_json": str(combined_protocol),
            "protocol_jsonl": str(combined_protocol),
            "protocol_record": record_index,
            "conversation_transcript_txt": str(combined_transcript),
            "transcript_line_start": line_start,
            "transcript_line_end": line_end,
        })
        protocol["artifact_compaction"] = {
            "schema_version": 1,
            "protocol_record": record_index,
            "transcript_line_start": line_start,
            "transcript_line_end": line_end,
            "source_protocol_sha256": _sha256_file(protocol_path),
            "source_transcript_sha256": _sha256_file(transcript_path),
        }
        protocol_records.append(protocol)
        source_paths.extend((protocol_path, transcript_path))
        compact_paths.append({
            **paths,
            "protocol": combined_protocol,
            "transcript_txt": combined_transcript,
            "protocol_record": record_index,
            "transcript_line_start": line_start,
            "transcript_line_end": line_end,
        })

    protocol_temp = Path(f"{combined_protocol}.tmp")
    transcript_temp = Path(f"{combined_transcript}.tmp")
    protocol_temp.write_text(
        "".join(json.dumps(record, ensure_ascii=True, default=_json_default) + "\n" for record in protocol_records),
        encoding="utf-8",
    )
    transcript_temp.write_text("\n".join(transcript_lines).rstrip() + "\n", encoding="utf-8")
    verified_records = [
        json.loads(line)
        for line in protocol_temp.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(verified_records) != len(protocol_records) or not transcript_temp.read_text(encoding="utf-8").strip():
        raise RuntimeError("Conversation artifact consolidation verification failed.")
    protocol_temp.replace(combined_protocol)
    transcript_temp.replace(combined_transcript)
    if remove_sources:
        for source in source_paths:
            if source not in {combined_protocol, combined_transcript} and source.exists():
                source.unlink()
    return compact_paths


def _structured_log_events(path):
    events = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL evidence at {path}:{line_number}") from exc
    return events


def consolidate_completed_runtime_logs(output_dir):
    """Combine condition session logs after completion and retain source checksums."""
    output_dir = Path(output_dir)
    event_sources = sorted(output_dir.glob("batch-*.jsonl"))
    text_sources = sorted(output_dir.glob("batch-*.log"))
    summary_sources = sorted(output_dir.glob("batch-*-summary.json"))
    event_records = []
    summary_records = []
    text_sections = []
    for source in event_sources:
        checksum = _sha256_file(source)
        event_records.extend({
            "source_session_file": source.name,
            "source_sha256": checksum,
            "event": event,
        } for event in _structured_log_events(source))
    for source in summary_sources:
        summary_records.append({
            "source_session_file": source.name,
            "source_sha256": _sha256_file(source),
            "summary": json.loads(source.read_text(encoding="utf-8")),
        })
    for source in text_sources:
        text_sections.extend((
            f"=== {source.name} | sha256={_sha256_file(source)} ===",
            source.read_text(encoding="utf-8").rstrip(),
            "",
        ))
    outputs = {
        "events": output_dir / "runtime_events.jsonl",
        "summaries": output_dir / "runtime_sessions.jsonl",
        "text": output_dir / "runtime.log",
    }
    temporary = {name: Path(f"{path}.tmp") for name, path in outputs.items()}
    write_jsonl(temporary["events"], event_records)
    write_jsonl(temporary["summaries"], summary_records)
    temporary["text"].write_text("\n".join(text_sections).rstrip() + "\n", encoding="utf-8")
    if len(_structured_log_events(temporary["events"])) != len(event_records):
        raise RuntimeError("Runtime event consolidation verification failed.")
    if len(_structured_log_events(temporary["summaries"])) != len(summary_records):
        raise RuntimeError("Runtime summary consolidation verification failed.")
    if text_sources and not temporary["text"].read_text(encoding="utf-8").strip():
        raise RuntimeError("Runtime text-log consolidation verification failed.")
    for name, path in outputs.items():
        temporary[name].replace(path)
    for source in (*event_sources, *summary_sources, *text_sources):
        source.unlink()
    provider_log = consolidate_provider_runtime_logs(output_dir)
    return {
        **outputs,
        "provider": provider_log,
        "source_event_file_count": len(event_sources),
        "source_summary_file_count": len(summary_sources),
        "source_text_file_count": len(text_sources),
        "event_count": len(event_records),
    }


def consolidate_provider_runtime_logs(output_dir):
    """Flatten provider diagnostics while preserving source identity and bytes."""
    output_dir = Path(output_dir)
    sources = sorted(output_dir.glob(".turn_audio/*/provider_*.log"))
    if not sources:
        return output_dir / "provider_runtime.log"
    sections = []
    for source in sources:
        sections.extend((
            f"=== {source.relative_to(output_dir)} | sha256={_sha256_file(source)} ===",
            source.read_text(encoding="utf-8", errors="replace").rstrip(),
            "",
        ))
    destination = output_dir / "provider_runtime.log"
    destination.write_text("\n".join(sections).rstrip() + "\n", encoding="utf-8")
    if not destination.read_text(encoding="utf-8").strip():
        raise RuntimeError("Provider-log consolidation verification failed.")
    turn_audio_root = (output_dir / ".turn_audio").resolve()
    for source in sources:
        source.unlink()
        remove_empty_parent_dirs(source.parent, [turn_audio_root])
    if turn_audio_root.is_dir() and not any(turn_audio_root.iterdir()):
        turn_audio_root.rmdir()
    return destination


def compact_incomplete_run_artifacts(run_dir):
    """Create readable combined artifacts for an interrupted run without deleting evidence."""
    run_dir = Path(run_dir)
    logs = sorted(run_dir.glob("batch-*.jsonl"))
    if not logs:
        return {"status": "no_structured_condition_logs", "run_dir": str(run_dir)}
    protocol_path = run_dir / "partial_conversation_protocols.jsonl"
    transcript_path = run_dir / "partial_conversation_transcripts.txt"
    records = []
    transcript_lines = []
    audio_created = 0
    recovered_conditions = []
    source_owners = {}
    for log_path in logs:
        events = _structured_log_events(log_path)
        start_event = next(
            (
                event for event in events
                if event.get("name") == "batch.condition.start"
                or (
                    event.get("kind") == "program.segment"
                    and event.get("payload", {}).get("segment") == "batch.condition"
                    and event.get("payload", {}).get("phase") == "start"
                )
            ),
            {},
        )
        condition_id = str(
            start_event.get("payload", {}).get("condition_id")
            or log_path.stem.removeprefix("batch-")
        )
        speech_turns = [
            event.get("payload", {})
            for event in events
            if event.get("name") == "telemetry.speech" and isinstance(event.get("payload"), dict)
        ]
        completed = any(
            event.get("name") == "batch.condition.end"
            or (
                event.get("kind") == "program.segment"
                and event.get("payload", {}).get("segment") == "batch.condition"
                and event.get("payload", {}).get("phase") == "end"
            )
            for event in events
        )
        recovered_conditions.append({
            "condition_id": condition_id,
            "log_path": log_path,
            "speech_turns": speech_turns,
            "completed": completed,
        })
        for turn in speech_turns:
            source_path = str(turn.get("audio", {}).get("path") or "")
            if source_path:
                source_owners.setdefault(source_path, set()).add(condition_id)

    for record_index, recovered in enumerate(recovered_conditions, start=1):
        condition_id = recovered["condition_id"]
        log_path = recovered["log_path"]
        speech_turns = recovered["speech_turns"]
        completed = recovered["completed"]
        section = [f"=== Recovered conversation {record_index}: {condition_id} ==="]
        for turn in speech_turns:
            section.extend((
                f"Turn {turn.get('turn', '?')} | {turn.get('speaker', 'unknown')}",
                f"  Intended: {turn.get('generated_text', '')}",
                f"  TTS speech: {turn.get('outgoing_text', '')}",
                f"  ASR raw: {turn.get('raw_asr_transcript', turn.get('incoming_transcript', ''))}",
                f"  Understood: {turn.get('agent_input_transcript', turn.get('incoming_transcript', ''))}",
                f"  Misinterpretations: {json.dumps(turn.get('misinterpreted_tokens', []), ensure_ascii=True)}",
                f"  Corrections: {json.dumps(turn.get('transcript_corrections', []), ensure_ascii=True)}",
            ))
        section.append("")
        line_start = len(transcript_lines) + 1
        transcript_lines.extend(section)
        line_end = len(transcript_lines) - 1
        conversation_wav = run_dir / f"{safe_artifact_name(condition_id)}_partial_conversation.wav"
        ambiguous_audio = sorted({
            str(turn.get("audio", {}).get("path") or "")
            for turn in speech_turns
            if len(source_owners.get(str(turn.get("audio", {}).get("path") or ""), ())) > 1
        })
        if ambiguous_audio:
            conversation_wav.unlink(missing_ok=True)
            audio_manifest = {
                "created": False,
                "reason": "source audio paths are shared across conditions and cannot be attributed safely",
                "ambiguous_source_paths": ambiguous_audio,
                "output": str(conversation_wav),
            }
        else:
            audio_manifest = write_conversation_wav(conversation_wav, speech_turns)
        audio_created += int(bool(audio_manifest.get("created")))
        source_integrity = []
        source_rows = []
        for turn in speech_turns:
            audio = turn.get("audio", {})
            source_rows.append({
                "path": audio.get("path"),
                "transcript_path": audio.get("transcript_path"),
            })
        seen_sources = set()
        for source in source_rows:
            for key in ("path", "transcript_path"):
                source_path = Path(str(source.get(key) or ""))
                source_key = (key, str(source_path))
                if source_path.is_file() and source_key not in seen_sources:
                    seen_sources.add(source_key)
                    source_integrity.append({
                        "kind": key,
                        "path": str(source_path),
                        "bytes": source_path.stat().st_size,
                        "sha256": _sha256_file(source_path),
                        "condition_reference_count": len(source_owners.get(str(source_path), {condition_id})),
                    })
        records.append({
            "schema_version": TRACE_SCHEMA_VERSION,
            "recovery_status": "condition_completed_before_batch_interruption" if completed else "condition_interrupted",
            "condition_id": condition_id,
            "protocol_record": record_index,
            "source_log": str(log_path),
            "source_log_sha256": _sha256_file(log_path),
            "transcript_file": str(transcript_path),
            "transcript_line_start": line_start,
            "transcript_line_end": line_end,
            "speech_pipeline": speech_turns,
            "audio_manifest": audio_manifest,
            "audio_recovery_status": "verified_unique_sources" if audio_manifest.get("created") else "unavailable_or_ambiguous",
            "source_integrity": source_integrity,
            "preservation_policy": "original turn files retained because the batch did not finalize",
        })
    protocol_temp = Path(f"{protocol_path}.tmp")
    transcript_temp = Path(f"{transcript_path}.tmp")
    protocol_temp.write_text(
        "".join(json.dumps(record, ensure_ascii=True, default=_json_default) + "\n" for record in records),
        encoding="utf-8",
    )
    transcript_temp.write_text("\n".join(transcript_lines).rstrip() + "\n", encoding="utf-8")
    if len(_structured_log_events(protocol_temp)) != len(records):
        raise RuntimeError(f"Recovered protocol verification failed for {run_dir}")
    protocol_temp.replace(protocol_path)
    transcript_temp.replace(transcript_path)
    return {
        "status": "recovered_incomplete_run",
        "run_dir": str(run_dir),
        "conversation_count": len(records),
        "conversation_audio_count": audio_created,
        "protocol": str(protocol_path),
        "transcript": str(transcript_path),
        "source_turn_files_removed": 0,
    }


def _refresh_compacted_run_metadata(run_dir, compacted_paths):
    """Replace stale per-condition artifact references after verified compaction."""
    run_dir = Path(run_dir)
    existing_index = []
    index_path = run_dir / "index.jsonl"
    if index_path.is_file():
        existing_index = _structured_log_events(index_path)
    index_by_condition = {
        str(row.get("condition_id")): row
        for row in existing_index
        if row.get("condition_id") is not None
    }
    protocol_records = _structured_log_events(compacted_paths[0]["protocol"])
    index_rows = []
    for paths in compacted_paths:
        protocol = protocol_records[int(paths["protocol_record"]) - 1]
        condition_id = str(protocol.get("summary", {}).get("condition_id") or "")
        row = dict(index_by_condition.get(condition_id, {}))
        row.update({
            "condition_id": condition_id,
            "protocol_file": Path(paths["protocol"]).name,
            "protocol_record": paths["protocol_record"],
            "transcript_file": Path(paths["transcript_txt"]).name,
            "transcript_line_start": paths["transcript_line_start"],
            "transcript_line_end": paths["transcript_line_end"],
            "conversation_wav": Path(paths["conversation_wav"]).name,
        })
        index_rows.append(row)
    write_jsonl(index_path, index_rows)

    summary_path = run_dir / "run_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["artifacts"] = [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "extension": path.suffix.lower(),
            }
            for path in sorted(item for item in run_dir.iterdir() if item.is_file())
            if path != summary_path
        ]
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )


def compact_existing_result_tree(results_root):
    """Compact finalized runs without modifying canonical observations."""
    results_root = Path(results_root).resolve()
    reports = []
    candidate_dirs = {
        path.parent for path in results_root.rglob(RESULT_FILES["summary"])
    } | {
        path.parent for path in results_root.rglob("experiment_job.json")
    }
    for run_dir in sorted(candidate_dirs):
        if (run_dir / RESULT_FILES["summary"]).is_file():
            protocol_files = sorted(run_dir.glob("*_protocol.json"))
            artifact_paths = []
            for protocol_file in protocol_files:
                prefix = protocol_file.name.removesuffix("_protocol.json")
                transcript = run_dir / f"{prefix}_conversation_transcript.txt"
                conversation_wav = run_dir / f"{prefix}_conversation.wav"
                if transcript.is_file():
                    artifact_paths.append({
                        "protocol": protocol_file,
                        "transcript_txt": transcript,
                        "conversation_wav": conversation_wav,
                    })
            if artifact_paths:
                compacted = consolidate_completed_conversation_artifacts(run_dir, artifact_paths)
                _refresh_compacted_run_metadata(run_dir, compacted)
                reports.append({
                    "status": "compacted_finalized_run",
                    "run_dir": str(run_dir),
                    "conversation_count": len(compacted),
                    "protocol": str(compacted[0]["protocol"]),
                    "transcript": str(compacted[0]["transcript_txt"]),
                })
            else:
                reports.append({"status": "already_compact", "run_dir": str(run_dir)})
            removed = remove_redundant_derived_artifacts(run_dir)
            consolidate_provider_runtime_logs(run_dir)
            refresh_result_inventory(run_dir)
            reports[-1]["removed_redundant_artifacts"] = removed
        else:
            reports.append(compact_incomplete_run_artifacts(run_dir))
    return reports


REDUNDANT_DERIVED_ARTIFACTS = (
    "condition_configuration_breakdown.json",
    "condition_configuration_breakdown.md",
    "metric_catalog.json",
    "metrics_by_phase.jsonl",
    "metrics_long.jsonl",
    "metrics_wide.jsonl",
    "retrospective_metrics.json",
    "phase_metric_overview.html",
    "phase_metric_summary.csv",
)


def remove_redundant_derived_artifacts(run_dir):
    """Remove verified copies whose information is present in canonical tables."""
    run_dir = Path(run_dir)
    metric_inputs = run_dir / RESULT_FILES["metric_inputs"]
    metrics_long = run_dir / RESULT_FILES["metrics_long"]
    if not metric_inputs.is_file() or not metrics_long.is_file():
        return []
    with metrics_long.open("r", encoding="utf-8", newline="") as handle:
        header = set(next(csv.reader(handle), ()))
    required_columns = {"phase", "metric_key", "value", "formula", "operands_json", "unavailable_reason"}
    if not required_columns <= header:
        return []
    removed = []
    for name in REDUNDANT_DERIVED_ARTIFACTS:
        path = run_dir / name
        if path.is_file():
            path.unlink()
            removed.append(name)
    return removed


def refresh_result_inventory(run_dir):
    """Refresh integrity metadata for one already completed result folder."""
    run_dir = Path(run_dir)
    summary_path = run_dir / RESULT_FILES["summary"]
    if not summary_path.is_file():
        raise FileNotFoundError(f"Completed result has no {RESULT_FILES['summary']}: {run_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest_name = summary.get("manifest")
    manifest_path = run_dir / str(manifest_name or "")
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if "analysis_artifacts" in manifest:
            manifest["analysis_artifacts"] = {
                "wide_workbook": RESULT_FILES["metrics_workbook"],
                "long_csv": RESULT_FILES["metrics_long"],
                "wide_csv": RESULT_FILES["metrics_wide"],
                "raw_metric_inputs": RESULT_FILES["metric_inputs"],
                "failure_indicators": "failure_indicators.json",
            }
            manifest["result_layout_version"] = RESULT_SCHEMA_VERSION
            manifest_path.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=True),
                encoding="utf-8",
            )
    inventory_path = write_artifact_inventory(run_dir)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    summary["artifact_inventory"] = inventory_path.name
    summary["artifact_count"] = inventory["artifact_count"]
    summary["canonical_artifact_count"] = inventory["canonical_count"]
    summary["recommended_analysis_tables"] = [
        RESULT_FILES["condition_analysis"],
        RESULT_FILES["phase_scorecard"],
        RESULT_FILES["performance_band_summary"],
        RESULT_FILES["metrics_long"],
        RESULT_FILES["metrics_wide"],
        RESULT_FILES["conditions"],
    ]
    summary.pop("artifacts", None)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    return inventory_path


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
    metrics_file = run_dir / RESULT_FILES["metrics_workbook"]
    phase_log_dir = run_dir
    export_context = _metric_export_context(run_dir, "single_run")
    write_metrics_file([metric], metrics_file, context=export_context)
    metric_exports = write_metric_phase_logs([metric], phase_log_dir, result_scope="single_run")
    network_paths = write_network_research_artifacts(
        scenario["start_time_min"],
        run_dir,
        picture_dir=run_dir,
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
            "metrics_long_csv": Path(metric_exports["metric_long_csv"]).name,
            "metrics_wide_csv": Path(metric_exports["metric_wide_csv"]).name,
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
        "network_json": network_paths["network_json"],
        "network_graph": network_paths["network_graph"],
        "run_summary": standard_summary["summary"],
        "conditions": standard_summary["conditions"],
    }


def write_conversation_protocols(results, output_dir):
    """Write one protocol JSONL and one transcript TXT for a completed batch."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [write_conversation_protocol(result, output_dir) for result in results]
    paths = consolidate_completed_conversation_artifacts(output_dir, paths)
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
            "protocol_file": Path(paths[index]["protocol"]).name,
            "protocol_record": paths[index]["protocol_record"],
            "transcript_file": Path(paths[index]["transcript_txt"]).name,
            "transcript_line_start": paths[index]["transcript_line_start"],
            "transcript_line_end": paths[index]["transcript_line_end"],
            "conversation_wav": Path(paths[index]["conversation_wav"]).name,
        }
        for index, result in enumerate(results)
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
        condition_row = {
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
            "speech_performance_band": parameters.get("speech_performance_band"),
            "speech_performance_rank": parameters.get("speech_performance_rank"),
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
            "network_seed": parameters.get("network_seed"),
            "transfer_tolerance": parameters.get("transfer_tolerance"),
            "dialogue_stagnation_limit": parameters.get("dialogue_stagnation_limit"),
            "matrix_family": parameters.get("matrix_family"),
            "experiment_platform": parameters.get("experiment_platform"),
            "experiment_seed": parameters.get("experiment_seed", parameters.get("network_seed")),
            "repetition": parameters.get("repetition", extras.get("iteration")),
            "run_mode": parameters.get("run_mode"),
            "slurm_condition_index": parameters.get("slurm_condition_index"),
            "slurm_grid_name": parameters.get("slurm_grid_name"),
            "configuration_fingerprint": (
                extras.get("condition_provenance", {}).get("fingerprint_sha256")
                or extras.get("configuration_provenance", {}).get("fingerprint_sha256")
            ),
            "base_configuration_fingerprint": (
                extras.get("condition_provenance", {}).get("base_fingerprint_sha256")
                or extras.get("configuration_provenance", {}).get("fingerprint_sha256")
            ),
            "execution_status": extras.get("execution_status", "completed"),
            "pipeline_failure_type": (extras.get("pipeline_failure") or {}).get("exception_type"),
            "pipeline_failure_message": (extras.get("pipeline_failure") or {}).get("message"),
            "route_valid": bool(result.route_valid),
            "route_reaches_goal": bool(result.route_reaches_goal),
            "route_correct": bool(result.route_correct),
            "route_duration_min": result.route_duration_min,
            "turn_count": int(extras.get("messages", len(result.conversation))),
            "runtime_sec": result.runtime_sec,
            "automatic_eval_score": getattr(metric, "automatic_eval_score", None),
            "task_success": getattr(metric, "success", bool(result.route_correct)),
        }
        from coop_navigation_sds.ResultsAndArtifacts.comparison import _comparison_condition_key
        condition_row["model_comparison_condition_key"] = _comparison_condition_key({
            **condition_row,
            "tts_engine": condition_row["configured_tts_engine"],
            "asr_engine": condition_row["configured_asr_engine"],
            "asr_beam_size": condition_row["asr_search_width"],
        })
        condition_rows.append(condition_row)
    conditions_path = output_dir / RESULT_FILES["conditions"]
    write_jsonl(conditions_path, condition_rows)
    summary = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "result_scope": result_scope,
        "result_run_id": output_dir.name,
        "condition_count": len(condition_rows),
        "completed_condition_count": sum(row["execution_status"] == "completed" for row in condition_rows),
        "failed_condition_count": sum(row["execution_status"] == "failed" for row in condition_rows),
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
        "artifact_inventory": RESULT_FILES["artifact_inventory"],
    }
    summary_path = output_dir / RESULT_FILES["summary"]
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    from coop_navigation_sds.ResultsAndArtifacts.comparison import write_run_analysis_outputs
    analysis_paths = write_run_analysis_outputs(output_dir)
    summary["recommended_analysis_tables"] = [
        Path(analysis_paths["run_analysis"]).name,
        Path(analysis_paths["condition_analysis"]).name,
        Path(analysis_paths["run_phase_scorecard"]).name,
        Path(analysis_paths["performance_band_summary"]).name,
        "metrics_long.csv",
        "metrics_wide.csv",
        conditions_path.name,
    ]
    summary["analysis_overview"] = Path(analysis_paths["analysis_overview"]).name
    inventory_path = write_artifact_inventory(output_dir)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    summary["artifact_count"] = inventory["artifact_count"]
    summary["canonical_artifact_count"] = inventory["canonical_count"]
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
            "long_csv": RESULT_FILES["metrics_long"],
            "wide_csv": RESULT_FILES["metrics_wide"],
            "raw_metric_inputs": RESULT_FILES["metric_inputs"],
            "failure_indicators": "failure_indicators.json",
        },
        "conditions": rows,
    }
    path = output_dir / "experiment_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    return path
