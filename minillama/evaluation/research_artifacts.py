"""Research artifact writers for model, network, and metric analysis data."""
from __future__ import annotations

from dataclasses import asdict
import csv
import json
from pathlib import Path

from minillama.evaluation.xlsx_export import write_metrics_xlsx
from minillama.model.network_overview import build_network_overview
from minillama.model.network_picture import write_network_svg


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
