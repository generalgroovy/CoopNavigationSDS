"""Retrospective NISQA v2 evaluation for generated speech artifacts."""
from __future__ import annotations

from importlib import metadata
from pathlib import Path
import statistics
import struct
import wave


NISQA_DIMENSIONS = (
    "overall_mos",
    "noisiness",
    "discontinuity",
    "coloration",
    "loudness",
)


def read_mono_pcm_wave(path):
    """Read a PCM WAV as mono floating-point samples without audio dependencies."""
    path = Path(path)
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frame_count = handle.getnframes()
        compression = handle.getcomptype()
        payload = handle.readframes(frame_count)
    if compression != "NONE":
        raise ValueError(f"NISQA requires uncompressed PCM WAV audio, received {compression}.")
    if channels < 1:
        raise ValueError("NISQA received a WAV file without audio channels.")
    formats = {
        1: ("B", 128.0, 128.0),
        2: ("h", 32768.0, 0.0),
        4: ("i", 2147483648.0, 0.0),
    }
    if sample_width not in formats:
        raise ValueError(f"NISQA does not support {sample_width * 8}-bit PCM WAV audio.")
    code, scale, offset = formats[sample_width]
    count = frame_count * channels
    values = struct.unpack(f"<{count}{code}", payload)
    mono = []
    for frame_index in range(frame_count):
        start = frame_index * channels
        channel_values = values[start:start + channels]
        mono.append(
            sum((float(value) - offset) / scale for value in channel_values)
            / channels
        )
    return mono, sample_rate


class NISQAEvaluator:
    """Evaluate WAV artifacts with TorchMetrics' NISQA v2 implementation."""

    estimator_name = "TorchMetrics NISQA v2.0"

    def evaluate(self, items):
        requested_items = tuple(items)
        requests = []
        for item in requested_items:
            if isinstance(item, dict):
                path = item.get("path")
                context = {
                    key: value for key, value in item.items()
                    if key != "path"
                }
            else:
                path = item
                context = {}
            if path and Path(path).is_file():
                requests.append((str(Path(path)), context))
        unique_requests = {
            path: context for path, context in requests
        }
        report = {
            "status": "unavailable",
            "estimator": self.estimator_name,
            "implementation": "torchmetrics.functional.audio.nisqa",
            "implementation_version": None,
            "score": None,
            "dimensions": {key: None for key in NISQA_DIMENSIONS},
            "evaluated_file_count": 0,
            "requested_file_count": len(requested_items),
            "files": [],
            "by_agent": {},
            "errors": [],
        }
        if not unique_requests:
            report["reason"] = "no_readable_wav_artifacts"
            return report
        try:
            import torch
            from torchmetrics.functional.audio.nisqa import (
                non_intrusive_speech_quality_assessment,
            )
        except Exception as exc:
            report["reason"] = "nisqa_dependencies_unavailable"
            report["errors"].append(f"{type(exc).__name__}: {exc}")
            return report

        try:
            report["implementation_version"] = metadata.version("torchmetrics")
        except metadata.PackageNotFoundError:
            report["implementation_version"] = "unknown"

        rows = []
        for audio_path, context in unique_requests.items():
            try:
                samples, sample_rate = read_mono_pcm_wave(audio_path)
                waveform = torch.tensor(samples, dtype=torch.float32)
                with torch.inference_mode():
                    values = non_intrusive_speech_quality_assessment(
                        waveform,
                        sample_rate,
                    )
                scores = [float(value) for value in values.detach().cpu().reshape(-1)]
                if len(scores) != len(NISQA_DIMENSIONS):
                    raise ValueError(
                        f"NISQA returned {len(scores)} values; expected {len(NISQA_DIMENSIONS)}."
                    )
                row = {
                    "path": audio_path,
                    "sample_rate": sample_rate,
                    **context,
                    **dict(zip(NISQA_DIMENSIONS, scores)),
                }
                rows.append(row)
            except Exception as exc:
                report["errors"].append(
                    f"{audio_path}: {type(exc).__name__}: {exc}"
                )

        report["files"] = rows
        report["evaluated_file_count"] = len(rows)
        if not rows:
            report["reason"] = "nisqa_evaluation_failed"
            return report
        report["status"] = "available"
        report["reason"] = None
        report["dimensions"] = {
            key: round(statistics.fmean(row[key] for row in rows), 6)
            for key in NISQA_DIMENSIONS
        }
        speakers = {
            row.get("speaker") for row in rows if row.get("speaker")
        }
        report["by_agent"] = {
            speaker: {
                "evaluated_file_count": len(agent_rows),
                "dimensions": {
                    key: round(statistics.fmean(row[key] for row in agent_rows), 6)
                    for key in NISQA_DIMENSIONS
                },
            }
            for speaker in sorted(speakers)
            if (agent_rows := [row for row in rows if row.get("speaker") == speaker])
        }
        for agent_report in report["by_agent"].values():
            agent_report["score"] = agent_report["dimensions"]["overall_mos"]
        report["score"] = report["dimensions"]["overall_mos"]
        return report
