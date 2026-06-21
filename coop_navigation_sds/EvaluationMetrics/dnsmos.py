"""Retrospective DNSMOS evaluation for generated speech artifacts."""
from __future__ import annotations

from importlib import metadata
from pathlib import Path
import statistics

from coop_navigation_sds.EvaluationMetrics.nisqa import read_mono_pcm_wave


DNSMOS_DIMENSIONS = (
    "p808_mos",
    "signal_mos",
    "background_mos",
    "overall_mos",
)


class DNSMOSEvaluator:
    """Evaluate WAV artifacts with TorchMetrics' DNSMOS implementation."""

    estimator_name = "TorchMetrics DNSMOS"

    def __init__(self, personalized=False, device="cpu", num_threads=None):
        self.personalized = bool(personalized)
        self.device = device
        self.num_threads = num_threads

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
            "implementation": "torchmetrics.functional.audio.dnsmos",
            "implementation_version": None,
            "onnxruntime_version": None,
            "personalized": self.personalized,
            "device": self.device,
            "num_threads": self.num_threads,
            "score": None,
            "dimensions": {key: None for key in DNSMOS_DIMENSIONS},
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
            from torchmetrics.functional.audio.dnsmos import (
                deep_noise_suppression_mean_opinion_score,
            )
        except Exception as exc:
            report["reason"] = "dnsmos_dependencies_unavailable"
            report["errors"].append(f"{type(exc).__name__}: {exc}")
            return report

        try:
            report["implementation_version"] = metadata.version("torchmetrics")
        except metadata.PackageNotFoundError:
            report["implementation_version"] = "unknown"
        try:
            report["onnxruntime_version"] = metadata.version("onnxruntime")
        except metadata.PackageNotFoundError:
            report["onnxruntime_version"] = "unknown"

        rows = []
        for audio_path, context in unique_requests.items():
            try:
                samples, sample_rate = read_mono_pcm_wave(audio_path)
                waveform = torch.tensor(samples, dtype=torch.float32)
                with torch.inference_mode():
                    values = deep_noise_suppression_mean_opinion_score(
                        waveform,
                        sample_rate,
                        personalized=self.personalized,
                        device=self.device,
                        num_threads=self.num_threads,
                        cache_session=True,
                    )
                scores = [float(value) for value in values.detach().cpu().reshape(-1)]
                if len(scores) != len(DNSMOS_DIMENSIONS):
                    raise ValueError(
                        f"DNSMOS returned {len(scores)} values; expected {len(DNSMOS_DIMENSIONS)}."
                    )
                rows.append({
                    "path": audio_path,
                    "sample_rate": sample_rate,
                    **context,
                    **dict(zip(DNSMOS_DIMENSIONS, scores)),
                })
            except Exception as exc:
                report["errors"].append(
                    f"{audio_path}: {type(exc).__name__}: {exc}"
                )

        report["files"] = rows
        report["evaluated_file_count"] = len(rows)
        if not rows:
            report["reason"] = "dnsmos_evaluation_failed"
            return report
        report["status"] = "available"
        report["reason"] = None
        report["dimensions"] = {
            key: round(statistics.fmean(row[key] for row in rows), 6)
            for key in DNSMOS_DIMENSIONS
        }
        speakers = {
            row.get("speaker") for row in rows if row.get("speaker")
        }
        report["by_agent"] = {
            speaker: {
                "evaluated_file_count": len(agent_rows),
                "dimensions": {
                    key: round(statistics.fmean(row[key] for row in agent_rows), 6)
                    for key in DNSMOS_DIMENSIONS
                },
            }
            for speaker in sorted(speakers)
            if (agent_rows := [row for row in rows if row.get("speaker") == speaker])
        }
        for agent_report in report["by_agent"].values():
            agent_report["score"] = agent_report["dimensions"]["overall_mos"]
        report["score"] = report["dimensions"]["overall_mos"]
        return report
