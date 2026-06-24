"""Retrospective intrusive speech-quality metrics for aligned audio pairs."""
from __future__ import annotations

import math
from pathlib import Path
import statistics

from coop_navigation_sds.EvaluationMetrics.nisqa import read_mono_pcm_wave


def _aligned_pair(reference_path, degraded_path):
    reference, reference_rate = read_mono_pcm_wave(reference_path)
    degraded, degraded_rate = read_mono_pcm_wave(degraded_path)
    if reference_rate != degraded_rate:
        raise ValueError("Reference and synthesized WAV sample rates differ.")
    sample_count = min(len(reference), len(degraded))
    if sample_count < max(800, reference_rate // 4):
        raise ValueError("Aligned audio pair is too short for intrusive evaluation.")
    return reference[:sample_count], degraded[:sample_count], reference_rate


def _si_sdr(reference, degraded):
    reference_energy = sum(value * value for value in reference)
    if reference_energy <= 1e-12:
        raise ValueError("Reference audio has no measurable signal energy.")
    scale = sum(r * d for r, d in zip(reference, degraded)) / reference_energy
    target = [scale * value for value in reference]
    noise_energy = sum((d - t) ** 2 for d, t in zip(degraded, target))
    target_energy = sum(value * value for value in target)
    if noise_energy <= 1e-12:
        return 60.0
    return 10.0 * math.log10(max(target_energy, 1e-12) / noise_energy)


class ReferenceAudioQualityEvaluator:
    """Evaluate PESQ, STOI, SI-SDR, and externally supplied licensed POLQA."""

    def evaluate(self, items):
        requested = tuple(items)
        rows = []
        errors = []
        for item in requested:
            if not isinstance(item, dict):
                continue
            degraded_path = item.get("path")
            reference_path = item.get("reference_path")
            if not degraded_path or not reference_path:
                continue
            if not Path(degraded_path).is_file() or not Path(reference_path).is_file():
                errors.append(f"Unreadable pair: reference={reference_path}, synthesized={degraded_path}")
                continue
            row = {
                "path": str(degraded_path),
                "reference_path": str(reference_path),
                "speaker": item.get("speaker"),
                "turn_index": item.get("turn_index"),
                "pesq": None,
                "stoi": None,
                "si_sdr_db": None,
                "polqa": item.get("polqa_score"),
            }
            try:
                reference, degraded, sample_rate = _aligned_pair(reference_path, degraded_path)
                row["sample_rate"] = sample_rate
                row["si_sdr_db"] = round(_si_sdr(reference, degraded), 6)
                try:
                    import numpy as np
                    from pesq import pesq
                    if sample_rate not in {8000, 16000}:
                        raise ValueError("PESQ requires 8 kHz or 16 kHz aligned audio.")
                    mode = "wb" if sample_rate == 16000 else "nb"
                    row["pesq"] = round(float(pesq(
                        sample_rate,
                        np.asarray(reference, dtype=np.float32),
                        np.asarray(degraded, dtype=np.float32),
                        mode,
                    )), 6)
                except Exception as exc:
                    errors.append(f"PESQ {degraded_path}: {type(exc).__name__}: {exc}")
                try:
                    import numpy as np
                    from pystoi import stoi
                    row["stoi"] = round(float(stoi(
                        np.asarray(reference, dtype=np.float32),
                        np.asarray(degraded, dtype=np.float32),
                        sample_rate,
                        extended=False,
                    )), 6)
                except Exception as exc:
                    errors.append(f"STOI {degraded_path}: {type(exc).__name__}: {exc}")
                rows.append(row)
            except Exception as exc:
                errors.append(f"Pair {degraded_path}: {type(exc).__name__}: {exc}")

        def mean(key):
            values = [float(row[key]) for row in rows if row.get(key) is not None]
            return round(statistics.fmean(values), 6) if values else None

        scores = {
            "pesq": mean("pesq"),
            "polqa": mean("polqa"),
            "stoi": mean("stoi"),
            "si_sdr_db": mean("si_sdr_db"),
        }
        return {
            "status": "available" if any(value is not None for value in scores.values()) else "unavailable",
            "reason": None if any(value is not None for value in scores.values()) else "no_aligned_reference_audio_pairs",
            "requested_file_count": len(requested),
            "evaluated_pair_count": len(rows),
            "scores": scores,
            "files": rows,
            "errors": errors,
            "polqa_policy": (
                "POLQA is patent-licensed; provide a score from a licensed implementation "
                "as polqa_score in the audio trace. No surrogate score is generated."
            ),
        }
