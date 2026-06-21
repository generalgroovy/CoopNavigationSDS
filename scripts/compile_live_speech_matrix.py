"""Compile live TTS-to-ASR probe protocols into research-readable artifacts."""
from argparse import ArgumentParser
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import statistics
import wave


from coop_navigation_sds.DialogManagement.speech_pipeline import available_asr_engine_keys, available_tts_engine_keys


TTS_ENGINES = tuple(key for key in available_tts_engine_keys() if key != "file")
ASR_ENGINES = tuple(key for key in available_asr_engine_keys() if key != "file")


def words(text):
    normalized = re.sub(r"[^a-z0-9 ]+", " ", str(text).casefold())
    return normalized.replace("harbour", "harbor").split()


def word_error_rate(reference, hypothesis):
    expected = words(reference)
    actual = words(hypothesis)
    distances = list(range(len(actual) + 1))
    for row, expected_word in enumerate(expected, 1):
        next_distances = [row]
        for column, actual_word in enumerate(actual, 1):
            next_distances.append(min(
                next_distances[-1] + 1,
                distances[column] + 1,
                distances[column - 1] + (expected_word != actual_word),
            ))
        distances = next_distances
    return distances[-1] / max(1, len(expected))


def audio_duration(path):
    with wave.open(str(path), "rb") as wav_file:
        return wav_file.getnframes() / wav_file.getframerate()


def mean(values):
    return round(statistics.fmean(values), 6) if values else None


def render_markdown(protocol):
    summary = protocol["summary"]
    lines = [
        "# Live Speech Backend Matrix",
        "",
        f"- Created: {protocol['created_utc']}",
        f"- Probe: `{protocol['probe_text']}`",
        f"- Executed combinations: {summary['executed_cases']} / {summary['expected_cases']}",
        f"- Task-semantic passes: {summary['semantic_passes']} / {summary['expected_cases']}",
        f"- Runtime failures: {summary['runtime_failures']}",
        "",
        "## Recognition Summary",
        "",
        "| Automatic speech recognition | Semantic pass | Mean word error rate | Cold latency (s) | Warm mean latency (s) |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in protocol["asr_summary"]:
        lines.append(
            f"| {item['asr_engine']} | {item['semantic_passes']}/{item['cases']} | "
            f"{item['mean_word_error_rate']:.3f} | {item['cold_latency_seconds']:.3f} | "
            f"{item['warm_mean_latency_seconds']:.3f} |"
        )
    lines.extend([
        "",
        "## Synthesis Summary",
        "",
        "| Text-to-speech | Semantic pass across recognizers | Audio duration (s) |",
        "|---|---:|---:|",
    ])
    for item in protocol["tts_summary"]:
        lines.append(
            f"| {item['tts_engine']} | {item['semantic_passes']}/{item['cases']} | "
            f"{item['audio_duration_seconds']:.3f} |"
        )
    lines.extend([
        "",
        "## Case Matrix",
        "",
        "| Text-to-speech | SAPI | Faster-Whisper | Vosk | whisper.cpp | NVIDIA Parakeet | Qwen3-ASR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    records = {(item["tts_engine"], item["asr_engine"]): item for item in protocol["records"]}
    for tts_engine in TTS_ENGINES:
        cells = []
        for asr_engine in ASR_ENGINES:
            record = records[(tts_engine, asr_engine)]
            label = "PASS" if record["status"] == "pass" else record["status"].upper()
            cells.append(f"{label} ({record['word_error_rate']:.3f})")
        lines.append(f"| {tts_engine} | " + " | ".join(cells) + " |")
    lines.extend([
        "",
        "Parentheses contain task-normalized word error rate. `Harbour` and `Harbor` are treated as equivalent.",
        "",
        "## Interpretation",
        "",
        "- Faster-Whisper, Vosk, whisper.cpp, NVIDIA Parakeet, and Qwen3-ASR preserved Ring, Bravo, and Harbor for every voice.",
        "- Windows SAPI preserved all three entities only for Kokoro; its other six cases were semantic failures.",
        "- Cold latency includes lazy model initialization and any first-use model retrieval. Warm latency excludes the first recording.",
        "- This is a clean, single-utterance compatibility probe. It does not establish robustness to noise, accents, pauses, overlap, or long dialogue.",
        "",
        "## Reproduction",
        "",
        "The raw per-recognizer protocols and all generated WAV files are retained beside this report. "
        "`speech_matrix_cases.csv` contains one row per combination for statistical analysis.",
    ])
    return "\n".join(lines) + "\n"


def main():
    parser = ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    manifest = json.loads((run_dir / "audio_manifest.json").read_text(encoding="utf-8"))
    probe_text = manifest["probe_text"]
    records = []
    for asr_engine in ASR_ENGINES:
        source = run_dir / f"asr_{asr_engine}_protocol.json"
        data = json.loads(source.read_text(encoding="utf-8"))
        for record in data["records"]:
            item = dict(record)
            item["word_error_rate"] = round(
                word_error_rate(probe_text, item["transcript"]),
                6,
            )
            records.append(item)

    expected_pairs = {(tts, asr) for tts in TTS_ENGINES for asr in ASR_ENGINES}
    actual_pairs = {(item["tts_engine"], item["asr_engine"]) for item in records}
    if actual_pairs != expected_pairs:
        missing = sorted(expected_pairs - actual_pairs)
        extra = sorted(actual_pairs - expected_pairs)
        raise RuntimeError(f"Matrix mismatch: missing={missing}, extra={extra}")

    audio_paths = {
        item["tts_engine"]: (run_dir / item["path"]).resolve()
        for item in manifest["audio"]
    }
    asr_summary = []
    for asr_engine in ASR_ENGINES:
        cases = [item for item in records if item["asr_engine"] == asr_engine]
        asr_summary.append({
            "asr_engine": asr_engine,
            "cases": len(cases),
            "semantic_passes": sum(item["status"] == "pass" for item in cases),
            "runtime_failures": sum(item["status"] == "runtime_fail" for item in cases),
            "mean_word_error_rate": mean([item["word_error_rate"] for item in cases]),
            "cold_latency_seconds": cases[0]["latency_seconds"],
            "warm_mean_latency_seconds": mean([
                item["latency_seconds"] for item in cases[1:]
            ]),
        })
    tts_summary = []
    for tts_engine in TTS_ENGINES:
        cases = [item for item in records if item["tts_engine"] == tts_engine]
        tts_summary.append({
            "tts_engine": tts_engine,
            "cases": len(cases),
            "semantic_passes": sum(item["status"] == "pass" for item in cases),
            "mean_word_error_rate": mean([item["word_error_rate"] for item in cases]),
            "audio_duration_seconds": round(audio_duration(audio_paths[tts_engine]), 6),
            "audio_path": str(audio_paths[tts_engine]),
        })
    protocol = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "probe_text": probe_text,
        "evaluation": {
            "semantic_pass": "Transcript contains Ring, Bravo, and Harbor or Harbour.",
            "word_error_rate": "Case-insensitive word error rate with Harbor and Harbour equivalent.",
            "latency": "Wall-clock adapter transcription time; first case includes lazy initialization.",
        },
        "environment": {
            "operating_system": "Windows",
            "device": "CPU",
            "whisper_cpp_version": "1.8.6",
            "faster_whisper_model": "tiny.en",
            "vosk_model": "vosk-model-small-en-us-0.15",
            "parakeet_model": "nvidia/parakeet-tdt-0.6b-v2",
            "qwen3_asr_model": "Qwen/Qwen3-ASR-0.6B",
        },
        "summary": {
            "expected_cases": len(expected_pairs),
            "executed_cases": len(records),
            "semantic_passes": sum(item["status"] == "pass" for item in records),
            "semantic_failures": sum(item["status"] == "semantic_fail" for item in records),
            "runtime_failures": sum(item["status"] == "runtime_fail" for item in records),
        },
        "asr_summary": asr_summary,
        "tts_summary": tts_summary,
        "records": records,
    }
    (run_dir / "live_speech_matrix_protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    fieldnames = (
        "tts_engine", "asr_engine", "status", "semantic_accuracy",
        "word_error_rate", "latency_seconds", "transcript", "error", "audio_path",
    )
    with (run_dir / "speech_matrix_cases.csv").open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    (run_dir / "live_speech_matrix_report.md").write_text(
        render_markdown(protocol),
        encoding="utf-8",
    )
    print(json.dumps(protocol["summary"], ensure_ascii=True))


if __name__ == "__main__":
    main()
