"""Transcribe a fixed TTS audio corpus with one configured ASR backend."""
from argparse import ArgumentParser
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coop_navigation_sds.DialogManagement.speech_pipeline import SpeechPipelineConfig, SpeechSignal, SpeechTransport


EXPECTED_ENTITIES = (("ring",), ("bravo",), ("harbor", "harbour"))


def semantic_accuracy(transcript):
    folded = str(transcript or "").casefold()
    return sum(
        any(variant in folded for variant in variants)
        for variants in EXPECTED_ENTITIES
    ) / len(EXPECTED_ENTITIES)


def main():
    parser = ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--executable", default="")
    parser.add_argument("--vad-model", default="")
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = SpeechPipelineConfig(
        tts_engine="file",
        asr_engine=args.engine,
        audio_dir=str(manifest_path.parent),
        asr_model=args.model,
        asr_device=args.device,
        asr_compute_type=args.compute_type,
        asr_executable=args.executable,
        asr_vad_model=args.vad_model,
        asr_timeout_sec=args.timeout_seconds,
        asr_language="en-US",
        asr_beam_size=5,
        playback_enabled=False,
        realtime_enabled=False,
    )
    records = []
    try:
        asr = SpeechTransport(config=config).asr_engine
        initialization_error = ""
    except Exception as exc:
        asr = None
        initialization_error = f"{type(exc).__name__}: {exc}"

    for item in manifest["audio"]:
        audio_path = (manifest_path.parent / item["path"]).resolve()
        signal = SpeechSignal(
            speaker="Agent A",
            text=manifest["probe_text"],
            audio={"path": str(audio_path)},
            diagnostics={},
        )
        started = time.perf_counter()
        if initialization_error:
            transcript = ""
            accuracy = 0.0
            status = "runtime_fail"
            error = initialization_error
        else:
            try:
                transcript = asr.transcribe(signal)
                accuracy = semantic_accuracy(transcript)
                status = "pass" if accuracy == 1.0 else "semantic_fail"
                error = ""
            except Exception as exc:
                transcript = ""
                accuracy = 0.0
                status = "runtime_fail"
                error = f"{type(exc).__name__}: {exc}"
        records.append({
            "tts_engine": item["tts_engine"],
            "asr_engine": args.engine,
            "status": status,
            "semantic_accuracy": round(accuracy, 6),
            "transcript": transcript,
            "latency_seconds": round(time.perf_counter() - started, 6),
            "error": error,
            "audio_path": str(audio_path),
            "diagnostics": signal.diagnostics or {},
        })

    protocol = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "asr_engine": args.engine,
        "configuration": {
            "model": args.model,
            "device": args.device,
            "compute_type": args.compute_type,
            "executable": args.executable,
            "vad_model": args.vad_model,
        },
        "summary": {
            "case_count": len(records),
            "passed": sum(record["status"] == "pass" for record in records),
            "semantic_failed": sum(
                record["status"] == "semantic_fail" for record in records
            ),
            "runtime_failed": sum(
                record["status"] == "runtime_fail" for record in records
            ),
        },
        "records": records,
    }
    output = manifest_path.parent / f"asr_{args.engine}_protocol.json"
    output.write_text(json.dumps(protocol, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(protocol["summary"], ensure_ascii=True))
    print(output)


if __name__ == "__main__":
    main()
