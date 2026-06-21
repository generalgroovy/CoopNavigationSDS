"""JSON-lines worker used by isolated speech-provider environments."""
from argparse import ArgumentParser
from contextlib import redirect_stdout
import json
import sys
import traceback


def _write(response):
    print(json.dumps(response, ensure_ascii=True, default=str), flush=True)


def main():
    parser = ArgumentParser()
    parser.add_argument("--stage", choices=("tts", "asr"), required=True)
    parser.add_argument("--engine", required=True)
    args = parser.parse_args()

    from coop_navigation_sds.DialogManagement.speech_pipeline import SpeechPipelineConfig, SpeechSignal, SpeechTransport

    transport = None
    for line in sys.stdin:
        try:
            payload = json.loads(line)
            if transport is None:
                config = SpeechPipelineConfig(**payload["config"])
                with redirect_stdout(sys.stderr):
                    transport = SpeechTransport(config=config)
            if args.stage == "tts" and payload.get("command") == "synthesize":
                with redirect_stdout(sys.stderr):
                    signal = transport.tts_engine.synthesize(
                        payload["speaker"],
                        payload["text"],
                    )
                response = {
                    "ok": True,
                    "text": signal.text,
                    "audio": signal.audio,
                    "diagnostics": signal.diagnostics or {},
                }
            elif args.stage == "asr" and payload.get("command") == "transcribe":
                values = payload["signal"]
                signal = SpeechSignal(
                    speaker=values["speaker"],
                    text=values.get("text", ""),
                    audio=values.get("audio"),
                    diagnostics=values.get("diagnostics") or {},
                )
                with redirect_stdout(sys.stderr):
                    transcript = transport.asr_engine.transcribe(signal)
                response = {
                    "ok": True,
                    "transcript": transcript,
                    "diagnostics": signal.diagnostics or {},
                }
            else:
                response = {"ok": False, "error": "Unsupported provider command."}
        except Exception as exc:
            response = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        _write(response)


if __name__ == "__main__":
    main()
