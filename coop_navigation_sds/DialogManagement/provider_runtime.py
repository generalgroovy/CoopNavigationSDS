"""Persistent subprocess bridge for dependency-isolated speech providers."""
from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import queue
import subprocess
import threading


WORKER_ENV = "MINILLAMA_SPEECH_PROVIDER_WORKER"
PYTHON_NAMES = (
    Path("Scripts/python.exe"),
    Path("bin/python"),
    Path("python.exe"),
    Path("python"),
)


class ProviderProcessStoppedError(RuntimeError):
    """Raised when an isolated provider exits before returning a response."""


def _provider_key(engine):
    return str(engine or "").strip().lower().replace("-", "_").replace(".", "_")


def _valid_python(path):
    candidate = Path(path).expanduser()
    return candidate.resolve() if candidate.is_file() else None


def resolve_provider_python(engine, explicit="", environment_dir=".speech-providers"):
    """Resolve an explicit or manifest-managed provider interpreter."""
    if os.environ.get(WORKER_ENV) == "1":
        return None
    if str(explicit or "").strip():
        resolved = _valid_python(explicit)
        if resolved is None:
            raise FileNotFoundError(f"Configured provider Python does not exist: {explicit}")
        return resolved

    root = Path(environment_dir or ".speech-providers").expanduser()
    manifest_path = root / "providers.json"
    if manifest_path.is_file():
        try:
            document = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            document = {}
        entry = document.get("providers", {}).get(_provider_key(engine), {})
        configured = entry.get("python") if isinstance(entry, dict) else entry
        if configured:
            candidate = Path(configured)
            if not candidate.is_absolute():
                candidate = manifest_path.parent / candidate
            resolved = _valid_python(candidate)
            if resolved:
                return resolved

    provider_dir = root / _provider_key(engine)
    for relative in PYTHON_NAMES:
        resolved = _valid_python(provider_dir / relative)
        if resolved:
            return resolved
    return None


class ProviderProcessClient:
    """Maintain one JSON-lines worker so model weights remain loaded."""

    def __init__(self, python, stage, engine, timeout_seconds, log_path=None):
        self.python = Path(python)
        self.stage = stage
        self.engine = engine
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.process = None
        self.responses = queue.Queue()
        self.lock = threading.Lock()
        self.log_path = Path(log_path) if log_path else None
        self.stderr_handle = None

    def _start(self):
        if self.process is not None and self.process.poll() is None:
            return False
        root = Path(__file__).resolve().parents[2]
        environment = dict(os.environ)
        environment[WORKER_ENV] = "1"
        environment["PYTHONPATH"] = os.pathsep.join(
            part for part in (str(root), environment.get("PYTHONPATH", "")) if part
        )
        environment["PATH"] = os.pathsep.join(
            part
            for part in (
                str(self.python.parent),
                environment.get("PATH", ""),
            )
            if part
        )
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.stderr_handle = self.log_path.open("a", encoding="utf-8")
        self.process = subprocess.Popen(
            [
                str(self.python),
                "-m",
                "coop_navigation_sds.DialogManagement.provider_worker",
                "--stage",
                self.stage,
                "--engine",
                self.engine,
            ],
            cwd=str(root),
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr_handle,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        response_queue = self.responses
        process = self.process
        threading.Thread(
            target=self._read_responses,
            args=(process, response_queue),
            daemon=True,
        ).start()
        return True

    @staticmethod
    def _read_responses(process, response_queue):
        for line in process.stdout:
            response_queue.put(line)
        response_queue.put(None)

    def request(self, payload):
        with self.lock:
            for restart_count in range(2):
                try:
                    response = self._request_once(payload)
                    response["provider_restart_count"] = restart_count
                    return response
                except ProviderProcessStoppedError:
                    self.close()
                    if restart_count:
                        raise
                    self.responses = queue.Queue()

    def _request_once(self, payload):
        cold_start = self._start()
        self.process.stdin.write(json.dumps(payload, ensure_ascii=True) + "\n")
        self.process.stdin.flush()
        try:
            cold_timeout = (
                1800.0
                if self.engine in {"qwen3_tts", "qwen3_asr", "parakeet", "f5_tts"}
                else 600.0
            )
            line = self.responses.get(
                timeout=max(self.timeout_seconds, cold_timeout)
                if cold_start
                else self.timeout_seconds
            )
        except queue.Empty as exc:
            self.close()
            raise TimeoutError(
                f"{self.engine} {self.stage} provider exceeded "
                f"{self.timeout_seconds:.1f} seconds."
            ) from exc
        if line is None:
            return_code = self.process.poll()
            raise ProviderProcessStoppedError(
                f"{self.engine} {self.stage} provider stopped unexpectedly "
                f"with exit code {return_code}."
            )
        response = json.loads(line)
        if not response.get("ok"):
            raise RuntimeError(
                f"{self.engine} {self.stage} provider failed: "
                f"{response.get('error', 'unknown error')}\n"
                f"{response.get('traceback', '')}".rstrip()
            )
        return response

    def close(self):
        process = self.process
        self.process = None
        if process is None:
            return
        if process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                process.kill()
                process.wait(timeout=5)
        for stream in (process.stdin, process.stdout):
            if stream is not None and not stream.closed:
                stream.close()
        if self.stderr_handle is not None and not self.stderr_handle.closed:
            self.stderr_handle.close()
        self.stderr_handle = None

    def __del__(self):
        self.close()


def _worker_config(config, stage, engine):
    values = asdict(config)
    values["tts_python_executable"] = ""
    values["asr_python_executable"] = ""
    if stage == "tts":
        values["tts_engine"] = engine
        values["asr_engine"] = "file"
        # Audio devices belong to the application process, not model workers.
        values["playback_enabled"] = False
        values["realtime_enabled"] = False
    else:
        values["tts_engine"] = "file"
        values["asr_engine"] = engine
    return values


class IsolatedTextToSpeech:
    """Text-to-speech proxy backed by a provider-specific Python runtime."""

    def __init__(self, engine, python, config):
        self.engine = engine
        self.name = f"isolated-{engine}-tts"
        self.config = config
        self.client = ProviderProcessClient(
            python,
            "tts",
            engine,
            config.tts_timeout_sec,
            Path(config.audio_dir) / f"provider_tts_{engine}.log",
        )

    def synthesize(self, speaker, text):
        from coop_navigation_sds.DialogManagement.speech_pipeline import (
            SpeechPipelineError,
            SpeechSignal,
            WaveFileTextToSpeech,
        )

        response = self.client.request({
            "command": "synthesize",
            "config": _worker_config(self.config, "tts", self.engine),
            "speaker": speaker,
            "text": text,
        })
        audio = dict(response["audio"])
        project_root = Path(__file__).resolve().parents[2]
        for key in ("path", "transcript_path"):
            value = audio.get(key)
            if value and not Path(value).is_absolute():
                audio[key] = str((project_root / value).resolve())
        played = False
        if self.config.playback_enabled:
            played = WaveFileTextToSpeech._play_wave(
                Path(audio["path"]),
                realtime=self.config.realtime_enabled,
                fallback_duration=float(audio.get("duration_sec") or 0.0),
            )
            if not played:
                raise SpeechPipelineError(
                    "Text-to-speech audio was generated but playback failed.",
                    {"engine": self.name, "path": audio.get("path"), "stage": "playback"},
                )
        audio.update({
            "played": played,
            "realtime": bool(self.config.realtime_enabled),
            "waited": bool(played and self.config.realtime_enabled),
        })
        diagnostics = dict(response.get("diagnostics") or {})
        diagnostics["provider_restart_count"] = int(
            response.get("provider_restart_count", 0)
        )
        return SpeechSignal(
            speaker=speaker,
            text=response["text"],
            audio=audio,
            diagnostics=diagnostics,
        )


class IsolatedSpeechToText:
    """Automatic speech recognition proxy backed by an isolated runtime."""

    def __init__(self, engine, python, config):
        self.engine = engine
        self.name = f"isolated-{engine}-asr"
        self.config = config
        self.client = ProviderProcessClient(
            python,
            "asr",
            engine,
            config.asr_timeout_sec,
            Path(config.audio_dir) / f"provider_asr_{engine}.log",
        )

    def transcribe(self, signal):
        response = self.client.request({
            "command": "transcribe",
            "config": _worker_config(self.config, "asr", self.engine),
            "signal": {
                "speaker": signal.speaker,
                "text": signal.text,
                "audio": signal.audio,
                "diagnostics": signal.diagnostics or {},
            },
        })
        signal.diagnostics = dict(response.get("diagnostics") or {})
        signal.diagnostics["provider_restart_count"] = int(
            response.get("provider_restart_count", 0)
        )
        return response["transcript"]
