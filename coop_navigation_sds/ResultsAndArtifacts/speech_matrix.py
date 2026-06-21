"""Cartesian speech-backend verification and research protocol output."""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import wave

from coop_navigation_sds.DialogManagement.speech_pipeline import (
    ASR_ENGINE_SPECS,
    TTS_ENGINE_SPECS,
    SpeechPipelineConfig,
    SpeechPipelineError,
    SpeechSignal,
    SpeechTransport,
)
from coop_navigation_sds.DialogManagement.provider_runtime import resolve_provider_python
from coop_navigation_sds.DialogManagement.whisper_cpp_runtime import whisper_cpp_ready
from coop_navigation_sds.Configuration.component_catalog import apply_speech_engine_profiles


PROBE_TEXT = {
    "Agent A": "I am at Bravo at eight seven, going to Harbor.",
    "Agent B": "Take metro line M1 from Bravo to Harbor with no changes.",
}

PROBE_ENTITIES = {
    "Agent A": (("bravo",), ("harbor", "harbour"), ("eight seven", "8 7", "87")),
    "Agent B": (("ring",), ("bravo",), ("harbor", "harbour")),
}


@dataclass
class SpeechMatrixCase:
    """Result of one text-to-speech and recognition pairing."""

    case_id: str
    tts_engine: str
    asr_engine: str
    tts_adapter: str = ""
    asr_adapter: str = ""
    contract_status: str = "not_run"
    contract_turns: int = 0
    contract_latency_seconds: float | None = None
    contract_error: str = ""
    tts_live_ready: bool = False
    tts_readiness: str = ""
    asr_live_ready: bool = False
    asr_readiness: str = ""
    live_status: str = "not_requested"
    live_quality_status: str = "not_evaluated"
    live_turns: int = 0
    live_latency_seconds: float | None = None
    live_semantic_accuracy: float | None = None
    live_agent_a_transcript: str = ""
    live_agent_b_transcript: str = ""
    live_error: str = ""
    live_quality_error: str = ""


def _module_available(module_name):
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _configured_path(value):
    raw = str(value or "").strip()
    if not raw:
        return None
    path = Path(raw)
    return path if path.exists() else None


def _isolated_python(engine, explicit, environment_dir):
    try:
        return resolve_provider_python(
            engine,
            explicit=explicit,
            environment_dir=environment_dir,
        )
    except FileNotFoundError:
        return None


def _provider_import_ready(engine, explicit, environment_dir, module_name):
    python = _isolated_python(engine, explicit, environment_dir)
    if python is None:
        return _module_available(module_name), None
    try:
        probe = subprocess.run(
            [str(python), "-c", f"import importlib; importlib.import_module({module_name!r})"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    return probe.returncode == 0, None if probe.returncode == 0 else "provider import failed"


def tts_live_readiness(engine, config):
    """Return whether an actual synthesis attempt is locally configured."""
    isolated = _isolated_python(
        engine,
        config.tts_python_executable,
        config.provider_environment_dir,
    )
    if engine == "file":
        return True, "dependency-free deterministic WAV backend"
    if engine == "sapi":
        ready = platform.system() == "Windows" and bool(shutil.which("powershell"))
        return ready, "Windows PowerShell and System.Speech required"
    if engine == "chattts":
        package, package_error = _provider_import_ready(engine, config.tts_python_executable, config.provider_environment_dir, "ChatTTS")
        model = _configured_path(config.tts_model)
        return package and bool(model), f"ChatTTS package and local assets required: {model or 'not configured'}; {package_error or 'provider import ready'}"
    if engine == "melotts":
        executable = isolated or shutil.which("melo") or shutil.which("melotts")
        return bool(executable), f"MeloTTS command: {executable or 'not found'}"
    if engine == "piper":
        model = _configured_path(config.tts_model)
        package, package_error = _provider_import_ready(engine, config.tts_python_executable, config.provider_environment_dir, "piper")
        ready = package and bool(model)
        return ready, f"piper-tts package and local ONNX voice required: {model or 'not configured'}"
    if engine == "espeak_ng":
        executable = config.tts_executable or shutil.which("espeak-ng") or shutil.which("espeak")
        return bool(executable), f"eSpeak NG executable: {executable or 'not found'}"
    if engine == "coqui":
        package, package_error = _provider_import_ready(engine, config.tts_python_executable, config.provider_environment_dir, "TTS.api")
        model = _configured_path(config.tts_model)
        return package and bool(model), f"Coqui TTS package and local assets required: {model or 'not configured'}; {package_error or 'provider import ready'}"
    if engine == "kokoro":
        package = bool(isolated) or _module_available("kokoro")
        ready = False
        return ready, (
            f"Kokoro package {'present' if package else 'missing'}; "
            "prepared local model assets required"
        )
    if engine == "f5_tts":
        executable = isolated or config.tts_executable or shutil.which("f5-tts_infer-cli")
        references = all(
            _configured_path(path)
            for path in (config.agent_a_reference_audio, config.agent_b_reference_audio)
        )
        ready = bool(executable and references)
        return ready, "F5-TTS command and reference audio for both agents required"
    if engine == "qwen3_tts":
        model = _configured_path(config.tts_model)
        package = bool(isolated) or _module_available("qwen_tts")
        ready = package and bool(model)
        return ready, (
            f"qwen-tts package {'present' if package else 'missing'}; "
            f"local model: {model or 'not configured'}"
        )
    return False, "unknown text-to-speech backend"


def asr_live_readiness(engine, config):
    """Return whether an actual recognition attempt is locally configured."""
    isolated = _isolated_python(
        engine,
        config.asr_python_executable,
        config.provider_environment_dir,
    )
    if engine == "file":
        return True, "dependency-free deterministic transcript sidecar"
    if engine == "sapi":
        ready = platform.system() == "Windows" and bool(shutil.which("powershell"))
        return ready, "Windows PowerShell and System.Speech recognizer required"
    if engine == "faster_whisper":
        model = _configured_path(config.asr_model)
        package, package_error = _provider_import_ready(engine, config.asr_python_executable, config.provider_environment_dir, "faster_whisper")
        ready = package and bool(model)
        return ready, (
            f"faster-whisper package {'present' if package else 'missing'}; "
            f"local model: {model or 'not configured'}"
        )
    if engine == "vosk":
        model = _configured_path(config.asr_model)
        package, package_error = _provider_import_ready(engine, config.asr_python_executable, config.provider_environment_dir, "vosk")
        ready = package and bool(model)
        return ready, "Vosk package and explicit local model directory required"
    if engine == "sherpa_onnx":
        model = _configured_path(config.asr_model)
        package = bool(isolated) or _module_available("sherpa_onnx")
        ready = package and bool(model and Path(model).is_dir())
        return ready, "sherpa-onnx package and explicit offline model directory required"
    if engine == "whisper_cpp":
        ready, message, resolved = whisper_cpp_ready(
            executable=config.asr_executable,
            model=config.asr_model,
            vad_model=config.asr_vad_model,
            environment_dir=config.provider_environment_dir,
        )
        return ready, (
            f"{message}; executable: {resolved['executable'] or 'not configured'}; "
            f"model: {resolved['model'] or 'not configured'}"
        )
    if engine == "parakeet":
        model = _configured_path(config.asr_model)
        package = bool(isolated) or _module_available("nemo.collections.asr")
        ready = package and bool(model)
        return ready, (
            f"NVIDIA NeMo ASR {'present' if package else 'missing'}; "
            f"local checkpoint: {model or 'not configured'}"
        )
    if engine == "qwen3_asr":
        model = _configured_path(config.asr_model)
        package = bool(isolated) or _module_available("qwen_asr")
        ready = package and bool(model)
        return ready, (
            f"qwen-asr package {'present' if package else 'missing'}; "
            f"local model: {model or 'not configured'}"
        )
    return False, "unknown automatic speech recognition backend"


def _write_probe_wave(path):
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\0\0" * 3200)


def _contract_synthesizer(audio_dir, engine_name):
    counter = 0

    def synthesize(speaker, text):
        nonlocal counter
        counter += 1
        path = Path(audio_dir) / f"contract_{counter:02d}.wav"
        _write_probe_wave(path)
        return SpeechSignal(
            speaker=speaker,
            text=text,
            audio={
                "engine": engine_name,
                "path": str(path),
                "duration_sec": 0.2,
                "waited": False,
            },
            diagnostics={"matrix_contract_tts": engine_name},
        )

    return synthesize


def _contract_transcriber(engine_name):
    def transcribe(signal):
        diagnostics = signal.diagnostics if isinstance(signal.diagnostics, dict) else {}
        signal.diagnostics = diagnostics
        diagnostics["matrix_contract_asr"] = engine_name
        return signal.text

    return transcribe


def _pipeline_config(base_config, tts_engine, asr_engine, audio_dir):
    allowed = {field.name for field in fields(SpeechPipelineConfig)}
    values = {
        key: value
        for key, value in dict(base_config or {}).items()
        if key in allowed
    }
    selected_tts = str(dict(base_config or {}).get("tts_engine") or "")
    selected_asr = str(dict(base_config or {}).get("asr_engine") or "")
    if tts_engine != selected_tts:
        values["tts_python_executable"] = ""
    if asr_engine != selected_asr:
        values["asr_python_executable"] = ""
    values.update({"tts_engine": tts_engine, "asr_engine": asr_engine})
    values = apply_speech_engine_profiles(
        values,
        replace=tuple(
            stage for stage, changed in (
                ("tts", tts_engine != selected_tts),
                ("asr", asr_engine != selected_asr),
            ) if changed
        ),
    )
    values.update({
        "audio_dir": str(audio_dir),
        "playback_enabled": False,
        "realtime_enabled": False,
        "pattern_key": "clean",
    })
    return SpeechPipelineConfig(**values)


def _run_contract_case(case, config):
    started = time.perf_counter()
    transport = None
    try:
        transport = SpeechTransport(config=config)
        case.tts_adapter = type(transport.tts_engine).__name__
        case.asr_adapter = type(transport.asr_engine).__name__
        transport.tts_engine.synthesize = _contract_synthesizer(
            config.audio_dir,
            transport.tts_engine.name,
        )
        transport.asr_engine.transcribe = _contract_transcriber(
            transport.asr_engine.name
        )
        traces = [
            transport.transmit_trace(speaker, text)
            for speaker, text in PROBE_TEXT.items()
        ]
        if not all(
            trace.pipeline_ok
            and trace.incoming_transcript == trace.generated_text
            and Path(trace.audio["path"]).is_file()
            for trace in traces
        ):
            raise SpeechPipelineError("Contract probe trace validation failed.")
        case.contract_status = "pass"
        case.contract_turns = len(traces)
    except Exception as exc:
        case.contract_status = "fail"
        case.contract_error = f"{type(exc).__name__}: {exc}"
    finally:
        if transport is not None:
            transport.close()
    case.contract_latency_seconds = round(time.perf_counter() - started, 6)


def _run_live_case(case, config):
    if (case.tts_engine == "file") != (case.asr_engine == "file"):
        case.live_status = "skipped_incompatible_control"
        case.live_quality_status = "not_applicable"
        return
    if not (case.tts_live_ready and case.asr_live_ready):
        case.live_status = "skipped_not_ready"
        return
    started = time.perf_counter()
    transport = None
    try:
        transport = SpeechTransport(config=config)
        traces = [
            transport.transmit_trace(speaker, text)
            for speaker, text in PROBE_TEXT.items()
        ]
        case.live_turns = len(traces)
        transcripts = {
            trace.speaker: trace.incoming_transcript.strip()
            for trace in traces
        }
        case.live_agent_a_transcript = transcripts.get("Agent A", "")
        case.live_agent_b_transcript = transcripts.get("Agent B", "")
        matches = []
        for speaker, expected_groups in PROBE_ENTITIES.items():
            folded = transcripts.get(speaker, "").casefold()
            matches.extend(
                any(variant in folded for variant in variants)
                for variants in expected_groups
            )
        case.live_semantic_accuracy = round(sum(matches) / len(matches), 6)
        case.live_status = "pass" if all(trace.pipeline_ok for trace in traces) else "fail"
        case.live_quality_status = "pass" if case.live_semantic_accuracy == 1.0 else "fail"
        if case.live_quality_status == "fail":
            case.live_quality_error = "Critical route entities were not preserved across both live probe turns."
    except Exception as exc:
        case.live_status = "fail"
        case.live_quality_status = "not_evaluated"
        case.live_error = f"{type(exc).__name__}: {exc}"
    finally:
        if transport is not None:
            transport.close()
    case.live_latency_seconds = round(time.perf_counter() - started, 6)


def run_speech_backend_matrix(
    output_dir,
    base_config=None,
    run_live=False,
):
    """Execute every registered TTS/ASR pairing and write its protocol."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    for tts_engine in TTS_ENGINE_SPECS:
        for asr_engine in ASR_ENGINE_SPECS:
            case = SpeechMatrixCase(
                case_id=f"tts_{tts_engine}__asr_{asr_engine}",
                tts_engine=tts_engine,
                asr_engine=asr_engine,
            )
            with tempfile.TemporaryDirectory(prefix="coop_navigation_sds_speech_matrix_") as audio_dir:
                config = _pipeline_config(base_config, tts_engine, asr_engine, audio_dir)
                case.tts_live_ready, case.tts_readiness = tts_live_readiness(
                    tts_engine,
                    config,
                )
                case.asr_live_ready, case.asr_readiness = asr_live_readiness(
                    asr_engine,
                    config,
                )
                _run_contract_case(case, config)
                if run_live:
                    _run_live_case(case, config)
            cases.append(case)
    protocol = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "asset_mode": "prepared_local_only",
        },
        "method": {
            "contract_execution": (
                "Each registered adapter pair is constructed by SpeechTransport. "
                "Backend model calls are isolated with deterministic audio and transcript "
                "probes; adapter-specific provider contracts are covered by unit tests."
            ),
            "live_execution": (
                "Actual providers are attempted only when requested and when local "
                "dependencies and models are ready. No model is downloaded implicitly."
            ),
            "probe_turns_per_case": len(PROBE_TEXT),
        },
        "summary": _matrix_summary(cases),
        "cases": [asdict(case) for case in cases],
    }
    paths = _write_matrix_protocol(output_dir, protocol)
    return protocol, paths


def _matrix_summary(cases):
    return {
        "tts_engine_count": len(TTS_ENGINE_SPECS),
        "asr_engine_count": len(ASR_ENGINE_SPECS),
        "combination_count": len(cases),
        "contract_passed": sum(case.contract_status == "pass" for case in cases),
        "contract_failed": sum(case.contract_status == "fail" for case in cases),
        "live_ready_combinations": sum(
            case.tts_live_ready
            and case.asr_live_ready
            and ((case.tts_engine == "file") == (case.asr_engine == "file"))
            for case in cases
        ),
        "live_passed": sum(case.live_status == "pass" for case in cases),
        "live_failed": sum(case.live_status == "fail" for case in cases),
        "live_skipped": sum(case.live_status.startswith("skipped_") for case in cases),
        "live_incompatible_controls": sum(case.live_status == "skipped_incompatible_control" for case in cases),
        "live_quality_passed": sum(case.live_quality_status == "pass" for case in cases),
        "live_quality_failed": sum(case.live_quality_status == "fail" for case in cases),
        "live_not_requested": sum(case.live_status == "not_requested" for case in cases),
    }


def _write_matrix_protocol(output_dir, protocol):
    json_path = output_dir / "speech_backend_matrix_protocol.json"
    csv_path = output_dir / "speech_backend_matrix_cases.csv"
    markdown_path = output_dir / "speech_backend_matrix_report.md"
    json_path.write_text(json.dumps(protocol, indent=2, ensure_ascii=True), encoding="utf-8")
    rows = protocol["cases"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = protocol["summary"]
    lines = [
        "# Speech Backend Matrix",
        "",
        f"- Text-to-speech engines: {summary['tts_engine_count']}",
        f"- Automatic speech recognition engines: {summary['asr_engine_count']}",
        f"- Combinations: {summary['combination_count']}",
        f"- Contract passed: {summary['contract_passed']}",
        f"- Contract failed: {summary['contract_failed']}",
        f"- Live-ready combinations: {summary['live_ready_combinations']}",
        f"- Live passed: {summary['live_passed']}",
        f"- Live failed: {summary['live_failed']}",
        f"- Live skipped: {summary['live_skipped']}",
        f"- Recognition quality passed: {summary['live_quality_passed']}",
        f"- Recognition quality failed: {summary['live_quality_failed']}",
        "",
        "| TTS | ASR | Contract | Live readiness | Execution | Recognition quality | Semantic accuracy |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        readiness = "ready" if row["tts_live_ready"] and row["asr_live_ready"] else "not ready"
        lines.append(
            f"| `{row['tts_engine']}` | `{row['asr_engine']}` | "
            f"{row['contract_status']} | {readiness} | {row['live_status']} | "
            f"{row['live_quality_status']} | "
            f"{row['live_semantic_accuracy'] if row['live_semantic_accuracy'] is not None else 'not run'} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "json": json_path,
        "csv": csv_path,
        "markdown": markdown_path,
    }
