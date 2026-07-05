"""Install and verify the speech stack in a supported project Python runtime."""
from argparse import ArgumentParser
import importlib
import json
from pathlib import Path
import subprocess
import sys
import shutil


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coop_navigation_sds.TextToSpeech.setup import (
    MAXIMUM_PROJECT_PYTHON_EXCLUSIVE,
    MINIMUM_PROJECT_PYTHON,
    PROJECT_PROVIDER_PROFILE,
    PROVIDER_PROFILES,
    project_python_supported,
)
from coop_navigation_sds.DialogManagement.whisper_cpp_runtime import whisper_cpp_ready
from coop_navigation_sds.DialogManagement.speech_pipeline import resolve_espeak_executable


MODULES = {
    "chattts": "ChatTTS",
    "faster_whisper": "faster_whisper",
    "piper": "piper",
    "qwen3_asr": "qwen_asr",
    "sherpa_onnx": "sherpa_onnx",
    "vosk": "vosk",
}


def prepare_project_layout(provider_dir):
    """Create the committed manifest's runtime directories without model downloads."""
    provider_dir = Path(provider_dir)
    platform_manifest = ROOT / "coop_navigation_sds" / "Configuration" / "platform_manifest.json"
    layout = json.loads(platform_manifest.read_text(encoding="utf-8"))
    for configured in layout.get("runtime_model_assets", {}).values():
        path = ROOT / configured
        path.mkdir(parents=True, exist_ok=True)
    provider_dir.mkdir(parents=True, exist_ok=True)
    provider_manifest = provider_dir / "providers.json"
    if not provider_manifest.exists():
        provider_manifest.write_text(
            json.dumps({"schema_version": 1, "providers": {}}, indent=2),
            encoding="utf-8",
        )
    return provider_manifest


def register_whisper_cpp(provider_dir, executable, model, vad_model=""):
    """Register an existing whisper.cpp executable and GGML model."""
    provider_dir = Path(provider_dir)
    provider_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = provider_dir / "providers.json"
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        document = {"providers": {}}
    providers = document.setdefault("providers", {})
    providers["whisper_cpp"] = {
        "executable": str(Path(executable).expanduser().resolve()),
        "model": str(Path(model).expanduser().resolve()),
    }
    if vad_model:
        providers["whisper_cpp"]["vad_model"] = str(Path(vad_model).expanduser().resolve())
    manifest_path.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def run(command):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run([str(value) for value in command], check=True)


def register_provider_python(provider_dir, engine, python):
    manifest_path = prepare_project_layout(provider_dir)
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    document.setdefault("providers", {})[engine] = {"python": str(Path(python).resolve())}
    manifest_path.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")


def prepare_coqui_provider(provider_dir, python311):
    if not python311 or not Path(python311).is_file():
        print("Coqui provider skipped: configure --coqui-python with Python 3.10 or 3.11.")
        return False
    environment = Path(provider_dir) / "coqui"
    python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not python.is_file():
        run((python311, "-m", "venv", environment))
    run((python, "-m", "pip", "install", "--upgrade", "pip", "wheel"))
    run((python, "-m", "pip", "install", "torch", "torchaudio", "coqui-tts==0.27.5"))
    register_provider_python(provider_dir, "coqui", python)
    return True


def interpreter_details(python):
    probe = subprocess.run(
        [
            str(python),
            "-c",
            (
                "import json,sys;"
                "print(json.dumps({'executable':sys.executable,"
                "'version':[sys.version_info.major,sys.version_info.minor]}))"
            ),
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(probe.stdout)


def verify_current_runtime(provider_dir=".speech-providers"):
    status = {}
    errors = {}
    for engine, module in MODULES.items():
        try:
            if engine == "chattts":
                from coop_navigation_sds.DialogManagement.speech_pipeline import ChatTTSTextToSpeech, SpeechPipelineConfig

                ChatTTSTextToSpeech(SpeechPipelineConfig())._import_chattts()
            else:
                importlib.import_module(module)
            status[engine] = True
        except Exception as exc:
            status[engine] = False
            errors[engine] = f"{type(exc).__name__}: {exc}"
    ready, message, resolved = whisper_cpp_ready(environment_dir=provider_dir)
    status["whisper_cpp"] = ready
    if not ready:
        errors["whisper_cpp"] = message
    manifest_path = Path(provider_dir) / "providers.json"
    try:
        coqui_entry = json.loads(manifest_path.read_text(encoding="utf-8"))["providers"]["coqui"]
        coqui_python = coqui_entry["python"] if isinstance(coqui_entry, dict) else coqui_entry
        coqui_probe = subprocess.run(
            [str(coqui_python), "-c", "from TTS.api import TTS"],
            text=True,
            capture_output=True,
            timeout=60,
        )
        status["coqui"] = coqui_probe.returncode == 0
        if not status["coqui"]:
            errors["coqui"] = coqui_probe.stderr.strip().splitlines()[-1]
    except Exception as exc:
        status["coqui"] = False
        errors["coqui"] = f"{type(exc).__name__}: {exc}"
    status["espeak_ng"] = bool(resolve_espeak_executable())
    if not status["espeak_ng"]:
        errors["espeak_ng"] = "eSpeak NG executable not found"
    status["sapi"] = sys.platform == "win32" and bool(shutil.which("powershell"))
    status["file"] = True
    print(json.dumps({
        "python": sys.executable,
        "version": list(sys.version_info[:2]),
        "providers": status,
        "errors": errors,
        "whisper_cpp": resolved,
    }, indent=2, sort_keys=True))
    return project_python_supported(sys.version_info[:2]) and all(
        status[engine] for engine in MODULES
    )


def main():
    parser = ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--provider-dir", default=".speech-providers")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--register-whisper-cpp", action="store_true")
    parser.add_argument("--whisper-cpp-executable", default="")
    parser.add_argument("--whisper-cpp-model", default="")
    parser.add_argument("--whisper-cpp-vad-model", default="")
    default_coqui_python = (
        ROOT / ".runtime" / "python311" / "python.exe"
        if sys.platform == "win32"
        else Path(shutil.which("python3.11") or "")
    )
    parser.add_argument("--coqui-python", default=str(default_coqui_python))
    parser.add_argument("--coqui-only", action="store_true")
    args = parser.parse_args()
    prepare_project_layout(args.provider_dir)

    if args.coqui_only:
        raise SystemExit(0 if prepare_coqui_provider(args.provider_dir, args.coqui_python) else 2)

    if args.register_whisper_cpp:
        ready, message, _resolved = whisper_cpp_ready(
            executable=args.whisper_cpp_executable,
            model=args.whisper_cpp_model,
            vad_model=args.whisper_cpp_vad_model,
            environment_dir=args.provider_dir,
        )
        if not ready:
            raise RuntimeError(
                f"Cannot register whisper.cpp: {message}. "
                "Pass --whisper-cpp-executable and --whisper-cpp-model."
            )
        path = register_whisper_cpp(
            args.provider_dir,
            args.whisper_cpp_executable,
            args.whisper_cpp_model,
            args.whisper_cpp_vad_model,
        )
        print(f"Registered whisper.cpp in {path}.")
        return

    if args.status:
        raise SystemExit(0 if verify_current_runtime(args.provider_dir) else 1)

    details = interpreter_details(args.python)
    if not project_python_supported(details["version"]):
        raise RuntimeError(
            "Speech providers require a project Python version from "
            f"{MINIMUM_PROJECT_PYTHON[0]}.{MINIMUM_PROJECT_PYTHON[1]} through "
            f"{MAXIMUM_PROJECT_PYTHON_EXCLUSIVE[0]}.{MAXIMUM_PROJECT_PYTHON_EXCLUSIVE[1] - 1}; "
            f"received {details['executable']} version {details['version']}."
        )

    profile = PROVIDER_PROFILES[PROJECT_PROVIDER_PROFILE]
    run((args.python, "-m", "pip", "install", "--upgrade", "pip", "wheel"))
    run((args.python, "-m", "pip", "install", *profile.packages))
    run((args.python, "-m", "pip", "check"))
    prepare_coqui_provider(args.provider_dir, args.coqui_python)
    print(
        "Installed the supported speech stack in "
        f"{details['executable']} (Python {'.'.join(map(str, details['version']))})."
    )


if __name__ == "__main__":
    main()
