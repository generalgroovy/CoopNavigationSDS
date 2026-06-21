"""Selectable experiment component catalog for GUI and batch tooling."""
import json
from pathlib import Path
import platform

from coop_navigation_sds.Configuration.speech import AGENT_B_PLUGIN, speech_pattern_keys
from coop_navigation_sds.DialogManagement.speech_pipeline import (
    available_asr_engine_keys,
    available_tts_engine_keys,
)
from coop_navigation_sds.NaturalLanguageGeneration.assistant.plugin_registry import (
    available_agent_b_plugin_keys,
)
from coop_navigation_sds.NaturalLanguageGeneration.caller.config import PERSONAS
from coop_navigation_sds.NaturalLanguageGeneration.caller.responder import available_agent_a_types
from coop_navigation_sds.NaturalLanguageGeneration.models import available_model_provider_keys
from coop_navigation_sds.TextToSpeech.personas import audio_persona_keys
from coop_navigation_sds.TransportNetwork.constraints import OBJECTIVE_MODES
from coop_navigation_sds.TransportNetwork.test_cases import TEST_CASES


PROJECT_ROOT = Path(__file__).resolve().parents[2]


TTS_RUNTIME_PROFILES = {
    "sapi": {"tts_model": "", "tts_executable": "", "tts_python_executable": ""},
    "chattts": {"tts_model": ".speech-providers/models/chattts", "tts_executable": "", "tts_python_executable": "", "tts_timeout_sec": 180.0},
    "piper": {"tts_model": ".speech-providers/models/piper/en_US-lessac-medium.onnx", "tts_executable": "", "tts_python_executable": ""},
    "espeak_ng": {"tts_model": "", "tts_executable": "", "tts_python_executable": ""},
    "coqui": {"tts_model": ".speech-providers/models/coqui", "tts_executable": "", "tts_python_executable": ""},
    "file": {"tts_model": "", "tts_executable": "", "tts_python_executable": ""},
}

ASR_RUNTIME_PROFILES = {
    "sapi": {"asr_model": "", "asr_executable": "", "asr_python_executable": "", "asr_vad_model": ""},
    "faster_whisper": {"asr_model": ".speech-providers/models/faster-whisper", "asr_executable": "", "asr_python_executable": "", "asr_vad_model": ""},
    "vosk": {"asr_model": ".speech-providers/models/vosk/vosk-model-small-en-us-0.15", "asr_executable": "", "asr_python_executable": "", "asr_vad_model": ""},
    "whisper_cpp": {"asr_model": ".speech-providers/models/whisper.cpp/ggml-base.en.bin", "asr_executable": "", "asr_python_executable": "", "asr_vad_model": ""},
    "qwen3_asr": {"asr_model": ".speech-providers/models/qwen3-asr", "asr_executable": "", "asr_python_executable": "", "asr_vad_model": ""},
    "sherpa_onnx": {"asr_model": ".speech-providers/models/sherpa-onnx", "asr_executable": "", "asr_python_executable": "", "asr_vad_model": ""},
    "file": {"asr_model": "", "asr_executable": "", "asr_python_executable": "", "asr_vad_model": ""},
}


def speech_engine_profile(stage, engine):
    """Return an independent copy of one backend's local runtime defaults."""
    profiles = TTS_RUNTIME_PROFILES if stage == "tts" else ASR_RUNTIME_PROFILES
    key = str(engine or "").strip().lower()
    profile = dict(profiles.get(key, {}))
    if stage == "asr" and key == "faster_whisper":
        cache_root = (PROJECT_ROOT / profile["asr_model"]).resolve()
        snapshots = sorted(cache_root.glob("models--*/snapshots/*/model.bin"))
        if snapshots:
            profile["asr_model"] = str(snapshots[-1].parent)
    for field, value in tuple(profile.items()):
        if isinstance(value, str) and value.startswith(".speech-providers"):
            profile[field] = str((PROJECT_ROOT / value).resolve())
    return profile


def resolve_prepared_asset_path(value):
    """Anchor project-managed provider paths independently of process cwd."""
    text = str(value or "").strip()
    if not text:
        return text
    path = Path(text).expanduser()
    if path.is_absolute():
        return str(path)
    if path.parts and path.parts[0] == ".speech-providers":
        return str((PROJECT_ROOT / path).resolve())
    return text


def apply_speech_engine_profiles(config, *, replace=False):
    """Complete a configuration without carrying settings across backends."""
    resolved = dict(config or {})
    replace_stages = (
        {"tts", "asr"}
        if replace is True
        else set(replace) if isinstance(replace, (tuple, list, set, frozenset)) else set()
    )
    for stage in ("tts", "asr"):
        profile = speech_engine_profile(stage, resolved.get(f"{stage}_engine"))
        for key, value in profile.items():
            if (
                stage in replace_stages
                or key not in resolved
                or resolved[key] is None
                or (resolved[key] == "" and value != "")
            ):
                resolved[key] = value
    return resolved


def startup_choices(extra_agent_b_plugin=AGENT_B_PLUGIN):
    """Return all configurable implementation and scenario choices."""
    manifest_path = Path(__file__).with_name("platform_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    platform_choices = manifest.get(platform.system(), manifest.get("Linux", {}))
    tts_engines = [
        key for key in available_tts_engine_keys()
        if key in platform_choices.get("text_to_speech", ())
    ]
    asr_engines = [
        key for key in available_asr_engine_keys()
        if key in platform_choices.get("automatic_speech_recognition", ())
    ]
    return {
        "test_case_keys": list(TEST_CASES),
        "persona_keys": list(PERSONAS),
        "agent_a_types": available_agent_a_types(),
        "agent_b_plugins": available_agent_b_plugin_keys(extra_agent_b_plugin),
        "model_providers": available_model_provider_keys(),
        "speech_patterns": speech_pattern_keys(),
        "tts_engines": tts_engines,
        "asr_engines": asr_engines,
        "agent_a_objective_modes": list(OBJECTIVE_MODES),
        "agent_a_audio_personas": audio_persona_keys("caller"),
        "agent_b_audio_personas": audio_persona_keys("assistant"),
    }
