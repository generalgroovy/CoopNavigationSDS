"""Selectable experiment component catalog for GUI and batch tooling."""
import json
from pathlib import Path
import platform

from coop_navigation_sds.Configuration.assets import (
    resolve_faster_whisper_model,
    resolve_project_asset_path,
)
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
from coop_navigation_sds.NaturalLanguageGeneration.models import (
    available_model_profile_keys,
    available_model_provider_keys,
    model_profile_defaults,
    research_model_profiles_by_tier,
)
from coop_navigation_sds.Configuration.pipeline import component_status
from coop_navigation_sds.TextToSpeech.personas import audio_persona_keys
from coop_navigation_sds.TransportNetwork.constraints import OBJECTIVE_MODES
from coop_navigation_sds.TransportNetwork.test_cases import TEST_CASES


PROJECT_ROOT = Path(__file__).resolve().parents[2]


TTS_RUNTIME_PROFILES = {
    "sapi": {"tts_model": "", "tts_executable": "", "tts_python_executable": ""},
    "chattts": {"tts_model": ".speech-providers/models/chattts", "tts_executable": "", "tts_python_executable": "", "tts_timeout_sec": 600.0},
    "piper": {"tts_model": ".speech-providers/models/piper/en_US-lessac-medium.onnx", "tts_executable": "", "tts_python_executable": ""},
    "espeak_ng": {"tts_model": "", "tts_executable": "", "tts_python_executable": ""},
    "coqui": {"tts_model": ".speech-providers/models/coqui", "tts_executable": "", "tts_python_executable": "", "tts_timeout_sec": 180.0},
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

INTERACTIVE_TTS_ENGINES = ("chattts", "piper", "coqui", "espeak_ng")
INTERACTIVE_ASR_ENGINES = ("faster_whisper", "vosk", "whisper_cpp", "sherpa_onnx")


def _with_current_choice(values, current):
    """Return values plus the current setting, preserving order and uniqueness."""
    items = [str(value) for value in values if str(value or "").strip()]
    selected = str(current or "").strip()
    if selected and selected not in items:
        items.append(selected)
    return list(dict.fromkeys(items))


def speech_engine_profile(stage, engine):
    """Return an independent copy of one backend's local runtime defaults."""
    profiles = TTS_RUNTIME_PROFILES if stage == "tts" else ASR_RUNTIME_PROFILES
    key = str(engine or "").strip().lower()
    profile = dict(profiles.get(key, {}))
    if stage == "asr" and key == "faster_whisper":
        profile["asr_model"] = resolve_faster_whisper_model(profile["asr_model"])
    for field, value in tuple(profile.items()):
        if isinstance(value, str) and value.startswith(".speech-providers"):
            profile[field] = str((PROJECT_ROOT / value).resolve())
    return profile


def resolve_prepared_asset_path(value):
    """Anchor project-managed provider paths independently of process cwd."""
    return resolve_project_asset_path(value)


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
    resolved["tts_model"] = resolve_project_asset_path(resolved.get("tts_model"))
    resolved["asr_model"] = resolve_project_asset_path(resolved.get("asr_model"))
    if str(resolved.get("asr_engine", "")).strip().lower() == "faster_whisper":
        resolved["asr_model"] = resolve_faster_whisper_model(resolved.get("asr_model"))
    return resolved


def startup_choices(extra_agent_b_plugin=AGENT_B_PLUGIN, config=None, *, operational_only=False):
    """Return implementation choices, optionally restricted to ready providers."""
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
    base_config = dict(config or {})
    status_rows = {"tts": {}, "asr": {}, "model_profiles": {}}
    for stage, engines in (("tts", tts_engines), ("asr", asr_engines)):
        for engine in engines:
            engine_config = apply_speech_engine_profiles(
                {**base_config, f"{stage}_engine": engine},
                replace=(stage,),
            )
            status_rows[stage][engine] = component_status(stage, engine, engine_config)
    tier_profiles = [
        key
        for keys in research_model_profiles_by_tier().values()
        for key in keys
    ]
    all_profiles = list(available_model_profile_keys())
    model_profiles = list(dict.fromkeys((
        "custom",
        *tier_profiles,
        *(key for key in all_profiles if key not in {"custom", *tier_profiles}),
    )))
    for profile in model_profiles:
        if profile == "custom":
            profile_config = base_config
        else:
            profile_config = {**base_config, **model_profile_defaults(profile)}
        provider = profile_config.get("model_provider", "")
        status_rows["model_profiles"][profile] = component_status(
            "model", provider, profile_config
        )
    if operational_only:
        tts_engines = [
            key for key in tts_engines
            if key in INTERACTIVE_TTS_ENGINES and status_rows["tts"][key].available
        ]
        asr_engines = [
            key for key in asr_engines
            if key in INTERACTIVE_ASR_ENGINES and status_rows["asr"][key].available
        ]
        model_profiles = [
            key for key in model_profiles
            if status_rows["model_profiles"][key].available
        ]
        model_providers = list(dict.fromkeys(
            model_profile_defaults(key).get("model_provider")
            if key != "custom" else base_config.get("model_provider")
            for key in model_profiles
        ))
        model_providers = [key for key in model_providers if key]
    else:
        model_providers = list(available_model_provider_keys())
    return {
        "test_case_keys": list(TEST_CASES),
        "persona_keys": list(PERSONAS),
        "agent_a_types": available_agent_a_types(),
        "agent_b_plugins": _with_current_choice(
            available_agent_b_plugin_keys(extra_agent_b_plugin),
            base_config.get("agent_b_plugin"),
        ),
        "model_providers": _with_current_choice(
            model_providers,
            base_config.get("model_provider"),
        ),
        "model_profiles": _with_current_choice(
            model_profiles,
            base_config.get("model_profile"),
        ),
        "speech_patterns": speech_pattern_keys(),
        "tts_engines": _with_current_choice(tts_engines, base_config.get("tts_engine")),
        "asr_engines": _with_current_choice(asr_engines, base_config.get("asr_engine")),
        "agent_a_objective_modes": list(OBJECTIVE_MODES),
        "agent_a_audio_personas": audio_persona_keys("caller"),
        "agent_b_audio_personas": audio_persona_keys("assistant"),
        "component_statuses": {
            group: {
                key: {
                    "available": status.available,
                    "reason": status.reason,
                }
                for key, status in statuses.items()
            }
            for group, statuses in status_rows.items()
        },
    }
