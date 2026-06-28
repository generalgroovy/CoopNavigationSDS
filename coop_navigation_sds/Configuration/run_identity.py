"""Compact run identifiers with full meaning retained in the run manifest."""
from __future__ import annotations

import json
import re
from pathlib import Path


NAMING_SCHEME_VERSION = "2026-06-23.3"


NAMING_SCHEME = {
    "STG": "Agent A deterministic staged caller",
    "TLA": "Agent A TinyLlama caller",
    "ULM": "Agent A UserLM caller using the selected language model",
    "SIM": "Agent B simple deterministic planner",
    "LLM": "Agent B configurable language-model backend",
    "PAR": "Agent B Pareto deterministic planner",
    "ROB": "Agent B robust deterministic planner",
    "DIV": "Agent B diverse deterministic planner",
    "CTT": "ChatTTS text-to-speech",
    "PIP": "Piper text-to-speech",
    "ESP": "eSpeak NG text-to-speech",
    "COQ": "Coqui text-to-speech",
    "SAP": "Windows SAPI speech engine",
    "FWH": "Faster-Whisper automatic speech recognition",
    "VOS": "Vosk automatic speech recognition",
    "WCP": "whisper.cpp automatic speech recognition",
    "SHO": "sherpa-onnx automatic speech recognition",
    "TL1": "TinyLlama 1.1B Chat language model",
    "Q05": "Qwen2.5 0.5B Instruct language model",
    "S36": "SmolLM2 360M Instruct language model",
    "S17": "SmolLM2 1.7B Instruct language model",
    "L31": "Llama 3.2 1B via Ollama language model",
    "L33": "Llama 3.2 3B via Ollama language model",
    "PH3": "Phi-3 Mini 3.8B via Ollama language model",
    "G22": "Gemma 2 2B via Ollama language model",
    "Q34": "Qwen3 4B via Ollama language model",
    "Q15": "Qwen2.5 1.5B via Ollama language model",
    "Q27": "Qwen2.5 7B via Ollama language model",
    "L38": "Llama 3.1 8B via Ollama language model",
    "M17": "Mistral 7B via Ollama language model",
    "QCP": "Qwen2.5 0.5B via llama.cpp language model",
    "CGM": "ChatGPT mini API language model",
    "SML": "Small Agent B language-model size tier (1.0B-1.5B parameters)",
    "MED": "Medium Agent B language-model size tier (3.0B-3.8B parameters)",
    "LRG": "Large Agent B language-model size tier (7.0B-8.0B parameters)",
}


COMPONENT_CODES = {
    "staged": "STG",
    "tinyllama": "TLA",
    "userlm": "ULM",
    "simple": "SIM",
    "llm": "LLM",
    "pareto": "PAR",
    "robust": "ROB",
    "diverse": "DIV",
    "chattts": "CTT",
    "piper": "PIP",
    "espeak_ng": "ESP",
    "coqui": "COQ",
    "sapi": "SAP",
    "faster_whisper": "FWH",
    "vosk": "VOS",
    "whisper_cpp": "WCP",
    "sherpa_onnx": "SHO",
    "tinyllama_1b_transformers": "TL1",
    "tinyllama/tinyllama-1.1b-chat-v1.0": "TL1",
    "qwen2_5_0_5b_transformers": "Q05",
    "qwen/qwen2.5-0.5b-instruct": "Q05",
    "smollm2_360m_transformers": "S36",
    "huggingfacetb/smollm2-360m-instruct": "S36",
    "smollm2_1_7b_transformers": "S17",
    "huggingfacetb/smollm2-1.7b-instruct": "S17",
    "llama3_2_1b_ollama": "L31",
    "llama3.2:1b": "L31",
    "llama3_2_3b_ollama": "L33",
    "llama3.2:3b": "L33",
    "llama3.2:latest": "L33",
    "phi3_3_8b_ollama": "PH3",
    "phi3:mini": "PH3",
    "gemma2_2b_ollama": "G22",
    "gemma2:2b": "G22",
    "qwen3_4b_ollama": "Q34",
    "qwen3:4b": "Q34",
    "qwen2_5_1_5b_ollama": "Q15",
    "qwen2.5:1.5b": "Q15",
    "qwen2_5_7b_ollama": "Q27",
    "qwen2.5:7b": "Q27",
    "llama3_1_8b_ollama": "L38",
    "llama3.1:8b": "L38",
    "mistral_7b_ollama": "M17",
    "mistral:7b": "M17",
    "qwen2_5_0_5b_llama_cpp": "QCP",
    "chatgpt_mini_api": "CGM",
    "gpt-4.1-mini": "CGM",
    "small": "SML",
    "medium": "MED",
    "large": "LRG",
}


def naming_scheme_document():
    """Return the abbreviation registry persisted with result artifacts."""
    return {
        "schema_version": NAMING_SCHEME_VERSION,
        "description": "Keys are compact abbreviations used in result folder and condition names; values describe the configuration setting.",
        "codes": dict(sorted(NAMING_SCHEME.items())),
        "source": "coop_navigation_sds.Configuration.run_identity",
    }


def write_naming_scheme(output_dir):
    """Write or update the naming-scheme registry in a result directory."""
    path = Path(output_dir) / "naming_scheme.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    document = naming_scheme_document()
    path.write_text(json.dumps(document, indent=2, ensure_ascii=True), encoding="utf-8")
    return path


def compact_code(value, width=5):
    """Return a deterministic readable code for an unregistered value."""
    text = str(value or "NA").strip().lower()
    if text in COMPONENT_CODES:
        return COMPONENT_CODES[text]
    words = re.findall(r"[a-z0-9]+", text)
    if not words:
        return "NA"
    initials = "".join(word[0] for word in words)
    code = initials if len(initials) >= 2 else words[0]
    return code[: max(2, int(width))].upper()


def single_run_label(config):
    """Encode the main independent variables without long folder names."""
    plugin = str(config.get("agent_b_plugin") or "llm").strip().lower()
    model = (
        config.get("model_profile") or config.get("model_name")
        if plugin == "llm"
        else plugin
    )
    return "-".join((
        "R",
        compact_code(config.get("test_case_key")),
        compact_code(config.get("persona_key")),
        compact_code(config.get("agent_a_type")),
        compact_code(model),
        compact_code(config.get("tts_engine")),
        compact_code(config.get("asr_engine")),
        f"S{int(config.get('network_seed', 0))}",
    ))


def batch_run_label(condition_count):
    return f"B-N{max(0, int(condition_count)):04d}"
