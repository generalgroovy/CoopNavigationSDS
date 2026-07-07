"""Model backend adapter interfaces and Hugging Face transformer implementation used by dialog controllers.
"""
import json
import logging
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Protocol
from urllib import error, parse, request

from coop_navigation_sds.Configuration.travel import (
    CHAT_API_KEY,
    CHAT_BASE_URL,
    CHAT_MODEL,
    CHAT_TIMEOUT_SEC,
    DEVICE,
    GENERATION_MAX_TIME_SEC,
    MAX_INPUT_TOKENS,
    MAX_NEW_TOKENS,
    MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)
from coop_navigation_sds.Configuration.model_matrix import (
    AGENT_B_MODEL_SIZE_TREATMENTS,
    resolve_agent_b_model_store,
)


@dataclass(frozen=True)
class ChatMessage:
    """Provider-neutral chat message."""
    role: str
    content: str


def _ollama_model_catalog(base_url, timeout_sec=5.0):
    endpoint = f"{str(base_url).rstrip('/')}/tags"
    with request.urlopen(endpoint, timeout=timeout_sec) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return tuple(
        {
            "name": str(item.get("name") or item.get("model") or ""),
            "digest": item.get("digest"),
            "size_bytes": item.get("size"),
            "modified_at": item.get("modified_at"),
            "details": dict(item.get("details") or {}),
        }
        for item in payload.get("models", [])
        if item.get("name") or item.get("model")
    )


def _ollama_model_names(base_url, timeout_sec=5.0):
    return tuple(item["name"] for item in _ollama_model_catalog(base_url, timeout_sec))


def _warm_ollama_model(base_url, model, timeout_sec):
    payload = {
        "model": model,
        "prompt": "",
        "stream": False,
        "keep_alive": "10m",
        "options": {"num_predict": 1},
    }
    req = request.Request(
        f"{str(base_url).rstrip('/')}/generate",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with request.urlopen(req, timeout=max(5.0, float(timeout_sec))) as response:
        response.read()


def ollama_executable():
    """Return the installed Ollama command path for Windows or Linux."""
    executable = shutil.which("ollama")
    if executable:
        return executable
    if platform.system() == "Windows":
        candidate = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
        if candidate.is_file():
            return str(candidate)
    return None


def ollama_model_inventory(
    base_url,
    *,
    autostart=True,
    timeout_sec=10.0,
    models_dir=None,
):
    """Return one reachable Ollama service and its installed model names."""
    base_url = str(base_url or OLLAMA_BASE_URL).rstrip("/")
    try:
        model_records = _ollama_model_catalog(base_url, min(float(timeout_sec), 5.0))
    except (OSError, error.URLError, TimeoutError):
        parsed = parse.urlparse(base_url)
        local = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
        executable = ollama_executable() if autostart and local else None
        if not executable:
            raise RuntimeError(
                f"Ollama is not reachable at {base_url}. Start `ollama serve` or correct the service URL."
            )
        environment = dict(os.environ)
        if parsed.netloc:
            environment["OLLAMA_HOST"] = parsed.netloc
        resolved_models_dir = resolve_agent_b_model_store(models_dir)
        resolved_models_dir.mkdir(parents=True, exist_ok=True)
        environment["OLLAMA_MODELS"] = str(resolved_models_dir)
        kwargs = {
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "env": environment,
        }
        if platform.system() == "Windows":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        else:
            kwargs["start_new_session"] = True
        subprocess.Popen([executable, "serve"], **kwargs)
        deadline = time.monotonic() + max(1.0, float(timeout_sec))
        while True:
            try:
                model_records = _ollama_model_catalog(base_url, 2.0)
                break
            except (OSError, error.URLError, TimeoutError) as exc:
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"Ollama was started but did not become ready at {base_url}."
                    ) from exc
                time.sleep(0.25)
    return {
        "base_url": base_url,
        "models_dir": str(resolve_agent_b_model_store(models_dir)),
        "available_models": tuple(record["name"] for record in model_records),
        "model_records": model_records,
    }


def ensure_ollama_models_ready(
    base_url,
    models,
    *,
    autostart=True,
    timeout_sec=10.0,
    warm_models=False,
    models_dir=None,
):
    """Verify every requested Ollama model before an experiment starts."""
    requested = tuple(dict.fromkeys(str(model).strip() for model in models if str(model).strip()))
    inventory = ollama_model_inventory(
        base_url,
        autostart=autostart,
        timeout_sec=timeout_sec,
        models_dir=models_dir,
    )
    installed = set(inventory["available_models"])
    missing = tuple(model for model in requested if model not in installed)
    if missing:
        commands = " ".join(f"--model {model}" for model in missing)
        raise RuntimeError(
            "Ollama model grid is incomplete. Missing: "
            f"{', '.join(missing)}. Prepare only the missing models with: "
            f"python scripts/setup_agent_b_models.py --pull {commands}"
        )
    if warm_models:
        for model in requested:
            try:
                _warm_ollama_model(inventory["base_url"], model, timeout_sec)
            except (OSError, error.URLError, TimeoutError) as exc:
                raise RuntimeError(
                    f"Ollama model '{model}' did not become generation-ready within "
                    f"{float(timeout_sec):g} seconds."
                ) from exc
    return {
        **inventory,
        "requested_models": requested,
        "missing_models": (),
    }


def ensure_ollama_ready(
    base_url,
    model,
    *,
    autostart=True,
    timeout_sec=10.0,
    warm_model=False,
    models_dir=None,
):
    """Ensure a loopback Ollama service is reachable and contains one model."""
    model = str(model or OLLAMA_MODEL).strip()
    inventory = ollama_model_inventory(
        base_url,
        autostart=autostart,
        timeout_sec=timeout_sec,
        models_dir=models_dir,
    )
    if model not in inventory["available_models"]:
        available = ", ".join(inventory["available_models"]) or "none"
        raise RuntimeError(
            f"Ollama model '{model}' is not installed. Available models: {available}. "
            "Install it in the project store with "
            f"`python scripts/setup_agent_b_models.py --pull --model {model}`."
        )
    if warm_model:
        try:
            _warm_ollama_model(inventory["base_url"], model, timeout_sec)
        except (OSError, error.URLError, TimeoutError) as exc:
            raise RuntimeError(
                f"Ollama model '{model}' did not become generation-ready within "
                f"{float(timeout_sec):g} seconds."
            ) from exc
    return {
        "base_url": inventory["base_url"],
        "model": model,
        "models_dir": inventory["models_dir"],
        "available_models": inventory["available_models"],
    }


class ModelAdapter(Protocol):
    """Protocol for language-model adapters used by dialog controllers.
    """
    name: str
    device: str

    def generate(self, prompt: str) -> str:
        """Generate method for this module's MVC responsibility.
        
        Args:
            prompt: Input value used by `generate`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        ...

    def generate_messages(self, messages: list[ChatMessage]) -> str:
        """Generate from provider-neutral chat messages."""
        ...


@dataclass(frozen=True)
class ModelParameterSet:
    """Data model for generation parameter presets.
    """
    key: str
    do_sample: bool = False
    temperature: float | None = None
    top_p: float | None = None


@dataclass(frozen=True)
class ModelProviderSpec:
    """Selectable Agent B language-model provider metadata."""

    key: str
    label: str
    description: str
    default_model: str
    requires_api_key: bool = False


@dataclass(frozen=True)
class ModelProfileSpec:
    """Reproducible Agent B model condition independent of runtime plumbing."""

    key: str
    label: str
    provider: str
    model: str
    experimental_value: str
    approximate_memory_gb: float | None = None
    base_url: str = ""
    optional_dependency: str = ""
    size_tier: str = "custom"
    parameter_count_billion: float | None = None
    family: str = "unknown"


MODEL_PROVIDER_SPECS = {
    "transformers": ModelProviderSpec(
        "transformers",
        "Hugging Face Transformers",
        "Runs a local causal language model through PyTorch and Transformers.",
        MODEL,
    ),
    "openai_compatible": ModelProviderSpec(
        "openai_compatible",
        "OpenAI-compatible API",
        "Uses ChatGPT or another service implementing chat completions.",
        CHAT_MODEL,
        requires_api_key=True,
    ),
    "ollama": ModelProviderSpec(
        "ollama",
        "Ollama",
        "Uses a locally served Ollama chat model through its native API.",
        OLLAMA_MODEL,
    ),
    "llama_cpp": ModelProviderSpec(
        "llama_cpp",
        "llama.cpp server",
        "Uses a local llama.cpp OpenAI-compatible server for quantized GGUF models.",
        "local-model",
    ),
}


MODEL_PROFILE_SPECS = {
    "smollm2_360m_transformers": ModelProfileSpec(
        "smollm2_360m_transformers", "SmolLM2 360M Instruct", "transformers",
        "HuggingFaceTB/SmolLM2-360M-Instruct",
        "Sub-billion-parameter family contrast for low-resource instruction following.", 2.0,
        optional_dependency="transformers, torch", size_tier="small",
        parameter_count_billion=0.36, family="SmolLM2",
    ),
    "tinyllama_1b_transformers": ModelProfileSpec(
        "tinyllama_1b_transformers", "TinyLlama 1.1B Chat", "transformers",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "Small local baseline with low memory use and limited instruction following.", 4.5,
        optional_dependency="transformers, torch", size_tier="small",
        parameter_count_billion=1.1, family="Llama-derived",
    ),
    "userlm_8b_transformers": ModelProfileSpec(
        "userlm_8b_transformers", "Microsoft UserLM 8B", "transformers",
        "microsoft/UserLM-8b",
        "Dedicated research user simulator trained to predict user turns.", 34.0,
        optional_dependency="transformers, torch; approximately 32.1 GB model repository",
        size_tier="large", parameter_count_billion=8.0, family="UserLM/Llama 3",
    ),
    "qwen2_5_0_5b_transformers": ModelProfileSpec(
        "qwen2_5_0_5b_transformers", "Qwen2.5 0.5B Instruct", "transformers",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Very small multilingual instruction model for speed and resource comparisons.", 2.5,
        optional_dependency="transformers, torch", size_tier="small",
        parameter_count_billion=0.5, family="Qwen2.5",
    ),
    "smollm2_1_7b_transformers": ModelProfileSpec(
        "smollm2_1_7b_transformers", "SmolLM2 1.7B Instruct", "transformers",
        "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "Different compact model family with stronger instruction tuning than TinyLlama.", 7.0,
        optional_dependency="transformers, torch", size_tier="medium",
        parameter_count_billion=1.7, family="SmolLM2",
    ),
    "qwen2_5_1_5b_transformers": ModelProfileSpec(
        "qwen2_5_1_5b_transformers", "Qwen2.5 1.5B Instruct", "transformers",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Multilingual compact instruction model without the Ollama service dependency.", 6.0,
        optional_dependency="transformers, torch", size_tier="medium",
        parameter_count_billion=1.5, family="Qwen2.5",
    ),
    "phi3_mini_4k_transformers": ModelProfileSpec(
        "phi3_mini_4k_transformers", "Phi-3 Mini 4K Instruct", "transformers",
        "microsoft/Phi-3-mini-4k-instruct",
        "Compact reasoning-focused non-Llama family for clarification and repair tests.", 10.0,
        optional_dependency="transformers, torch", size_tier="medium",
        parameter_count_billion=3.8, family="Phi-3",
    ),
    "gemma2_2b_it_transformers": ModelProfileSpec(
        "gemma2_2b_it_transformers", "Gemma 2 2B IT", "transformers",
        "google/gemma-2-2b-it",
        "Gemma-family architecture contrast for moderate-resource instruction following.", 8.0,
        optional_dependency="transformers, torch; Hugging Face license acceptance may be required",
        size_tier="medium", parameter_count_billion=2.0, family="Gemma 2",
    ),
    "qwen3_4b_instruct_transformers": ModelProfileSpec(
        "qwen3_4b_instruct_transformers", "Qwen3 4B Instruct 2507", "transformers",
        "Qwen/Qwen3-4B-Instruct-2507",
        "Newer Qwen-family instruction model for stronger constraint and repair behavior.", 12.0,
        optional_dependency="transformers, torch", size_tier="medium",
        parameter_count_billion=4.0, family="Qwen3",
    ),
    "llama3_2_1b_ollama": ModelProfileSpec(
        "llama3_2_1b_ollama", "Llama 3.2 1B via Ollama", "ollama", "llama3.2:1b",
        "Quantized local service condition separating model behavior from Transformers runtime.", 3.0,
        base_url=OLLAMA_BASE_URL, optional_dependency="Ollama service", size_tier="small",
        parameter_count_billion=1.0, family="Llama 3.2",
    ),
    "qwen2_5_1_5b_ollama": ModelProfileSpec(
        "qwen2_5_1_5b_ollama", "Qwen2.5 1.5B via Ollama", "ollama", "qwen2.5:1.5b",
        "Larger multilingual local model for quality-versus-latency comparison.", 4.0,
        base_url=OLLAMA_BASE_URL, optional_dependency="Ollama service", size_tier="small",
        parameter_count_billion=1.5, family="Qwen2.5",
    ),
    "llama3_2_3b_ollama": ModelProfileSpec(
        "llama3_2_3b_ollama", "Llama 3.2 3B via Ollama", "ollama", "llama3.2:3b",
        "Mid-size local chat model with stronger instruction following than 1B baselines.",
        6.0, base_url=OLLAMA_BASE_URL, optional_dependency="Ollama service",
        size_tier="medium", parameter_count_billion=3.0, family="Llama 3.2",
    ),
    "phi3_3_8b_ollama": ModelProfileSpec(
        "phi3_3_8b_ollama", "Phi-3 Mini 3.8B via Ollama", "ollama", "phi3:mini",
        "Mid-size non-Llama family for reasoning and instruction-following contrast.",
        7.0, base_url=OLLAMA_BASE_URL, optional_dependency="Ollama service",
        size_tier="medium", parameter_count_billion=3.8, family="Phi-3",
    ),
    "gemma2_2b_ollama": ModelProfileSpec(
        "gemma2_2b_ollama", "Gemma 2 2B via Ollama", "ollama", "gemma2:2b",
        "Compact Gemma-family instruction model for another local architecture contrast.",
        5.0, base_url=OLLAMA_BASE_URL, optional_dependency="Ollama service",
        size_tier="medium", parameter_count_billion=2.0, family="Gemma 2",
    ),
    "qwen3_4b_ollama": ModelProfileSpec(
        "qwen3_4b_ollama", "Qwen3 4B via Ollama", "ollama", "qwen3:4b",
        "Newer Qwen-family local model for stronger route-repair behavior with moderate resources.",
        8.0, base_url=OLLAMA_BASE_URL, optional_dependency="Ollama service",
        size_tier="medium", parameter_count_billion=4.0, family="Qwen3",
    ),
    "qwen2_5_7b_ollama": ModelProfileSpec(
        "qwen2_5_7b_ollama", "Qwen2.5 7B via Ollama", "ollama", "qwen2.5:7b",
        "Large local multilingual instruction model for high-quality route dialogue.",
        10.0, base_url=OLLAMA_BASE_URL, optional_dependency="Ollama service",
        size_tier="large", parameter_count_billion=7.0, family="Qwen2.5",
    ),
    "llama3_1_8b_ollama": ModelProfileSpec(
        "llama3_1_8b_ollama", "Llama 3.1 8B via Ollama", "ollama", "llama3.1:8b",
        "Large local Llama-family baseline for quality, latency, and repair comparisons.",
        12.0, base_url=OLLAMA_BASE_URL, optional_dependency="Ollama service",
        size_tier="large", parameter_count_billion=8.0, family="Llama 3.1",
    ),
    "mistral_7b_ollama": ModelProfileSpec(
        "mistral_7b_ollama", "Mistral 7B via Ollama", "ollama", "mistral:7b",
        "Large non-Llama local baseline with different instruction-tuning behavior.",
        10.0, base_url=OLLAMA_BASE_URL, optional_dependency="Ollama service",
        size_tier="large", parameter_count_billion=7.0, family="Mistral",
    ),
    "qwen2_5_7b_transformers": ModelProfileSpec(
        "qwen2_5_7b_transformers", "Qwen2.5 7B Instruct", "transformers",
        "Qwen/Qwen2.5-7B-Instruct",
        "Large multilingual instruction model without requiring an Ollama service.", 22.0,
        optional_dependency="transformers, torch", size_tier="large",
        parameter_count_billion=7.0, family="Qwen2.5",
    ),
    "mistral_7b_transformers": ModelProfileSpec(
        "mistral_7b_transformers", "Mistral 7B Instruct v0.3", "transformers",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "Large Mistral-family model for non-Qwen/non-Llama repair behavior.", 22.0,
        optional_dependency="transformers, torch", size_tier="large",
        parameter_count_billion=7.0, family="Mistral",
    ),
    "llama3_1_8b_transformers": ModelProfileSpec(
        "llama3_1_8b_transformers", "Llama 3.1 8B Instruct", "transformers",
        "meta-llama/Llama-3.1-8B-Instruct",
        "Large general instruction model with strong expected dialogue-management behavior.", 28.0,
        optional_dependency="transformers, torch; gated Hugging Face access may be required",
        size_tier="large", parameter_count_billion=8.0, family="Llama 3.1",
    ),
    "falcon3_7b_transformers": ModelProfileSpec(
        "falcon3_7b_transformers", "Falcon3 7B Instruct", "transformers",
        "tiiuae/Falcon3-7B-Instruct",
        "Large Falcon-family model for architecture diversity without using UserLM as Agent B.", 22.0,
        optional_dependency="transformers, torch", size_tier="large",
        parameter_count_billion=7.0, family="Falcon3",
    ),
    "qwen2_5_0_5b_llama_cpp": ModelProfileSpec(
        "qwen2_5_0_5b_llama_cpp", "Qwen2.5 0.5B GGUF via llama.cpp", "llama_cpp", "local-model",
        "CPU-oriented quantized backend for Windows/Linux runtime-efficiency experiments.", 2.0,
        base_url="http://127.0.0.1:8080/v1", optional_dependency="llama.cpp server and GGUF model",
        size_tier="small", parameter_count_billion=0.5, family="Qwen2.5",
    ),
    "chatgpt_mini_api": ModelProfileSpec(
        "chatgpt_mini_api", "ChatGPT mini API condition", "openai_compatible", "gpt-4.1-mini",
        "Hosted instruction model condition for local-versus-API quality and latency comparisons.",
        base_url=CHAT_BASE_URL, optional_dependency="OpenAI-compatible API key",
        size_tier="hosted", family="OpenAI GPT",
    ),
}

AGENT_A_TINYLLAMA_PROFILE_KEY = "tinyllama_1b_transformers"
AGENT_A_USERLM_PROFILE_KEY = "userlm_8b_transformers"


def available_model_profile_keys():
    return ("custom", *MODEL_PROFILE_SPECS)


def research_model_profiles_by_tier():
    """Return the two primary Agent B model contrasts in each size tier."""
    profile_by_model = {
        spec.model: key for key, spec in MODEL_PROFILE_SPECS.items()
        if spec.provider == "ollama"
    }
    return {
        tier: tuple(
            profile_by_model[model]
            for model in treatment.models
            if model in profile_by_model
        )
        for tier, treatment in AGENT_B_MODEL_SIZE_TREATMENTS.items()
    }


def model_profile_defaults(key):
    """Return provider settings for a registered model condition."""
    spec = MODEL_PROFILE_SPECS.get(str(key or "").strip())
    if spec is None:
        return {}
    return {
        "model_profile": spec.key,
        "model_provider": spec.provider,
        "model_name": spec.model,
        "model_base_url": spec.base_url,
    }


def model_profile_metadata(key):
    spec = MODEL_PROFILE_SPECS.get(str(key or "").strip())
    return asdict(spec) if spec else None


def model_memory_requirement_gb(model, provider=None):
    """Return the registered approximate runtime memory for one model."""
    model = str(model or "").strip()
    provider = str(provider or "").strip().lower()
    matches = [
        spec for spec in MODEL_PROFILE_SPECS.values()
        if spec.model == model and (not provider or spec.provider == provider)
    ]
    return matches[0].approximate_memory_gb if len(matches) == 1 else None


def matching_model_profile(provider, model):
    """Return the registered condition matching a provider/model pair, if any."""
    provider = str(provider or "").strip().lower()
    model = str(model or "").strip()
    return next(
        (
            key for key, spec in MODEL_PROFILE_SPECS.items()
            if spec.provider == provider and spec.model == model
        ),
        "custom",
    )


def available_model_provider_keys():
    """Return the supported Agent B language-model implementations."""
    return tuple(MODEL_PROVIDER_SPECS)


def model_provider_description(key):
    spec = MODEL_PROVIDER_SPECS.get(str(key or "").strip().lower())
    return spec.description if spec else "Custom language-model provider."


def model_provider_defaults(key):
    """Return provider-specific model and endpoint defaults for configuration UIs."""
    normalized = str(key or "").strip().lower()
    spec = MODEL_PROVIDER_SPECS.get(normalized)
    if normalized == "openai_compatible":
        base_url = CHAT_BASE_URL
    elif normalized == "ollama":
        base_url = OLLAMA_BASE_URL
    elif normalized == "llama_cpp":
        base_url = "http://127.0.0.1:8080/v1"
    else:
        base_url = ""
    return {
        "model_name": spec.default_model if spec else "",
        "model_base_url": base_url,
        "model_timeout_sec": 180.0 if normalized == "ollama" else CHAT_TIMEOUT_SEC,
        "requires_api_key": bool(spec and spec.requires_api_key),
    }


MODEL_PARAMETER_SETS = {
    "greedy": ModelParameterSet("greedy", do_sample=False),
    "temp0.7": ModelParameterSet("temp0.7", do_sample=True, temperature=0.7),
    "temp1.0": ModelParameterSet("temp1.0", do_sample=True, temperature=1.0),
    "nucleus0.9": ModelParameterSet("nucleus0.9", do_sample=True, temperature=0.8, top_p=0.9),
}


def get_model_parameter_set(key: str) -> ModelParameterSet:
    """Get model parameter set function for this module's MVC responsibility.
    
    Args:
        key: Input value used by `get_model_parameter_set`; see the function signature and caller context for the expected type.
    
    Returns:
        The computed value or side effect documented by the implementation.
    """
    return MODEL_PARAMETER_SETS.get(key, MODEL_PARAMETER_SETS["greedy"])


def messages_to_prompt(messages: list[ChatMessage]) -> str:
    """Serialize chat messages for text-completion style models."""
    parts = []
    for message in messages:
        role = message.role.strip().lower()
        if role == "system":
            parts.append(f"<|system|>\n{message.content}\n</s>\n")
        elif role == "assistant":
            parts.append(f"<|assistant|>\n{message.content}\n</s>\n")
        else:
            parts.append(f"<|user|>\n{message.content}\n</s>\n")
    parts.append("<|assistant|>\n")
    return "".join(parts)


def openai_messages(messages: list[ChatMessage]) -> list[dict]:
    """Convert neutral messages to OpenAI-compatible JSON messages."""
    return [{"role": message.role, "content": message.content} for message in messages]


class TransformersModelAdapter:
    """Hugging Face transformer adapter implementing the model-generation protocol.
    """
    def __init__(
        self,
        name: str,
        tokenizer,
        model,
        device: str = DEVICE,
        max_new_tokens: int = MAX_NEW_TOKENS,
        max_input_tokens: int = MAX_INPUT_TOKENS,
        model_parameters: ModelParameterSet | None = None,
        max_time_sec: int | float | None = GENERATION_MAX_TIME_SEC,
    ):
        """  init   method for this module's MVC responsibility.
        
        Args:
            name: Input value used by `__init__`; see the function signature and caller context for the expected type.
            tokenizer: Input value used by `__init__`; see the function signature and caller context for the expected type.
            model: Input value used by `__init__`; see the function signature and caller context for the expected type.
            device: Input value used by `__init__`; see the function signature and caller context for the expected type.
            max_new_tokens: Input value used by `__init__`; see the function signature and caller context for the expected type.
            max_input_tokens: Input value used by `__init__`; see the function signature and caller context for the expected type.
            model_parameters: Input value used by `__init__`; see the function signature and caller context for the expected type.
            max_time_sec: Input value used by `__init__`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        self.name = name
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.max_input_tokens = max_input_tokens
        self.model_parameters = model_parameters or get_model_parameter_set("greedy")
        self.max_time_sec = max_time_sec
        self.generation_history = []
        if hasattr(self.model.generation_config, "max_length"):
            self.model.generation_config.max_length = None

    def with_model_params(self, model_param_key: str):
        """With model params method for this module's MVC responsibility.
        
        Args:
            model_param_key: Input value used by `with_model_params`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        return TransformersModelAdapter(
            self.name,
            self.tokenizer,
            self.model,
            device=self.device,
            max_new_tokens=self.max_new_tokens,
            max_input_tokens=self.max_input_tokens,
            model_parameters=get_model_parameter_set(model_param_key),
            max_time_sec=self.max_time_sec,
        )

    def generate(self, prompt: str) -> str:
        """Generate method for this module's MVC responsibility.
        
        Args:
            prompt: Input value used by `generate`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=False,
        )

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        if input_ids.shape[1] > self.max_input_tokens:
            logging.warning(
                "Prompt has %s tokens, above configured MAX_INPUT_TOKENS=%s. "
                "Keeping the full prompt to preserve context.",
                input_ids.shape[1],
                self.max_input_tokens,
            )

        logging.info("LLM input shape: %s", tuple(input_ids.shape))

        generation_config = deepcopy(self.model.generation_config)
        # Transformers 5 checks both this object and the model-level config
        # before merging defaults. Leave max_length unset so max_new_tokens is
        # the sole output-length control.
        if hasattr(generation_config, "max_length"):
            generation_config.max_length = None
        if hasattr(generation_config, "max_new_tokens"):
            generation_config.max_new_tokens = self.max_new_tokens
        generation_config.do_sample = self.model_parameters.do_sample
        if self.max_time_sec:
            generation_config.max_time = self.max_time_sec
        if self.model_parameters.temperature is not None:
            generation_config.temperature = self.model_parameters.temperature
        if self.model_parameters.top_p is not None:
            generation_config.top_p = self.model_parameters.top_p

        import torch

        started = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
            )

        generated = outputs[0]
        generated_length = (
            int(generated.shape[-1])
            if hasattr(generated, "shape")
            else len(generated)
        )
        output_tokens = max(0, generated_length - int(input_ids.shape[1]))
        self.generation_history.append({
            "provider": "transformers",
            "model": self.name,
            "input_tokens": int(input_ids.shape[1]),
            "output_tokens": output_tokens,
            "latency_sec": round(time.perf_counter() - started, 6),
        })
        return self.tokenizer.decode(
            outputs[0][input_ids.shape[1]:],
            skip_special_tokens=True,
        )

    def generate_messages(self, messages: list[ChatMessage]) -> str:
        """Generate from chat messages using tokenizer chat templates when available."""
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                prompt = self.tokenizer.apply_chat_template(
                    openai_messages(messages),
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                prompt = messages_to_prompt(messages)
        else:
            prompt = messages_to_prompt(messages)
        return self.generate(prompt)


class OpenAICompatibleChatAdapter:
    """Adapter for OpenAI/ChatGPT-compatible chat-completions APIs."""

    device = "api"

    def __init__(
        self,
        model: str = CHAT_MODEL,
        api_key: str | None = CHAT_API_KEY,
        base_url: str = CHAT_BASE_URL,
        timeout_sec: float = CHAT_TIMEOUT_SEC,
        max_new_tokens: int = MAX_NEW_TOKENS,
        model_parameters: ModelParameterSet | None = None,
    ):
        hostname = parse.urlparse(base_url).hostname
        if not api_key and hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise RuntimeError("An API key is required for the OpenAI-compatible provider.")
        self.name = model
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.max_new_tokens = max_new_tokens
        self.model_parameters = model_parameters or get_model_parameter_set("greedy")
        self.generation_history = []

    def with_model_params(self, model_param_key: str):
        return OpenAICompatibleChatAdapter(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            timeout_sec=self.timeout_sec,
            max_new_tokens=self.max_new_tokens,
            model_parameters=get_model_parameter_set(model_param_key),
        )

    def generate(self, prompt: str) -> str:
        return self.generate_messages([ChatMessage("user", prompt)])

    def generate_messages(self, messages: list[ChatMessage]) -> str:
        payload = {
            "model": self.model,
            "messages": openai_messages(messages),
            "max_tokens": self.max_new_tokens,
        }
        if self.model_parameters.temperature is not None:
            payload["temperature"] = self.model_parameters.temperature
        if self.model_parameters.top_p is not None:
            payload["top_p"] = self.model_parameters.top_p

        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            method="POST",
            headers=headers,
        )
        started = time.perf_counter()
        with request.urlopen(req, timeout=self.timeout_sec) as response:
            data = json.loads(response.read().decode("utf-8"))
        usage = data.get("usage") or {}
        self.generation_history.append({
            "provider": "openai_compatible",
            "model": self.model,
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "latency_sec": round(time.perf_counter() - started, 6),
        })
        return data["choices"][0]["message"]["content"]


class OllamaChatAdapter:
    """Adapter for a local Ollama `/api/chat` endpoint."""

    device = "local-api"

    def __init__(
        self,
        model: str = OLLAMA_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        timeout_sec: float = CHAT_TIMEOUT_SEC,
        max_new_tokens: int = MAX_NEW_TOKENS,
        model_parameters: ModelParameterSet | None = None,
    ):
        self.name = model
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.max_new_tokens = max_new_tokens
        self.model_parameters = model_parameters or get_model_parameter_set("greedy")
        self.generation_history = []

    def with_model_params(self, model_param_key: str):
        return OllamaChatAdapter(
            model=self.model,
            base_url=self.base_url,
            timeout_sec=self.timeout_sec,
            max_new_tokens=self.max_new_tokens,
            model_parameters=get_model_parameter_set(model_param_key),
        )

    def generate(self, prompt: str) -> str:
        return self.generate_messages([ChatMessage("user", prompt)])

    def generate_messages(self, messages: list[ChatMessage]) -> str:
        options = {"num_predict": self.max_new_tokens}
        if not self.model_parameters.do_sample:
            options["temperature"] = 0.0
        elif self.model_parameters.temperature is not None:
            options["temperature"] = self.model_parameters.temperature
        if self.model_parameters.top_p is not None:
            options["top_p"] = self.model_parameters.top_p
        payload = {
            "model": self.model,
            "messages": openai_messages(messages),
            "stream": False,
            "keep_alive": "10m",
            "options": options,
        }
        req = request.Request(
            f"{self.base_url}/chat",
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        started = time.perf_counter()
        try:
            with request.urlopen(req, timeout=self.timeout_sec) as response:
                data = json.loads(response.read().decode("utf-8"))
        except TimeoutError as exc:
            raise TimeoutError(
                f"Ollama model '{self.model}' exceeded the configured "
                f"{self.timeout_sec:g}-second response timeout."
            ) from exc
        self.generation_history.append({
            "provider": "ollama",
            "model": self.model,
            "input_tokens": data.get("prompt_eval_count"),
            "output_tokens": data.get("eval_count"),
            "latency_sec": round(time.perf_counter() - started, 6),
            "load_duration_ns": data.get("load_duration"),
            "evaluation_duration_ns": data.get("eval_duration"),
        })
        return data["message"]["content"]


def model_adapter_runtime_metadata(adapter, *, provider=None, profile="custom", roles=()):
    """Return provider-neutral audit metadata for an adapter actually used in a run."""
    if adapter is None:
        return {
            "used": False,
            "profile": "not_applicable",
            "profile_metadata": None,
            "provider": "none",
            "model": None,
            "roles": [],
            "generation_history": [],
        }
    if provider is None:
        if isinstance(adapter, TransformersModelAdapter):
            provider = "transformers"
        elif isinstance(adapter, OllamaChatAdapter):
            provider = "ollama"
        elif isinstance(adapter, OpenAICompatibleChatAdapter):
            provider = "openai_compatible"
        else:
            provider = type(adapter).__name__
    return {
        "used": True,
        "profile": profile or "custom",
        "profile_metadata": model_profile_metadata(profile),
        "provider": provider,
        "model": getattr(adapter, "name", type(adapter).__name__),
        "roles": list(roles),
        "generation_history": list(getattr(adapter, "generation_history", [])),
    }
