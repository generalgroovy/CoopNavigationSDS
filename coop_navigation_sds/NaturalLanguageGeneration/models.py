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
from dataclasses import dataclass
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


@dataclass(frozen=True)
class ChatMessage:
    """Provider-neutral chat message."""
    role: str
    content: str


def _ollama_model_names(base_url, timeout_sec=5.0):
    endpoint = f"{str(base_url).rstrip('/')}/tags"
    with request.urlopen(endpoint, timeout=timeout_sec) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return tuple(
        str(item.get("name") or item.get("model") or "")
        for item in payload.get("models", [])
        if item.get("name") or item.get("model")
    )


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


def _ollama_executable():
    executable = shutil.which("ollama")
    if executable:
        return executable
    if platform.system() == "Windows":
        candidate = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
        if candidate.is_file():
            return str(candidate)
    return None


def ensure_ollama_ready(base_url, model, *, autostart=True, timeout_sec=10.0, warm_model=False):
    """Ensure a loopback Ollama service is reachable and contains the selected model."""
    base_url = str(base_url or OLLAMA_BASE_URL).rstrip("/")
    model = str(model or OLLAMA_MODEL).strip()
    try:
        names = _ollama_model_names(base_url, min(float(timeout_sec), 5.0))
    except (OSError, error.URLError, TimeoutError):
        parsed = parse.urlparse(base_url)
        local = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
        executable = _ollama_executable() if autostart and local else None
        if not executable:
            raise RuntimeError(
                f"Ollama is not reachable at {base_url}. Start `ollama serve` or correct the service URL."
            )
        environment = dict(os.environ)
        if parsed.netloc:
            environment["OLLAMA_HOST"] = parsed.netloc
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
                names = _ollama_model_names(base_url, 2.0)
                break
            except (OSError, error.URLError, TimeoutError) as exc:
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"Ollama was started but did not become ready at {base_url}."
                    ) from exc
                time.sleep(0.25)
    if model not in names:
        available = ", ".join(names) or "none"
        raise RuntimeError(
            f"Ollama model '{model}' is not installed. Available models: {available}. "
            f"Install it before the experiment with `ollama pull {model}`."
        )
    if warm_model:
        try:
            _warm_ollama_model(base_url, model, timeout_sec)
        except (OSError, error.URLError, TimeoutError) as exc:
            raise RuntimeError(
                f"Ollama model '{model}' did not become generation-ready within "
                f"{float(timeout_sec):g} seconds."
            ) from exc
    return {"base_url": base_url, "model": model, "available_models": names}


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
}


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

        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
            )

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
        if not api_key:
            raise RuntimeError("An API key is required for the OpenAI-compatible provider.")
        self.name = model
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.max_new_tokens = max_new_tokens
        self.model_parameters = model_parameters or get_model_parameter_set("greedy")

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
        req = request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        with request.urlopen(req, timeout=self.timeout_sec) as response:
            data = json.loads(response.read().decode("utf-8"))
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
        try:
            with request.urlopen(req, timeout=self.timeout_sec) as response:
                data = json.loads(response.read().decode("utf-8"))
        except TimeoutError as exc:
            raise TimeoutError(
                f"Ollama model '{self.model}' exceeded the configured "
                f"{self.timeout_sec:g}-second response timeout."
            ) from exc
        return data["message"]["content"]
