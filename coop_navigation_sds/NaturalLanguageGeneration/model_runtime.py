"""Runtime loader and provider-neutral model-adapter factory."""
import os
from pathlib import Path

from coop_navigation_sds.Configuration.travel import (
    ALLOW_MODEL_DOWNLOAD,
    CHAT_API_KEY,
    CHAT_BASE_URL,
    CHAT_MODEL,
    CHAT_TIMEOUT_SEC,
    DEVICE,
    MAX_INPUT_TOKENS,
    MAX_NEW_TOKENS,
    MODEL,
    MODEL_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    TOKEN,
)
from coop_navigation_sds.NaturalLanguageGeneration.models import (
    OllamaChatAdapter,
    OpenAICompatibleChatAdapter,
    TransformersModelAdapter,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _model_cache_dir():
    configured = Path(os.environ.get(
        "COOP_NAVIGATION_SDS_MODEL_CACHE_DIR",
        ".speech-providers/models/huggingface",
    )).expanduser()
    if not configured.is_absolute():
        configured = PROJECT_ROOT / configured
    return configured.resolve()


MODEL_CACHE_DIR = str(_model_cache_dir())
USERLM_MODEL_NAME = "microsoft/UserLM-8b"


def _prepared_model(model_name):
    configured = Path(model_name).expanduser()
    if configured.exists():
        return str(configured.resolve())
    prepared = Path(MODEL_CACHE_DIR) / model_name.replace("/", "--")
    return str(prepared.resolve()) if prepared.is_dir() else model_name


def _trust_remote_code(model_name):
    """Return whether a registered model explicitly requires Hub model code."""
    normalized = str(model_name).replace("\\", "/").rstrip("/")
    return normalized.endswith((USERLM_MODEL_NAME, "microsoft--UserLM-8b"))


def load_model_and_tokenizer(
    model_name: str = MODEL,
    token: str | None = TOKEN,
    device: str = DEVICE,
    allow_model_download: bool = ALLOW_MODEL_DOWNLOAD,
):
    """Load model and tokenizer function for this module's MVC responsibility.
    
    Args:
        model_name: Input value used by `load_model_and_tokenizer`; see the function signature and caller context for the expected type.
        token: Input value used by `load_model_and_tokenizer`; see the function signature and caller context for the expected type.
        device: Input value used by `load_model_and_tokenizer`; see the function signature and caller context for the expected type.
    
    Returns:
        The computed value or side effect documented by the implementation.
    """
    tokenizer = _load_tokenizer(model_name, token, allow_model_download)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = _load_model(model_name, token, device, allow_model_download)
    model.to(device)
    model.eval()
    model.generation_config.do_sample = False
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.use_cache = True

    return tokenizer, model


def create_transformers_adapter(
    model_name: str = MODEL,
    token: str | None = TOKEN,
    device: str = DEVICE,
    max_new_tokens: int = MAX_NEW_TOKENS,
    max_input_tokens: int = MAX_INPUT_TOKENS,
    allow_model_download: bool = ALLOW_MODEL_DOWNLOAD,
):
    """Create transformers adapter function for this module's MVC responsibility.
    
    Args:
        model_name: Input value used by `create_transformers_adapter`; see the function signature and caller context for the expected type.
        token: Input value used by `create_transformers_adapter`; see the function signature and caller context for the expected type.
        device: Input value used by `create_transformers_adapter`; see the function signature and caller context for the expected type.
    
    Returns:
        The computed value or side effect documented by the implementation.
    """
    tokenizer, model = load_model_and_tokenizer(
        model_name,
        token,
        device,
        allow_model_download=allow_model_download,
    )
    return TransformersModelAdapter(
        model_name,
        tokenizer,
        model,
        device=device,
        max_new_tokens=max_new_tokens,
        max_input_tokens=max_input_tokens,
    )


def create_model_adapter(
    provider: str = MODEL_PROVIDER,
    *,
    model_name: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout_sec: float = CHAT_TIMEOUT_SEC,
    device: str = DEVICE,
    max_new_tokens: int = MAX_NEW_TOKENS,
    max_input_tokens: int = MAX_INPUT_TOKENS,
    allow_model_download: bool = ALLOW_MODEL_DOWNLOAD,
):
    """Create the configured provider-neutral model adapter."""
    provider = str(provider or MODEL_PROVIDER).strip().lower()
    if provider in {"openai", "openai_compatible", "llama_cpp"}:
        return OpenAICompatibleChatAdapter(
            model=model_name or ("local-model" if provider == "llama_cpp" else CHAT_MODEL),
            api_key=api_key or CHAT_API_KEY,
            base_url=base_url or (
                "http://127.0.0.1:8080/v1" if provider == "llama_cpp" else CHAT_BASE_URL
            ),
            timeout_sec=timeout_sec,
            max_new_tokens=max_new_tokens,
        )
    if provider == "ollama":
        return OllamaChatAdapter(
            model=model_name or OLLAMA_MODEL,
            base_url=base_url or OLLAMA_BASE_URL,
            timeout_sec=timeout_sec,
            max_new_tokens=max_new_tokens,
        )
    if provider == "transformers":
        return create_transformers_adapter(
            model_name=model_name or MODEL,
            device=device,
            max_new_tokens=max_new_tokens,
            max_input_tokens=max_input_tokens,
            allow_model_download=allow_model_download,
        )
    raise ValueError(f"Unsupported MODEL_PROVIDER: {provider}")


def _load_tokenizer(
    model_name: str,
    token: str | None,
    allow_model_download: bool = ALLOW_MODEL_DOWNLOAD,
):
    """ load tokenizer function for this module's MVC responsibility.
    
    Args:
        model_name: Input value used by `_load_tokenizer`; see the function signature and caller context for the expected type.
        token: Input value used by `_load_tokenizer`; see the function signature and caller context for the expected type.
    
    Returns:
        The computed value or side effect documented by the implementation.
    """
    from transformers import AutoTokenizer
    trust_remote_code = _trust_remote_code(model_name)
    model_name = _prepared_model(model_name)

    try:
        return AutoTokenizer.from_pretrained(
            model_name,
            token=token,
            cache_dir=MODEL_CACHE_DIR,
            local_files_only=not allow_model_download,
            trust_remote_code=trust_remote_code,
        )
    except OSError:
        raise RuntimeError(
            f"Tokenizer weights for {model_name} are unavailable. "
            + (
                "Check network access and model authorization."
                if allow_model_download
                else "Prepare the model locally or explicitly enable model downloads."
            )
        )


def _load_model(
    model_name: str,
    token: str | None,
    device: str,
    allow_model_download: bool = ALLOW_MODEL_DOWNLOAD,
):
    """ load model function for this module's MVC responsibility.
    
    Args:
        model_name: Input value used by `_load_model`; see the function signature and caller context for the expected type.
        token: Input value used by `_load_model`; see the function signature and caller context for the expected type.
        device: Input value used by `_load_model`; see the function signature and caller context for the expected type.
    
    Returns:
        The computed value or side effect documented by the implementation.
    """
    import torch
    from transformers import AutoModelForCausalLM
    trust_remote_code = _trust_remote_code(model_name)
    model_name = _prepared_model(model_name)

    dtype = torch.float16 if device == "cuda" else torch.float32
    model_kwargs = {
        "token": token,
        "dtype": dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": trust_remote_code,
    }

    try:
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=MODEL_CACHE_DIR,
            local_files_only=not allow_model_download,
            **model_kwargs,
        )
    except OSError:
        raise RuntimeError(
            f"Model weights for {model_name} are unavailable. "
            + (
                "Check network access, disk capacity, and model authorization."
                if allow_model_download
                else "Prepare the model locally or explicitly enable model downloads."
            )
        )
