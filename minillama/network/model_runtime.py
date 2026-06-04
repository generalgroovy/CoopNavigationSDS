"""Runtime loader for tokenizer/model objects and factory helpers that create model adapters for controllers.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from minillama.network.config import ALLOW_MODEL_DOWNLOAD, DEVICE, MODEL, MODEL_PROVIDER, TOKEN
from minillama.network.model_adapters import OpenAICompatibleChatAdapter, TransformersModelAdapter


def load_model_and_tokenizer(model_name: str = MODEL, token: str | None = TOKEN, device: str = DEVICE):
    """Load model and tokenizer function for this module's MVC responsibility.
    
    Args:
        model_name: Input value used by `load_model_and_tokenizer`; see the function signature and caller context for the expected type.
        token: Input value used by `load_model_and_tokenizer`; see the function signature and caller context for the expected type.
        device: Input value used by `load_model_and_tokenizer`; see the function signature and caller context for the expected type.
    
    Returns:
        The computed value or side effect documented by the implementation.
    """
    tokenizer = _load_tokenizer(model_name, token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = _load_model(model_name, token, device)
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
):
    """Create transformers adapter function for this module's MVC responsibility.
    
    Args:
        model_name: Input value used by `create_transformers_adapter`; see the function signature and caller context for the expected type.
        token: Input value used by `create_transformers_adapter`; see the function signature and caller context for the expected type.
        device: Input value used by `create_transformers_adapter`; see the function signature and caller context for the expected type.
    
    Returns:
        The computed value or side effect documented by the implementation.
    """
    tokenizer, model = load_model_and_tokenizer(model_name, token, device)
    return TransformersModelAdapter(model_name, tokenizer, model, device=device)


def create_model_adapter(provider: str = MODEL_PROVIDER):
    """Create the configured provider-neutral model adapter."""
    if provider == "openai":
        return OpenAICompatibleChatAdapter()
    if provider == "transformers":
        return create_transformers_adapter()
    raise ValueError(f"Unsupported MODEL_PROVIDER: {provider}")


def _load_tokenizer(model_name: str, token: str | None):
    """ load tokenizer function for this module's MVC responsibility.
    
    Args:
        model_name: Input value used by `_load_tokenizer`; see the function signature and caller context for the expected type.
        token: Input value used by `_load_tokenizer`; see the function signature and caller context for the expected type.
    
    Returns:
        The computed value or side effect documented by the implementation.
    """
    try:
        return AutoTokenizer.from_pretrained(model_name, token=token, local_files_only=True)
    except OSError:
        if not ALLOW_MODEL_DOWNLOAD:
            raise RuntimeError(
                f"Tokenizer weights for {model_name} are not available locally. "
                "Set MINILLAMA_ALLOW_MODEL_DOWNLOAD=1 to download them, or use the simple Agent B plugin."
            )
        return AutoTokenizer.from_pretrained(model_name, token=token)


def _load_model(model_name: str, token: str | None, device: str):
    """ load model function for this module's MVC responsibility.
    
    Args:
        model_name: Input value used by `_load_model`; see the function signature and caller context for the expected type.
        token: Input value used by `_load_model`; see the function signature and caller context for the expected type.
        device: Input value used by `_load_model`; see the function signature and caller context for the expected type.
    
    Returns:
        The computed value or side effect documented by the implementation.
    """
    dtype = torch.float16 if device == "cuda" else torch.float32
    model_kwargs = {
        "token": token,
        "dtype": dtype,
        "low_cpu_mem_usage": True,
    }

    try:
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            local_files_only=True,
            **model_kwargs,
        )
    except OSError:
        if not ALLOW_MODEL_DOWNLOAD:
            raise RuntimeError(
                f"Model weights for {model_name} are not available locally. "
                "Set MINILLAMA_ALLOW_MODEL_DOWNLOAD=1 to download them, or use the simple Agent B plugin."
            )
        return AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
