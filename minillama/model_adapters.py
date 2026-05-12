"""Model backend adapter interfaces and Hugging Face transformer implementation used by dialog controllers.
"""
import json
import logging
from dataclasses import dataclass
from typing import Protocol
from urllib import request

import torch

from minillama.config import (
    CHAT_API_KEY,
    CHAT_BASE_URL,
    CHAT_MODEL,
    CHAT_TIMEOUT_SEC,
    DEVICE,
    GENERATION_MAX_TIME_SEC,
    MAX_INPUT_TOKENS,
    MAX_NEW_TOKENS,
)


@dataclass(frozen=True)
class ChatMessage:
    """Provider-neutral chat message."""
    role: str
    content: str


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

        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.model_parameters.do_sample,
        }
        if self.max_time_sec:
            generation_kwargs["max_time"] = self.max_time_sec
        if self.model_parameters.temperature is not None:
            generation_kwargs["temperature"] = self.model_parameters.temperature
        if self.model_parameters.top_p is not None:
            generation_kwargs["top_p"] = self.model_parameters.top_p

        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_kwargs,
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
            raise RuntimeError("OPENAI_API_KEY is required for MODEL_PROVIDER=openai.")
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
