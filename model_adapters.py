import logging
from dataclasses import dataclass
from typing import Protocol

import torch

from config import DEVICE, MAX_INPUT_TOKENS, MAX_NEW_TOKENS


class ModelAdapter(Protocol):
    name: str
    device: str

    def generate(self, prompt: str) -> str:
        ...


@dataclass(frozen=True)
class ModelParameterSet:
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
    return MODEL_PARAMETER_SETS.get(key, MODEL_PARAMETER_SETS["greedy"])


class TransformersModelAdapter:
    def __init__(
        self,
        name: str,
        tokenizer,
        model,
        device: str = DEVICE,
        max_new_tokens: int = MAX_NEW_TOKENS,
        max_input_tokens: int = MAX_INPUT_TOKENS,
        model_parameters: ModelParameterSet | None = None,
    ):
        self.name = name
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.max_input_tokens = max_input_tokens
        self.model_parameters = model_parameters or get_model_parameter_set("greedy")

    def with_model_params(self, model_param_key: str):
        return TransformersModelAdapter(
            self.name,
            self.tokenizer,
            self.model,
            device=self.device,
            max_new_tokens=self.max_new_tokens,
            max_input_tokens=self.max_input_tokens,
            model_parameters=get_model_parameter_set(model_param_key),
        )

    def generate(self, prompt: str) -> str:
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
