"""Canonical Agent B model-size treatments used by research job matrices."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSizeTreatment:
    """One controlled model-size tier with two family-diverse local models."""

    key: str
    minimum_parameters_billion: float
    maximum_parameters_billion: float
    models: tuple[str, str]


AGENT_B_MODEL_SIZE_TREATMENTS = {
    "small": ModelSizeTreatment(
        "small", 1.0, 1.5, ("llama3.2:1b", "qwen2.5:1.5b")
    ),
    "medium": ModelSizeTreatment(
        "medium", 3.0, 3.8, ("llama3.2:3b", "phi3:mini")
    ),
    "large": ModelSizeTreatment(
        "large", 7.0, 8.0, ("qwen2.5:7b", "llama3.1:8b")
    ),
}


def models_for_size_treatments(treatment_keys):
    """Resolve ordered model names for one or more declared size treatments."""
    models = []
    for key in treatment_keys:
        normalized = str(key).strip().lower()
        if normalized not in AGENT_B_MODEL_SIZE_TREATMENTS:
            available = ", ".join(AGENT_B_MODEL_SIZE_TREATMENTS)
            raise ValueError(
                f"Unknown Agent B model-size treatment '{key}'. Use one of: {available}."
            )
        models.extend(AGENT_B_MODEL_SIZE_TREATMENTS[normalized].models)
    return tuple(dict.fromkeys(models))


def model_size_treatment(model_name):
    """Return the configured size-treatment key for a model, if registered."""
    model_name = str(model_name or "").strip()
    return next(
        (
            key
            for key, treatment in AGENT_B_MODEL_SIZE_TREATMENTS.items()
            if model_name in treatment.models
        ),
        None,
    )
