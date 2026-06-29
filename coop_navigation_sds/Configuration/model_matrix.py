"""Canonical Agent B model-size treatments used by research job matrices."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import platform
import re


PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGENT_B_MODEL_STORE_ROOT = Path(".model-providers") / "agent_b"
AGENT_B_OLLAMA_BASE_URL = os.environ.get(
    "COOP_NAVIGATION_SDS_OLLAMA_BASE_URL",
    os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11435/api"),
)


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

MODEL_SIZE_ORDER = tuple(AGENT_B_MODEL_SIZE_TREATMENTS)


def model_store_platform(system_name=None):
    """Return the stable project-folder key for a supported operating system."""
    system_name = str(system_name or platform.system()).strip().lower()
    if system_name.startswith("win"):
        return "windows"
    if system_name.startswith("linux"):
        return "linux"
    raise RuntimeError(
        f"Agent B local model storage supports Windows and Linux, not '{system_name}'."
    )


def agent_b_model_platform_dir(system_name=None, *, project_root=PROJECT_ROOT):
    """Return the platform-specific root containing model data and its catalog."""
    return Path(project_root) / AGENT_B_MODEL_STORE_ROOT / model_store_platform(system_name)


def agent_b_ollama_store_dir(system_name=None, *, project_root=PROJECT_ROOT):
    """Return the platform-specific Ollama blob and manifest store."""
    return agent_b_model_platform_dir(system_name, project_root=project_root) / "ollama"


def resolve_agent_b_model_store(path=None, system_name=None, *, project_root=PROJECT_ROOT):
    """Resolve an optional model-store setting against the repository root."""
    selected = Path(path).expanduser() if path else agent_b_ollama_store_dir(
        system_name,
        project_root=project_root,
    )
    if not selected.is_absolute():
        selected = Path(project_root) / selected
    return selected.resolve()


def model_catalog_folder(model_name):
    """Return a size-first, stable catalog path for one registered model."""
    tier = model_size_treatment(model_name)
    if tier is None:
        raise ValueError(f"Agent B model '{model_name}' is not registered.")
    tier_index = MODEL_SIZE_ORDER.index(tier) + 1
    model_index = AGENT_B_MODEL_SIZE_TREATMENTS[tier].models.index(model_name) + 1
    slug = re.sub(r"[^a-z0-9]+", "-", model_name.lower()).strip("-")
    return Path(f"{tier_index:02d}-{tier}") / f"{model_index:02d}-{slug}"


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
