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
    """One controlled model-size tier with provider-compatible model slots."""

    key: str
    minimum_parameters_billion: float
    maximum_parameters_billion: float
    models: tuple[str, ...]


@dataclass(frozen=True)
class AgentBModelProposal:
    """Documented Agent B candidate beyond the two canonical size slots."""

    size_tier: str
    slot: str
    model: str
    provider: str
    unique_aspect: str
    use_case: str


AGENT_B_MODEL_SIZE_TREATMENTS = {
    "small": ModelSizeTreatment(
        "small", 0.3, 1.7, (
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "Qwen/Qwen2.5-0.5B-Instruct",
            "HuggingFaceTB/SmolLM2-360M-Instruct",
            "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        )
    ),
    "medium": ModelSizeTreatment(
        "medium", 1.5, 4.0, (
            "Qwen/Qwen2.5-1.5B-Instruct",
            "microsoft/Phi-3-mini-4k-instruct",
            "google/gemma-2-2b-it",
            "Qwen/Qwen3-4B-Instruct-2507",
        )
    ),
    "large": ModelSizeTreatment(
        "large", 7.0, 8.0, (
            "Qwen/Qwen2.5-7B-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct",
            "tiiuae/Falcon3-7B-Instruct",
        )
    ),
}

MODEL_SIZE_ORDER = tuple(AGENT_B_MODEL_SIZE_TREATMENTS)


AGENT_B_MODEL_PROPOSALS = {
    "small": (
        AgentBModelProposal(
            "small", "small1", "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "transformers",
            "contained small chat baseline already used as the control model",
            "low-resource local dialogue-system baseline and Agent A control match",
        ),
        AgentBModelProposal(
            "small", "small2", "Qwen/Qwen2.5-0.5B-Instruct", "transformers",
            "very small multilingual Qwen-family model",
            "tests provider-compatible floor performance and multilingual route grounding",
        ),
        AgentBModelProposal(
            "small", "small3", "HuggingFaceTB/SmolLM2-360M-Instruct", "transformers",
            "sub-billion Transformer baseline with very low memory",
            "tests floor performance and fast smoke-test behavior",
        ),
        AgentBModelProposal(
            "small", "small4", "HuggingFaceTB/SmolLM2-1.7B-Instruct", "transformers",
            "compact model above the 1B boundary",
            "tests small-to-medium transition behavior",
        ),
    ),
    "medium": (
        AgentBModelProposal(
            "medium", "medium1", "Qwen/Qwen2.5-1.5B-Instruct", "transformers",
            "medium-lite multilingual Qwen-family model",
            "bridges compact models and heavier reasoning-focused models",
        ),
        AgentBModelProposal(
            "medium", "medium2", "microsoft/Phi-3-mini-4k-instruct", "transformers",
            "non-Llama Phi architecture with compact reasoning focus",
            "tests model-family effects on clarification and repair",
        ),
        AgentBModelProposal(
            "medium", "medium3", "google/gemma-2-2b-it", "transformers",
            "Gemma-family instruction model with lower memory than Phi",
            "tests another architecture at moderate local cost",
        ),
        AgentBModelProposal(
            "medium", "medium4", "Qwen/Qwen3-4B-Instruct-2507", "transformers",
            "newer Qwen generation with stronger multilingual instruction behavior",
            "tests whether newer training improves constraint handling",
        ),
    ),
    "large": (
        AgentBModelProposal(
            "large", "large1", "Qwen/Qwen2.5-7B-Instruct", "transformers",
            "large multilingual Qwen-family comparison model",
            "tests high-capacity route grounding and constraint handling",
        ),
        AgentBModelProposal(
            "large", "large3", "meta-llama/Llama-3.1-8B-Instruct", "transformers",
            "large Llama-family instruction baseline",
            "tests expected high-quality dialogue management against other 7B/8B families",
        ),
        AgentBModelProposal(
            "large", "large4", "tiiuae/Falcon3-7B-Instruct", "transformers",
            "large Falcon-family assistant model replacing UserLM as Agent B candidate",
            "tests architecture diversity without using a user simulator as the assistant",
        ),
    ),
}


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


def agent_b_model_proposals(size_tier=None):
    """Return documented model candidates grouped by experimental size tier."""
    if size_tier is None:
        return {
            key: tuple(proposals)
            for key, proposals in AGENT_B_MODEL_PROPOSALS.items()
        }
    normalized = str(size_tier).strip().lower()
    if normalized not in AGENT_B_MODEL_PROPOSALS:
        available = ", ".join(AGENT_B_MODEL_PROPOSALS)
        raise ValueError(f"Unknown model proposal tier '{size_tier}'. Use one of: {available}.")
    return tuple(AGENT_B_MODEL_PROPOSALS[normalized])


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
