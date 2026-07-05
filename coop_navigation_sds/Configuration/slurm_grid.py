"""Deterministic, scheduler-neutral condition grids for Slurm array jobs."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from coop_navigation_sds.Configuration.schema import safe_artifact_name
from coop_navigation_sds.Configuration.speech import speech_pattern_keys
from coop_navigation_sds.Configuration.travel import NETWORK_SEED
from coop_navigation_sds.NaturalLanguageGeneration.caller.config import PERSONAS
from coop_navigation_sds.NaturalLanguageGeneration.assistant.plugin_registry import (
    available_agent_b_plugin_keys,
)
from coop_navigation_sds.NaturalLanguageGeneration.models import model_profile_defaults
from coop_navigation_sds.TransportNetwork.test_cases import TEST_CASES


SLURM_GRID_SCHEMA_VERSION = 1
RUN_MODES = ("pure_text", "speech")


def _tinyllama_backend():
    values = model_profile_defaults("tinyllama_1b_transformers")
    return {
        "key": "minillama",
        "plugin": "llm",
        "model_profile": "tinyllama_1b_transformers",
        "model_provider": values["model_provider"],
        "model_name": values["model_name"],
        "model_base_url": values.get("model_base_url", ""),
    }


BUILTIN_AGENT_B_BACKENDS = {
    "simple": {
        "key": "simple",
        "plugin": "simple",
    },
    "minillama": _tinyllama_backend(),
}


def _immutable_mapping(values=None):
    return MappingProxyType(dict(values or {}))


def _canonical_hash(values, length=12):
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:length]


def _registry_selection(value, registry, label):
    if value in (None, "all"):
        return tuple(sorted(registry))
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be 'all' or a non-empty list.")
    unknown = sorted(set(value) - set(registry))
    if unknown:
        raise ValueError(f"Unknown {label}: {', '.join(unknown)}")
    return tuple(value)


def _task_profiles(document):
    """Resolve validated case/persona pairs without creating blind cross-products."""
    configured = document.get("task_profiles")
    if configured == "standard":
        return tuple(
            (key, test_case.persona_key)
            for key, test_case in sorted(TEST_CASES.items())
        )
    if configured is not None:
        if not isinstance(configured, list) or not configured:
            raise ValueError("task_profiles must be 'standard' or a non-empty list.")
        profiles = []
        for index, profile in enumerate(configured):
            if not isinstance(profile, dict):
                raise ValueError(f"task_profiles[{index}] must be an object.")
            test_case = str(profile.get("test_case") or "")
            persona = str(profile.get("persona") or "")
            if test_case not in TEST_CASES:
                raise ValueError(f"Unknown task_profiles[{index}] test case '{test_case}'.")
            if persona not in PERSONAS:
                raise ValueError(f"Unknown task_profiles[{index}] persona '{persona}'.")
            profiles.append((test_case, persona))
        if len(set(profiles)) != len(profiles):
            raise ValueError("task_profiles contains duplicate case/persona pairs.")
        return tuple(profiles)
    personas = _registry_selection(document.get("personas", "all"), PERSONAS, "personas")
    test_cases = _registry_selection(document.get("test_cases", "all"), TEST_CASES, "test cases")
    return tuple((test_case, persona) for persona in personas for test_case in test_cases)


@dataclass(frozen=True)
class AgentBBackend:
    """One plugin/model treatment independent of scheduler resources."""

    key: str
    plugin: str
    model_profile: str | None = None
    model_provider: str | None = None
    model_name: str | None = None
    model_base_url: str = ""
    extra_config: Mapping = field(default_factory=lambda: _immutable_mapping())

    @classmethod
    def from_value(cls, value):
        if isinstance(value, str):
            if value not in BUILTIN_AGENT_B_BACKENDS:
                raise ValueError(
                    f"Unknown Agent B backend '{value}'. Custom backends require an object."
                )
            value = BUILTIN_AGENT_B_BACKENDS[value]
        if not isinstance(value, dict):
            raise ValueError("Agent B backends must be built-in keys or configuration objects.")
        key = str(value.get("key") or "").strip()
        plugin = str(value.get("plugin") or "").strip()
        if not key or not plugin:
            raise ValueError("Every Agent B backend requires non-empty 'key' and 'plugin' values.")
        builtins = set(available_agent_b_plugin_keys())
        if plugin not in builtins and ":" not in plugin:
            raise ValueError(
                f"Custom Agent B plugin '{plugin}' must use package.module:factory syntax."
            )
        known = {
            "key", "plugin", "model_profile", "model_provider", "model_name",
            "model_base_url",
        }
        return cls(
            key=key,
            plugin=plugin,
            model_profile=value.get("model_profile"),
            model_provider=value.get("model_provider"),
            model_name=value.get("model_name"),
            model_base_url=str(value.get("model_base_url") or ""),
            extra_config=_immutable_mapping({k: v for k, v in value.items() if k not in known}),
        )

    def as_dict(self):
        values = {
            "key": self.key,
            "plugin": self.plugin,
            "model_profile": self.model_profile,
            "model_provider": self.model_provider,
            "model_name": self.model_name,
            "model_base_url": self.model_base_url,
            **dict(self.extra_config),
        }
        return {key: value for key, value in values.items() if value not in (None, "")}


@dataclass(frozen=True)
class SlurmCondition:
    """One immutable scheduler-array treatment."""

    grid_name: str
    index: int
    backend: AgentBBackend
    persona_key: str
    test_case_key: str
    run_mode: str
    speech_pattern_key: str | None
    seed: int
    repetition: int
    agent_a_type: str
    speech_config: Mapping = field(default_factory=lambda: _immutable_mapping())
    base_config: Mapping = field(default_factory=lambda: _immutable_mapping())

    def experimental_factors(self):
        values = {
            "grid_name": self.grid_name,
            "index": self.index,
            "agent_b": self.backend.as_dict(),
            "persona_key": self.persona_key,
            "test_case_key": self.test_case_key,
            "run_mode": self.run_mode,
            "seed": self.seed,
            "repetition": self.repetition,
            "agent_a_type": self.agent_a_type,
        }
        if self.run_mode == "speech":
            values["speech_pattern_key"] = self.speech_pattern_key
            values["speech"] = dict(self.speech_config)
        return values

    @property
    def condition_id(self):
        digest = _canonical_hash(self.experimental_factors())
        return f"SC{self.index:06d}-{safe_artifact_name(self.backend.key, 24)}-{digest}"

    def as_dict(self, runtime_overrides=None):
        values = {
            "schema_version": SLURM_GRID_SCHEMA_VERSION,
            "condition_id": self.condition_id,
            **self.experimental_factors(),
            "base_config": dict(self.base_config),
            "runtime_overrides": dict(runtime_overrides or {}),
        }
        return values


@dataclass(frozen=True)
class SlurmConditionGrid:
    """Validated condition collection with stable index ordering."""

    name: str
    conditions: tuple[SlurmCondition, ...]
    source: str = ""

    @classmethod
    def from_document(cls, document, source=""):
        if int(document.get("schema_version", SLURM_GRID_SCHEMA_VERSION)) != SLURM_GRID_SCHEMA_VERSION:
            raise ValueError("Unsupported Slurm grid schema version.")
        name = safe_artifact_name(document.get("name") or Path(source or "grid").stem)
        backends = tuple(AgentBBackend.from_value(value) for value in document.get("agent_b", ()))
        if not backends:
            raise ValueError("Slurm grid requires at least one Agent B backend.")
        task_profiles = _task_profiles(document)
        run_modes = tuple(document.get("run_modes") or ("pure_text",))
        invalid_modes = sorted(set(run_modes) - set(RUN_MODES))
        if invalid_modes:
            raise ValueError(f"Unknown run modes: {', '.join(invalid_modes)}")
        patterns = tuple(document.get("speech_patterns") or ("clean",))
        invalid_patterns = sorted(set(patterns) - set(speech_pattern_keys()))
        if invalid_patterns:
            raise ValueError(f"Unknown speech patterns: {', '.join(invalid_patterns)}")
        seeds = tuple(int(value) for value in document.get("seeds", (NETWORK_SEED,)))
        if not seeds:
            raise ValueError("Slurm grid requires at least one seed.")
        repetitions = max(1, int(document.get("repetitions", 1)))
        agent_a_type = str(document.get("agent_a_type") or "staged")
        speech = _immutable_mapping(document.get("speech"))
        if "speech" in run_modes:
            missing_speech = [key for key in ("tts_engine", "asr_engine") if not speech.get(key)]
            if missing_speech:
                raise ValueError(
                    "Speech runs require speech configuration fields: "
                    + ", ".join(missing_speech)
                )
        base_config = _immutable_mapping(document.get("base_config"))

        conditions = []
        for backend in backends:
            for test_case, persona in task_profiles:
                for run_mode in run_modes:
                    active_patterns = patterns if run_mode == "speech" else (None,)
                    for pattern in active_patterns:
                        for seed in seeds:
                            for repetition in range(repetitions):
                                conditions.append(SlurmCondition(
                                    grid_name=name,
                                    index=len(conditions),
                                    backend=backend,
                                    persona_key=persona,
                                    test_case_key=test_case,
                                    run_mode=run_mode,
                                    speech_pattern_key=pattern,
                                    seed=seed,
                                    repetition=repetition,
                                    agent_a_type=agent_a_type,
                                    speech_config=speech,
                                    base_config=base_config,
                                ))
        return cls(name=name, conditions=tuple(conditions), source=str(source or ""))

    @classmethod
    def from_file(cls, path):
        path = Path(path).expanduser().resolve()
        return cls.from_document(json.loads(path.read_text(encoding="utf-8")), source=str(path))

    def condition(self, index):
        index = int(index)
        if index < 0 or index >= len(self.conditions):
            raise IndexError(
                f"Condition index {index} is outside 0..{len(self.conditions) - 1}."
            )
        return self.conditions[index]


def condition_directory_name(condition, runtime_overrides=None):
    """Return a bounded directory name unique to factors and runtime overrides."""
    document = condition.as_dict(runtime_overrides)
    digest = _canonical_hash(document, length=10)
    mode = "txt" if condition.run_mode == "pure_text" else "sp"
    return safe_artifact_name(
        f"{condition.condition_id}-{mode}-s{condition.seed}-r{condition.repetition}-{digest}"
    )


def reserve_condition_directory(results_root, condition, runtime_overrides=None):
    """Atomically reserve a new task folder without overwriting a prior attempt."""
    root = Path(results_root).expanduser().resolve() / safe_artifact_name(condition.grid_name)
    root.mkdir(parents=True, exist_ok=True)
    base = condition_directory_name(condition, runtime_overrides)
    for attempt in range(1, 10000):
        suffix = "" if attempt == 1 else f"-a{attempt:02d}"
        candidate = root / f"{base}{suffix}"
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not reserve a unique result directory for {condition.condition_id}.")


def export_condition_json(condition, path, runtime_overrides=None):
    """Write the exact resolved condition once; refuse accidental replacement."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = condition.as_dict(runtime_overrides)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
    return path
