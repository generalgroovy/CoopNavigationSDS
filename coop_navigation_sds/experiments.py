"""Batch experiment controller that executes condition grids and serializes metric records.
"""
import time
import hashlib
import traceback
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, fields
from itertools import product
from pathlib import Path
from coop_navigation_sds.Configuration.component_catalog import apply_speech_engine_profiles
from coop_navigation_sds.Configuration.run_identity import compact_code
from coop_navigation_sds.Configuration.model_matrix import model_size_treatment
from coop_navigation_sds.Configuration.experiment import (
    ExperimentSpecification,
    configuration_fingerprint,
    ensure_experiment_specification,
    freeze_value,
    thaw_value,
)
from coop_navigation_sds.Configuration.pipeline import experiment_pipeline_contract

from coop_navigation_sds.NaturalLanguageGeneration.caller.responder import (
    LLMAgentAResponder,
    TemplateAgentAResponder,
    agent_a_uses_model,
    normalize_agent_a_type,
)
from coop_navigation_sds.NaturalLanguageGeneration.caller.config import DEFAULT_PERSONA, LLM_AGENT_A
from coop_navigation_sds.Configuration.speech import (
    AGENT_B_PLUGIN,
    DEFAULT_SPEECH_PATTERN,
    SPEECH_ASR_ENGINE,
    SPEECH_AUDIO_DIR,
    SPEECH_PERFORMANCE_PROFILES,
    SPEECH_PLAYBACK_ENABLED,
    SPEECH_REALTIME_ENABLED,
    SPEECH_TTS_ENGINE,
)
from coop_navigation_sds.NaturalLanguageGeneration.assistant.plugin_registry import AgentBPluginConfig, create_agent_b_plugin
from coop_navigation_sds.NaturalLanguageGeneration.models import model_adapter_runtime_metadata
from coop_navigation_sds.DialogManagement.speech_pipeline import (
    SpeechPipelineConfig,
    SpeechTransport,
    platform_default_asr_engine,
    platform_default_tts_engine,
)
from coop_navigation_sds.TextToSpeech.personas import DEFAULT_AGENT_A_AUDIO_PERSONA, DEFAULT_AGENT_B_AUDIO_PERSONA
from coop_navigation_sds.DialogManagement.manager import DEFAULT_MAX_TURN_ELAPSED_SEC, DialogManager
from coop_navigation_sds.DialogManagement.result import DialogResult, NullEventQueue
from coop_navigation_sds.Configuration.runtime import (
    AGENT_A_TRANSFER_TOLERANCE,
    CONSTRAINT_MISS_LIMIT,
    DEFAULT_MODEL_PARAM_KEY,
    INVALID_ROUTE_LIMIT,
    SESSION_LOG_DIR,
)
from coop_navigation_sds.EvaluationMetrics.metrics import (
    MetricComputer,
    apply_cross_run_metrics,
    apply_paired_run_metrics,
)
from coop_navigation_sds.ResultsAndArtifacts.artifacts import write_metrics_csv, write_metrics_file
from coop_navigation_sds.ResultsAndArtifacts.logging import LOG_PROFILE_OFF, MonitoringEventQueue, SessionLogger
from coop_navigation_sds.TransportNetwork.test_cases import TEST_CASES, get_test_case
from coop_navigation_sds.Configuration.travel import GENERATION_MAX_TIME_SEC
from coop_navigation_sds.TransportNetwork.constraints import OBJECTIVE_SHORTEST_WITH_CONSTRAINTS


class DropQueue:
    """Queue adapter that discards human-readable batch events."""

    def put(self, _event):
        return None


@dataclass(frozen=True)
class ExperimentCondition:
    """Data model for one batch experiment condition.
    """
    condition_id: str
    test_case_key: str
    persona_key: str
    scenario_key: str
    speech_pattern_key: str
    model_param_key: str
    objective_mode: str = OBJECTIVE_SHORTEST_WITH_CONSTRAINTS
    iteration: int = 0
    agent_a_audio_persona: str = DEFAULT_AGENT_A_AUDIO_PERSONA
    agent_b_audio_persona: str = DEFAULT_AGENT_B_AUDIO_PERSONA
    pair_id: str = ""
    run_type: str = "audio_variant"
    tts_engine: str = ""
    asr_engine: str = ""
    agent_b_model: str = ""
    parameter_values: Mapping = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "parameter_values", freeze_value(self.parameter_values))
        object.__setattr__(self, "objective_mode", OBJECTIVE_SHORTEST_WITH_CONSTRAINTS)


RUNTIME_PARAMETER_KEYS = {
    "network_seed", "num_turns", "transfer_tolerance", "invalid_route_limit",
    "constraint_miss_limit", "dialogue_stagnation_limit", "max_turn_elapsed_sec",
    "calculation_max_time_sec", "profile_key", "agent_b_llm_size",
    "matrix_family", "experiment_platform", "experiment_seed",
    "repetition", "slurm_condition_index", "slurm_grid_name", "run_mode",
    "agent_b_model_role", "agent_b_model_slot",
}


def resolved_condition_test_case(condition, scenario_overrides=None, *, default_num_turns=20):
    """Return the exact test-case shape used for route-stage preflight."""
    parameters = dict(condition.parameter_values)
    base_case = get_test_case(condition.test_case_key)
    test_case = base_case.with_persona(condition.persona_key)
    num_turns = max(1, int(parameters.get("num_turns", default_num_turns)))
    with_overrides = getattr(test_case, "with_scenario_overrides", None)
    if not callable(with_overrides):
        return test_case
    speech_parameter_keys = set(SpeechPipelineConfig.__dataclass_fields__)
    overrides = {
        **dict(scenario_overrides or {}),
        **{
            key: value for key, value in parameters.items()
            if key not in RUNTIME_PARAMETER_KEYS
            and not key.endswith("_profile_key")
            and key not in speech_parameter_keys
        },
    }
    overrides.pop("network_seed", None)
    return with_overrides(maximum_dialog_turns=num_turns, **overrides)


def condition_stage_viability(condition, scenario_overrides=None, *, default_transfer_tolerance=1, default_num_turns=20):
    """Return route-stage viability for a generated condition without running models."""
    parameters = dict(condition.parameter_values)
    network_seed = parameters.get("network_seed", (scenario_overrides or {}).get("network_seed"))
    if network_seed is not None:
        from coop_navigation_sds.TransportNetwork.network import rebuild_network
        rebuild_network(network_seed, force=True)
    test_case = resolved_condition_test_case(
        condition,
        scenario_overrides=scenario_overrides,
        default_num_turns=default_num_turns,
    )
    from coop_navigation_sds.TransportNetwork.network import STATIONS
    if {
        test_case.scenario.get("start_station"),
        test_case.scenario.get("destination_station"),
    }.issubset(STATIONS):
        from coop_navigation_sds.TransportNetwork.constraints import stage_viability_report
        max_constraints = max(
            0,
            int(test_case.scenario.get("maximum_progressive_constraints", 2)),
        )
        transfer_tolerance = max(
            0,
            int(parameters.get("transfer_tolerance", default_transfer_tolerance)),
        )
        return stage_viability_report(
            test_case.scenario,
            test_case.persona,
            transfer_tolerance=transfer_tolerance,
            max_constraints=max_constraints,
        )
    return {"all_stage_requirements_satisfied": True, "stages": []}


def valid_stage_condition(condition, scenario_overrides=None, *, default_transfer_tolerance=1, default_num_turns=20):
    """Return whether a condition can support the staged route-dialogue design."""
    return bool(condition_stage_viability(
        condition,
        scenario_overrides=scenario_overrides,
        default_transfer_tolerance=default_transfer_tolerance,
        default_num_turns=default_num_turns,
    ).get("all_stage_requirements_satisfied"))


def condition_configuration_provenance(specification, condition):
    """Return distinct, reproducible identities for a batch and one condition."""
    condition_values = {
        item.name: thaw_value(getattr(condition, item.name))
        for item in fields(condition)
    }
    return {
        "base_fingerprint_sha256": specification.fingerprint,
        "fingerprint_sha256": configuration_fingerprint({
            "base_configuration_fingerprint": specification.fingerprint,
            "condition": condition_values,
        }),
    }


class ExperimentRunner:
    """Batch controller for running one condition or a full condition grid.
    """
    def __init__(
        self,
        model_adapter,
        num_turns,
        agent_b_plugin_key=AGENT_B_PLUGIN,
        tts_engine=SPEECH_TTS_ENGINE,
        asr_engine=SPEECH_ASR_ENGINE,
        speech_audio_dir=SPEECH_AUDIO_DIR,
        speech_playback_enabled=SPEECH_PLAYBACK_ENABLED,
        speech_realtime_enabled=SPEECH_REALTIME_ENABLED,
        speech_synthesis_config=None,
        transfer_tolerance=AGENT_A_TRANSFER_TOLERANCE,
        invalid_route_limit=INVALID_ROUTE_LIMIT,
        constraint_miss_limit=CONSTRAINT_MISS_LIMIT,
        stagnation_limit=2,
        max_turn_elapsed_sec=DEFAULT_MAX_TURN_ELAPSED_SEC,
        calculation_max_time_sec=GENERATION_MAX_TIME_SEC,
        llm_agent_a=LLM_AGENT_A,
        agent_a_type=None,
        log_profile=LOG_PROFILE_OFF,
        log_dir=SESSION_LOG_DIR,
        scenario_overrides=None,
        model_adapter_factory=None,
        agent_a_model_adapter=None,
        experiment_specification=None,
    ):
        """  init   method for this module's MVC responsibility.
        
        Args:
            model_adapter: Input value used by `__init__`; see the function signature and caller context for the expected type.
            num_turns: Input value used by `__init__`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        self.model_adapter = model_adapter
        self.model_adapter_factory = model_adapter_factory
        self.agent_a_model_adapter = agent_a_model_adapter
        self._model_adapter_cache = {}
        self._active_network_seed = None
        self._viability_cache = {}
        self.num_turns = num_turns
        self.agent_b_plugin_key = agent_b_plugin_key
        self.tts_engine = tts_engine or platform_default_tts_engine()
        self.asr_engine = asr_engine or platform_default_asr_engine()
        self.speech_audio_dir = speech_audio_dir
        self.speech_playback_enabled = speech_playback_enabled
        self.speech_realtime_enabled = speech_realtime_enabled
        self.speech_synthesis_config = {
            "agent_a_custom_audio": False,
            "agent_b_custom_audio": False,
            **dict(speech_synthesis_config or {}),
        }
        self.transfer_tolerance = transfer_tolerance
        self.invalid_route_limit = invalid_route_limit
        self.constraint_miss_limit = constraint_miss_limit
        self.stagnation_limit = max(1, int(stagnation_limit or 2))
        self.max_turn_elapsed_sec = max_turn_elapsed_sec
        self.calculation_max_time_sec = calculation_max_time_sec
        self.agent_a_type = normalize_agent_a_type(agent_a_type, llm_agent_a)
        self.llm_agent_a = agent_a_uses_model(self.agent_a_type)
        self.log_profile = (log_profile or LOG_PROFILE_OFF).lower()
        self.log_dir = log_dir
        self.scenario_overrides = dict(scenario_overrides or {})
        self.metric_computer = MetricComputer()
        self.experiment_specification = (
            ensure_experiment_specification(
                experiment_specification,
                source="batch_run",
            )
            if experiment_specification is not None
            else None
        )

    def run_condition(self, condition: ExperimentCondition, *, compute_metrics=True, capture_failure=False):
        """Run condition method for this module's MVC responsibility.
        
        Args:
            condition: Input value used by `run_condition`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        parameters = dict(condition.parameter_values)
        network_seed = parameters.get("network_seed", self.scenario_overrides.get("network_seed"))
        if network_seed is not None:
            from coop_navigation_sds.TransportNetwork.network import rebuild_network
            # Every condition starts from a fresh deterministic network state.
            # This prevents mutable route/cache state from leaking across cases.
            rebuild_network(network_seed, force=True)
            if network_seed != self._active_network_seed:
                self._viability_cache.clear()
            self._active_network_seed = network_seed
        num_turns = max(1, int(parameters.get("num_turns", self.num_turns)))
        transfer_tolerance = max(0, int(parameters.get("transfer_tolerance", self.transfer_tolerance)))
        invalid_route_limit = max(1, int(parameters.get("invalid_route_limit", self.invalid_route_limit)))
        constraint_miss_limit = max(0, int(parameters.get("constraint_miss_limit", self.constraint_miss_limit)))
        stagnation_limit = max(1, int(parameters.get("dialogue_stagnation_limit", self.stagnation_limit)))
        max_turn_elapsed_sec = max(0.1, float(parameters.get("max_turn_elapsed_sec", self.max_turn_elapsed_sec)))
        calculation_max_time_sec = max(
            0.1,
            float(parameters.get("calculation_max_time_sec", self.calculation_max_time_sec)),
        )
        test_case = resolved_condition_test_case(
            condition,
            scenario_overrides=self.scenario_overrides,
            default_num_turns=self.num_turns,
        )
        from coop_navigation_sds.TransportNetwork.network import STATIONS
        if {
            test_case.scenario.get("start_station"),
            test_case.scenario.get("destination_station"),
        }.issubset(STATIONS):
            viability_key = (
                network_seed,
                test_case.key,
                transfer_tolerance,
                repr(sorted(test_case.scenario.items())),
            )
            viability = self._viability_cache.get(viability_key)
            if viability is None:
                viability = condition_stage_viability(
                    condition,
                    scenario_overrides=self.scenario_overrides,
                    default_transfer_tolerance=transfer_tolerance,
                    default_num_turns=self.num_turns,
                )
                self._viability_cache[viability_key] = viability
            if not viability["all_stage_requirements_satisfied"]:
                failed = [stage["stage"] for stage in viability["stages"] if not stage["requirement_satisfied"]]
                raise ValueError(
                    f"Condition '{condition.condition_id}' has insufficient viable route alternatives "
                    f"for conversation stage(s) {failed}."
                )
        agent_b_model_adapter = self._model_adapter_for(condition)
        agent_a_model_adapter = self.agent_a_model_adapter or agent_b_model_adapter
        self._configure_model_adapter_runtime(agent_b_model_adapter, calculation_max_time_sec)
        if agent_a_model_adapter is not agent_b_model_adapter:
            self._configure_model_adapter_runtime(agent_a_model_adapter, calculation_max_time_sec)
        speech_config = {
            "pattern_key": condition.speech_pattern_key,
            "tts_engine": "file" if condition.run_type == "text_only" else (condition.tts_engine or self.tts_engine),
            "asr_engine": "file" if condition.run_type == "text_only" else (condition.asr_engine or self.asr_engine),
            "audio_dir": str(
                Path(self.speech_audio_dir)
                / ".turn_audio"
                / hashlib.sha256(condition.condition_id.encode("utf-8")).hexdigest()[:12]
            ),
            "playback_enabled": self.speech_playback_enabled,
            "realtime_enabled": self.speech_realtime_enabled,
            "agent_a_audio_persona": condition.agent_a_audio_persona,
            "agent_b_audio_persona": condition.agent_b_audio_persona,
            **self.speech_synthesis_config,
        }
        selected_tts = speech_config["tts_engine"]
        selected_asr = speech_config["asr_engine"]
        speech_config = apply_speech_engine_profiles(
            speech_config,
            replace=tuple(
                stage for stage, changed in (
                    ("tts", condition.run_type == "text_only" or selected_tts != self.tts_engine),
                    ("asr", condition.run_type == "text_only" or selected_asr != self.asr_engine),
                ) if changed
            ),
        )
        speech_config.update({key: value for key, value in parameters.items() if key in speech_parameter_keys})
        speech_transport = SpeechTransport(
            config=SpeechPipelineConfig(**speech_config)
        )
        agent_b_plugin = create_agent_b_plugin(self.agent_b_plugin_key, agent_b_model_adapter)
        manager = DialogManager(
            test_case,
            agent_b_plugin,
            num_turns,
            speech_transport=speech_transport,
            agent_a_responder=self._agent_a_responder_for(agent_a_model_adapter),
            transfer_tolerance=transfer_tolerance,
            invalid_route_limit=invalid_route_limit,
            constraint_miss_limit=constraint_miss_limit,
            stagnation_limit=stagnation_limit,
            max_turn_elapsed_sec=max_turn_elapsed_sec,
            agent_a_objective_mode=condition.objective_mode,
        )

        started_perf = time.perf_counter()
        event_queue = self._event_queue_for(condition)
        failure = None
        try:
            segment = event_queue.segment if hasattr(event_queue, "segment") else lambda *_args, **_kwargs: nullcontext()
            with segment(
                "batch.condition",
                condition_id=condition.condition_id,
                test_case=condition.test_case_key,
                persona=condition.persona_key,
                speech_pattern=condition.speech_pattern_key,
                model_param=condition.model_param_key,
                calculation_max_time_sec=calculation_max_time_sec,
            ):
                result = manager.run(event_queue)
        except Exception as exc:
            if not capture_failure:
                raise
            failure = {
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "diagnostics": dict(getattr(exc, "diagnostics", {}) or {}),
                "traceback": traceback.format_exc(),
            }
            result = DialogResult(
                condition_id=condition.condition_id,
                test_case_key=condition.test_case_key,
                persona_key=condition.persona_key,
                scenario_key=condition.scenario_key,
                speech_pattern_key=condition.speech_pattern_key,
                model_name=condition.agent_b_model or str(getattr(agent_b_model_adapter, "name", "")),
                conversation=[],
                route=[],
                route_steps=[],
                route_valid=False,
                route_reaches_goal=False,
                route_correct=False,
                route_duration_min=None,
                runtime_sec=0.0,
                extra={
                    "execution_status": "failed",
                    "pipeline_failure": failure,
                    "conversation_outcome": "unsatisfied",
                    "messages": 0,
                },
            )
        finally:
            speech_transport.close()
            if hasattr(event_queue, "close"):
                event_queue.close()
        condition_runtime_sec = time.perf_counter() - started_perf
        result.runtime_sec = condition_runtime_sec
        result.extra.setdefault("execution_status", "completed")
        result.condition_id = condition.condition_id
        result.speech_pattern_key = condition.speech_pattern_key
        result.extra["model_param_key"] = condition.model_param_key
        result.extra["objective_mode"] = condition.objective_mode
        result.extra["iteration"] = condition.iteration
        result.extra["agent_a_audio_persona"] = condition.agent_a_audio_persona
        result.extra["agent_b_audio_persona"] = condition.agent_b_audio_persona
        result.extra["parameter_values"] = dict(condition.parameter_values)
        result.extra["pair_id"] = condition.pair_id
        result.extra["run_type"] = condition.run_type
        result.extra["tts_engine"] = speech_config["tts_engine"]
        result.extra["asr_engine"] = speech_config["asr_engine"]
        result.extra["configured_tts_engine"] = condition.tts_engine or self.tts_engine
        result.extra["configured_asr_engine"] = condition.asr_engine or self.asr_engine
        result.extra["agent_b_model"] = condition.agent_b_model
        result.extra["agent_a_type"] = self.agent_a_type
        result.extra["agent_b_plugin"] = self.agent_b_plugin_key
        result.extra["resolved_audio_personas"] = {
            "agent_a": speech_transport.config.prosody_for("Agent A"),
            "agent_b": speech_transport.config.prosody_for("Agent B"),
        }
        result.extra["condition_runtime_sec"] = round(condition_runtime_sec, 6)
        result.extra["resolved_scenario"] = dict(test_case.scenario)
        if self.experiment_specification is not None:
            result.extra["resolved_run_config"] = self.experiment_specification.to_dict()
            result.extra["configuration_provenance"] = self.experiment_specification.provenance()
            result.extra["condition_provenance"] = condition_configuration_provenance(
                self.experiment_specification,
                condition,
            )
            result.extra["pipeline_contract"] = experiment_pipeline_contract(
                self.experiment_specification
            )
        model_roles = []
        if AgentBPluginConfig(self.agent_b_plugin_key).needs_model:
            model_roles.append("agent_b")
        if agent_a_uses_model(self.agent_a_type) and self.agent_a_model_adapter is None:
            model_roles.append("agent_a")
        result.extra["model_backend"] = model_adapter_runtime_metadata(
            agent_b_model_adapter,
            roles=model_roles,
        )
        if self.agent_a_model_adapter is not None and agent_a_uses_model(self.agent_a_type):
            result.extra["agent_a_model_backend"] = model_adapter_runtime_metadata(
                agent_a_model_adapter,
                roles=["agent_a"],
            )
        model_parameters = getattr(agent_b_model_adapter, "model_parameters", None)
        if model_parameters is not None:
            result.extra["model_parameters"] = asdict(model_parameters)
        metric = self.metric_computer.compute(result, test_case.scenario) if compute_metrics else None
        if metric is not None and hasattr(metric, "pair_id"):
            metric.pair_id = condition.pair_id
            metric.run_type = condition.run_type
        return result, metric

    def _event_queue_for(self, condition: ExperimentCondition):
        """Return a no-op queue by default, or a structured logger for batch audits."""
        if self.log_profile == LOG_PROFILE_OFF:
            return NullEventQueue()
        logger = SessionLogger(
            f"batch-{condition.condition_id}",
            self.log_dir,
            profile=self.log_profile,
        )
        return MonitoringEventQueue(DropQueue(), logger)

    def _configure_model_adapter_runtime(self, model_adapter, calculation_max_time_sec=None):
        if model_adapter is None:
            return
        budget = max(1.0, float(calculation_max_time_sec or self.calculation_max_time_sec or GENERATION_MAX_TIME_SEC))
        if hasattr(model_adapter, "max_time_sec"):
            model_adapter.max_time_sec = budget

    def _agent_a_responder_for(self, model_adapter):
        if agent_a_uses_model(self.agent_a_type) and model_adapter is not None:
            return LLMAgentAResponder(model_adapter)
        return TemplateAgentAResponder()

    def _model_adapter_for(self, condition: ExperimentCondition):
        """ model adapter for method for this module's MVC responsibility.
        
        Args:
            condition: Input value used by `_model_adapter_for`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        adapter = self.model_adapter
        if condition.agent_b_model and self.model_adapter_factory is not None:
            base_name = str(getattr(self.model_adapter, "name", "") or "")
            if base_name == condition.agent_b_model:
                adapter = self.model_adapter
            else:
                adapter = self._model_adapter_cache.get(condition.agent_b_model)
                if adapter is None:
                    adapter = self.model_adapter_factory(condition.agent_b_model)
                    self._model_adapter_cache[condition.agent_b_model] = adapter
        if hasattr(adapter, "with_model_params"):
            return adapter.with_model_params(condition.model_param_key)
        return adapter

    def run_grid(self, conditions, collect_results=True):
        """Run grid method for this module's MVC responsibility.
        
        Args:
            conditions: Input value used by `run_grid`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        results = [] if collect_results else None
        metrics = []
        for condition in conditions:
            result, metric = self.run_condition(condition)
            if collect_results:
                results.append(result)
            metrics.append(metric)
            metric.pair_id = condition.pair_id
            metric.run_type = condition.run_type
        apply_cross_run_metrics(metrics)
        apply_paired_run_metrics(metrics)
        return (results if results is not None else []), metrics


def pairwise_factor_rows(factors):
    """Return deterministic strength-two rows covering every pair of factor levels."""
    normalized = []
    for name, values in factors:
        levels = list(values)
        if levels:
            normalized.append((str(name), levels))
    if not normalized:
        return [{}]
    if len(normalized) == 1:
        name, values = normalized[0]
        return [{name: value} for value in values]

    # Largest factors first minimizes vertical-growth rows without changing coverage.
    ordered = sorted(
        enumerate(normalized),
        key=lambda item: (-len(item[1][1]), item[0]),
    )
    ordered_factors = [factor for _index, factor in ordered]
    rows = [
        [left, right]
        for left in range(len(ordered_factors[0][1]))
        for right in range(len(ordered_factors[1][1]))
    ]

    for new_index in range(2, len(ordered_factors)):
        new_level_count = len(ordered_factors[new_index][1])
        uncovered = {
            (prior_index, prior_level, new_level)
            for prior_index in range(new_index)
            for prior_level in range(len(ordered_factors[prior_index][1]))
            for new_level in range(new_level_count)
        }

        for row in rows:
            selected = max(
                range(new_level_count),
                key=lambda level: (
                    sum(
                        (prior_index, row[prior_index], level) in uncovered
                        for prior_index in range(new_index)
                    ),
                    -level,
                ),
            )
            row.append(selected)
            for prior_index in range(new_index):
                uncovered.discard((prior_index, row[prior_index], selected))

        while uncovered:
            anchor_index, anchor_level, new_level = min(uncovered)
            row = [None] * new_index + [new_level]
            row[anchor_index] = anchor_level
            for prior_index in range(new_index):
                if row[prior_index] is None:
                    row[prior_index] = max(
                        range(len(ordered_factors[prior_index][1])),
                        key=lambda level: (
                            int((prior_index, level, new_level) in uncovered),
                            -level,
                        ),
                    )
                uncovered.discard((prior_index, row[prior_index], new_level))
            rows.append(row)

    return [
        {
            name: values[row[index]]
            for index, (name, values) in enumerate(ordered_factors)
        }
        for row in rows
    ]


def condition_coverage_report(conditions):
    """Summarize value and pair coverage from expanded audio conditions."""
    source = [condition for condition in conditions if condition.run_type == "audio_variant"]
    rows = []
    for condition in source:
        parameters = dict(condition.parameter_values)
        task_profile = parameters.get("task_profile_key")
        row = {
            "task_profile" if task_profile else "test_case": task_profile or condition.test_case_key,
            "speech_pattern": condition.speech_pattern_key,
            "model_parameters": condition.model_param_key,
            "objective": condition.objective_mode,
            "agent_a_audio_persona": condition.agent_a_audio_persona,
            "agent_b_audio_persona": condition.agent_b_audio_persona,
            "text_to_speech": condition.tts_engine,
            "automatic_speech_recognition": condition.asr_engine,
            "agent_b_model": condition.agent_b_model,
        }
        if not task_profile:
            row["persona"] = condition.persona_key
        for key, value in parameters.items():
            if key.endswith("_profile_key") and key not in {"task_profile_key"}:
                row[key.removesuffix("_key")] = value
        rows.append(row)

    factor_names = sorted({name for row in rows for name in row})
    levels = {
        name: sorted({str(row[name]) for row in rows if name in row})
        for name in factor_names
    }
    expected_pairs = set()
    observed_pairs = set()
    for left_index, left in enumerate(factor_names):
        for right in factor_names[left_index + 1:]:
            expected_pairs.update(
                (left, left_value, right, right_value)
                for left_value in levels[left]
                for right_value in levels[right]
            )
            observed_pairs.update(
                (left, str(row[left]), right, str(row[right]))
                for row in rows
                if left in row and right in row
            )
    missing = sorted(expected_pairs - observed_pairs)
    performance = speech_performance_coverage_report(source)
    return {
        "audio_condition_count": len(source),
        "factor_count": len(factor_names),
        "levels": levels,
        "expected_pair_count": len(expected_pairs),
        "covered_pair_count": len(expected_pairs) - len(missing),
        "pair_coverage_ratio": (
            round((len(expected_pairs) - len(missing)) / len(expected_pairs), 6)
            if expected_pairs else 1.0
        ),
        "missing_pairs": [
            {"left_factor": left, "left_value": left_value,
             "right_factor": right, "right_value": right_value}
            for left, left_value, right, right_value in missing
        ],
        "speech_performance_coverage": performance,
    }


def speech_performance_coverage_report(conditions):
    """Verify floor-to-ceiling bands within every comparable treatment cell."""
    required = tuple(SPEECH_PERFORMANCE_PROFILES)
    groups = {}
    for condition in conditions:
        if condition.run_type != "audio_variant":
            continue
        parameters = dict(condition.parameter_values)
        band = parameters.get("speech_performance_band")
        if not band:
            continue
        key = (
            condition.agent_b_model,
            condition.tts_engine,
            condition.asr_engine,
            condition.test_case_key,
            condition.persona_key,
        )
        groups.setdefault(key, set()).add(str(band))
    rows = []
    for key, observed in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0])):
        missing = [band for band in required if band not in observed]
        rows.append({
            "agent_b_model": key[0],
            "tts_engine": key[1],
            "asr_engine": key[2],
            "test_case_key": key[3],
            "persona_key": key[4],
            "observed_bands": [band for band in required if band in observed],
            "missing_bands": missing,
            "complete": not missing,
        })
    return {
        "applicable": bool(groups),
        "required_bands": list(required),
        "group_count": len(rows),
        "complete_group_count": sum(bool(row["complete"]) for row in rows),
        "complete": bool(rows) and all(row["complete"] for row in rows),
        "groups": rows,
    }


def build_condition_grid(
    test_case_keys=None,
    persona_keys=None,
    speech_pattern_keys=None,
    model_param_keys=None,
    objective_modes=None,
    agent_a_audio_persona_keys=None,
    agent_b_audio_persona_keys=None,
    tts_engine_keys=None,
    asr_engine_keys=None,
    agent_b_model_keys=None,
    iterations=1,
    parameter_grid=None,
    parameter_profiles=None,
    linked_profiles=None,
    coverage_strategy="full_factorial",
    pair_audio_with_text=False,
):
    """Build condition grid function for this module's MVC responsibility.
    
    Args:
        test_case_keys: Input value used by `build_condition_grid`; see the function signature and caller context for the expected type.
        persona_keys: Input value used by `build_condition_grid`; see the function signature and caller context for the expected type.
        speech_pattern_keys: Input value used by `build_condition_grid`; see the function signature and caller context for the expected type.
        model_param_keys: Input value used by `build_condition_grid`; see the function signature and caller context for the expected type.
        iterations: Input value used by `build_condition_grid`; see the function signature and caller context for the expected type.
    
    Returns:
        The computed value or side effect documented by the implementation.
    """
    test_case_keys = test_case_keys or list(TEST_CASES)
    persona_keys = persona_keys or [DEFAULT_PERSONA]
    speech_pattern_keys = speech_pattern_keys or [DEFAULT_SPEECH_PATTERN]
    model_param_keys = model_param_keys or [DEFAULT_MODEL_PARAM_KEY]
    # Kept in the signature for job/CLI compatibility; objective is controlled.
    objective_modes = [OBJECTIVE_SHORTEST_WITH_CONSTRAINTS]
    agent_a_audio_persona_keys = agent_a_audio_persona_keys or [DEFAULT_AGENT_A_AUDIO_PERSONA]
    agent_b_audio_persona_keys = agent_b_audio_persona_keys or [DEFAULT_AGENT_B_AUDIO_PERSONA]
    tts_engine_keys = tts_engine_keys or [""]
    asr_engine_keys = asr_engine_keys or [""]
    agent_b_model_keys = agent_b_model_keys or [""]
    parameter_grid = dict(parameter_grid or {})
    parameter_profiles = list(parameter_profiles or [{"profile_key": "default"}])
    linked_profiles = dict(linked_profiles or {})
    coverage_strategy = str(coverage_strategy or "full_factorial").strip().lower()
    if coverage_strategy not in {"full_factorial", "pairwise"}:
        raise ValueError(f"Unsupported coverage strategy '{coverage_strategy}'.")

    factors = [
        ("test_case_key", test_case_keys),
        ("persona_key", persona_keys),
        ("speech_pattern_key", speech_pattern_keys),
        ("model_param_key", model_param_keys),
        ("objective_mode", objective_modes),
        ("agent_a_audio_persona", agent_a_audio_persona_keys),
        ("agent_b_audio_persona", agent_b_audio_persona_keys),
        ("tts_engine", tts_engine_keys),
        ("asr_engine", asr_engine_keys),
        ("agent_b_model", agent_b_model_keys),
        *(
            (f"linked::{group}", profiles)
            for group, profiles in sorted(linked_profiles.items())
        ),
        *((f"parameter::{name}", parameter_grid[name]) for name in sorted(parameter_grid)),
        ("parameter_profile", parameter_profiles),
    ]
    if coverage_strategy == "pairwise":
        factor_rows = pairwise_factor_rows(factors)
    else:
        factor_rows = [
            dict(zip((name for name, _values in factors), values))
            for values in product(*(values for _name, values in factors))
        ]

    test_case_cache = {}
    for row, iteration in product(factor_rows, range(iterations)):
        parameter_values = {
            **{
                name.removeprefix("parameter::"): value
                for name, value in row.items()
                if name.startswith("parameter::")
            },
            **dict(row["parameter_profile"]),
        }
        for name, value in row.items():
            if name.startswith("linked::"):
                parameter_values.update(dict(value))

        core_values = {
            key: parameter_values.pop(key, row[key])
            for key in (
                "test_case_key", "persona_key", "speech_pattern_key",
                "model_param_key", "objective_mode", "agent_a_audio_persona",
                "agent_b_audio_persona", "tts_engine", "asr_engine",
                "agent_b_model",
            )
        }
        test_case_key = core_values["test_case_key"]
        persona_key = core_values["persona_key"]
        speech_pattern_key = core_values["speech_pattern_key"]
        model_param_key = core_values["model_param_key"]
        objective_mode = core_values["objective_mode"]
        agent_a_audio_persona = core_values["agent_a_audio_persona"]
        agent_b_audio_persona = core_values["agent_b_audio_persona"]
        tts_engine = core_values["tts_engine"]
        asr_engine = core_values["asr_engine"]
        agent_b_model = core_values["agent_b_model"]
        registered_size_treatment = model_size_treatment(agent_b_model)
        if registered_size_treatment:
            configured_size_treatment = parameter_values.get("agent_b_llm_size")
            if configured_size_treatment not in {None, registered_size_treatment}:
                raise ValueError(
                    f"Agent B model '{agent_b_model}' belongs to size treatment "
                    f"'{registered_size_treatment}', not '{configured_size_treatment}'."
                )
            parameter_values["agent_b_llm_size"] = registered_size_treatment
        profile_key = str(parameter_values.get("profile_key", "default"))
        parameter_label = "__".join(f"{key}-{value}" for key, value in parameter_values.items())
        base_id = (
            f"{test_case_key}__{persona_key}__{agent_a_audio_persona}__{agent_b_audio_persona}__"
            f"{speech_pattern_key}__{model_param_key}__{objective_mode}__{tts_engine or 'default_tts'}__"
            f"{asr_engine or 'default_asr'}__{agent_b_model or 'default_model'}__{iteration}"
            f"{('__' + parameter_label) if parameter_label else ''}"
        )
        digest = hashlib.sha256(base_id.encode("utf-8")).hexdigest()[:10]
        pair_id = f"P-{digest}"
        condition_parts = [
            "C",
            compact_code(test_case_key),
            compact_code(persona_key),
            compact_code(speech_pattern_key),
            compact_code(tts_engine or "default_tts"),
            compact_code(asr_engine or "default_asr"),
            compact_code(agent_b_model or "default_model"),
            compact_code(profile_key),
        ]
        if parameter_values.get("agent_b_llm_size"):
            condition_parts.append(compact_code(parameter_values["agent_b_llm_size"]))
        condition_parts.extend((f"I{iteration}", digest))
        condition_code = "-".join(condition_parts)
        base_case = test_case_cache.get(test_case_key)
        if base_case is None:
            base_case = get_test_case(test_case_key)
            test_case_cache[test_case_key] = base_case
        common = dict(
            test_case_key=test_case_key, persona_key=persona_key,
            scenario_key=base_case.scenario_key, speech_pattern_key=speech_pattern_key,
            model_param_key=model_param_key,
            objective_mode=OBJECTIVE_SHORTEST_WITH_CONSTRAINTS,
            iteration=iteration, agent_a_audio_persona=agent_a_audio_persona,
            agent_b_audio_persona=agent_b_audio_persona, pair_id=pair_id,
            tts_engine=tts_engine, asr_engine=asr_engine, agent_b_model=agent_b_model,
            parameter_values=parameter_values,
        )
        if pair_audio_with_text:
            yield ExperimentCondition(condition_id=f"{condition_code}-T", run_type="text_only", **common)
        yield ExperimentCondition(condition_id=f"{condition_code}-A", run_type="audio_variant", **common)
