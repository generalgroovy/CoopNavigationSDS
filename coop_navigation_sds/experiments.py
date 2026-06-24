"""Batch experiment controller that executes condition grids and serializes metric records.
"""
import time
import hashlib
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from itertools import product
from coop_navigation_sds.Configuration.component_catalog import apply_speech_engine_profiles
from coop_navigation_sds.Configuration.run_identity import compact_code

from coop_navigation_sds.NaturalLanguageGeneration.caller.responder import (
    LLMAgentAResponder,
    TemplateAgentAResponder,
    agent_a_uses_model,
    normalize_agent_a_type,
)
from coop_navigation_sds.NaturalLanguageGeneration.caller.config import DEFAULT_PERSONA, LLM_AGENT_A
from coop_navigation_sds.Configuration.speech import AGENT_B_PLUGIN
from coop_navigation_sds.Configuration.speech import DEFAULT_SPEECH_PATTERN
from coop_navigation_sds.Configuration.speech import SPEECH_ASR_ENGINE, SPEECH_AUDIO_DIR, SPEECH_PLAYBACK_ENABLED, SPEECH_REALTIME_ENABLED, SPEECH_TTS_ENGINE
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
from coop_navigation_sds.DialogManagement.result import NullEventQueue
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
from coop_navigation_sds.TransportNetwork.constraints import OBJECTIVE_MODES, OBJECTIVE_SHORTEST_WITH_CONSTRAINTS


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
    parameter_values: dict = field(default_factory=dict)


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
        self._model_adapter_cache = {}
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

    def run_condition(self, condition: ExperimentCondition, *, compute_metrics=True):
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
            rebuild_network(network_seed)
        base_case = get_test_case(condition.test_case_key)
        test_case = base_case.with_persona(condition.persona_key)
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
        runtime_parameter_keys = {
            "network_seed", "num_turns", "transfer_tolerance", "invalid_route_limit",
            "constraint_miss_limit", "dialogue_stagnation_limit", "max_turn_elapsed_sec",
            "calculation_max_time_sec", "profile_key",
        }
        speech_parameter_keys = set(SpeechPipelineConfig.__dataclass_fields__)
        with_overrides = getattr(test_case, "with_scenario_overrides", None)
        if callable(with_overrides):
            overrides = {
                **self.scenario_overrides,
                **{
                    key: value for key, value in parameters.items()
                    if key not in runtime_parameter_keys and key not in speech_parameter_keys
                },
            }
            overrides.pop("network_seed", None)
            test_case = with_overrides(
                maximum_dialog_turns=num_turns,
                **overrides,
            )
        from coop_navigation_sds.TransportNetwork.network import STATIONS
        if {
            test_case.scenario.get("start_station"),
            test_case.scenario.get("destination_station"),
        }.issubset(STATIONS):
            from coop_navigation_sds.TransportNetwork.constraints import stage_viability_report
            viability = stage_viability_report(test_case.scenario, test_case.persona)
            if not viability["all_stage_requirements_satisfied"]:
                failed = [stage["stage"] for stage in viability["stages"] if not stage["requirement_satisfied"]]
                raise ValueError(
                    f"Condition '{condition.condition_id}' has insufficient viable route alternatives "
                    f"for conversation stage(s) {failed}."
                )
        model_adapter = self._model_adapter_for(condition)
        self._configure_model_adapter_runtime(model_adapter, calculation_max_time_sec)
        speech_config = {
            "pattern_key": condition.speech_pattern_key,
            "tts_engine": "file" if condition.run_type == "text_only" else (condition.tts_engine or self.tts_engine),
            "asr_engine": "file" if condition.run_type == "text_only" else (condition.asr_engine or self.asr_engine),
            "audio_dir": self.speech_audio_dir,
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
        agent_b_plugin = create_agent_b_plugin(self.agent_b_plugin_key, model_adapter)
        manager = DialogManager(
            test_case,
            agent_b_plugin,
            num_turns,
            speech_transport=speech_transport,
            agent_a_responder=self._agent_a_responder_for(model_adapter),
            transfer_tolerance=transfer_tolerance,
            invalid_route_limit=invalid_route_limit,
            constraint_miss_limit=constraint_miss_limit,
            stagnation_limit=stagnation_limit,
            max_turn_elapsed_sec=max_turn_elapsed_sec,
            agent_a_objective_mode=condition.objective_mode,
        )

        started_perf = time.perf_counter()
        event_queue = self._event_queue_for(condition)
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
        finally:
            speech_transport.close()
            if hasattr(event_queue, "close"):
                event_queue.close()
        condition_runtime_sec = time.perf_counter() - started_perf
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
        result.extra["agent_b_model"] = condition.agent_b_model
        result.extra["agent_a_type"] = self.agent_a_type
        result.extra["agent_b_plugin"] = self.agent_b_plugin_key
        result.extra["resolved_audio_personas"] = {
            "agent_a": speech_transport.config.prosody_for("Agent A"),
            "agent_b": speech_transport.config.prosody_for("Agent B"),
        }
        result.extra["condition_runtime_sec"] = round(condition_runtime_sec, 6)
        result.extra["resolved_scenario"] = dict(test_case.scenario)
        model_roles = []
        if AgentBPluginConfig(self.agent_b_plugin_key).needs_model:
            model_roles.append("agent_b")
        if agent_a_uses_model(self.agent_a_type):
            model_roles.append("agent_a")
        result.extra["model_backend"] = model_adapter_runtime_metadata(
            model_adapter,
            roles=model_roles,
        )
        model_parameters = getattr(model_adapter, "model_parameters", None)
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
    objective_modes = objective_modes or [OBJECTIVE_SHORTEST_WITH_CONSTRAINTS]
    agent_a_audio_persona_keys = agent_a_audio_persona_keys or [DEFAULT_AGENT_A_AUDIO_PERSONA]
    agent_b_audio_persona_keys = agent_b_audio_persona_keys or [DEFAULT_AGENT_B_AUDIO_PERSONA]
    tts_engine_keys = tts_engine_keys or [""]
    asr_engine_keys = asr_engine_keys or [""]
    agent_b_model_keys = agent_b_model_keys or [""]
    parameter_grid = dict(parameter_grid or {})
    parameter_names = tuple(sorted(parameter_grid))
    parameter_combinations = list(product(*(parameter_grid[name] for name in parameter_names))) if parameter_names else [()]
    parameter_profiles = list(parameter_profiles or [{"profile_key": "default"}])

    test_case_cache = {}
    for (
        test_case_key,
        persona_key,
        speech_pattern_key,
        model_param_key,
        objective_mode,
        agent_a_audio_persona,
        agent_b_audio_persona,
        tts_engine,
        asr_engine,
        agent_b_model,
        iteration,
        parameter_combination,
        parameter_profile,
    ) in product(
        test_case_keys,
        persona_keys,
        speech_pattern_keys,
        model_param_keys,
        objective_modes,
        agent_a_audio_persona_keys,
        agent_b_audio_persona_keys,
        tts_engine_keys,
        asr_engine_keys,
        agent_b_model_keys,
        range(iterations),
        parameter_combinations,
        parameter_profiles,
    ):
        parameter_values = {
            **dict(zip(parameter_names, parameter_combination)),
            **dict(parameter_profile),
        }
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
        condition_code = "-".join((
            "C",
            compact_code(test_case_key),
            compact_code(persona_key),
            compact_code(speech_pattern_key),
            compact_code(tts_engine or "default_tts"),
            compact_code(asr_engine or "default_asr"),
            compact_code(agent_b_model or "default_model"),
            compact_code(profile_key),
            f"I{iteration}",
            digest,
        ))
        base_case = test_case_cache.get(test_case_key)
        if base_case is None:
            base_case = get_test_case(test_case_key)
            test_case_cache[test_case_key] = base_case
        common = dict(
            test_case_key=test_case_key, persona_key=persona_key,
            scenario_key=base_case.scenario_key, speech_pattern_key=speech_pattern_key,
            model_param_key=model_param_key,
            objective_mode=objective_mode if objective_mode in OBJECTIVE_MODES else OBJECTIVE_SHORTEST_WITH_CONSTRAINTS,
            iteration=iteration, agent_a_audio_persona=agent_a_audio_persona,
            agent_b_audio_persona=agent_b_audio_persona, pair_id=pair_id,
            tts_engine=tts_engine, asr_engine=asr_engine, agent_b_model=agent_b_model,
            parameter_values=parameter_values,
        )
        if pair_audio_with_text:
            yield ExperimentCondition(condition_id=f"{condition_code}-T", run_type="text_only", **common)
        yield ExperimentCondition(condition_id=f"{condition_code}-A", run_type="audio_variant", **common)
