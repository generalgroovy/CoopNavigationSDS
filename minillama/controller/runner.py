"""Batch experiment controller that executes condition grids and serializes metric records.
"""
import csv
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from itertools import product

from minillama.agent_a.config import DEFAULT_PERSONA
from minillama.agent_b.agent_b_plugins import create_agent_b_plugin
from minillama.agent_b.config import AGENT_B_PLUGIN
from minillama.agent_b.config import DEFAULT_SPEECH_PATTERN
from minillama.agent_b.config import SPEECH_AUDIO_DIR, SPEECH_ENGINE, SPEECH_INCOMING_ENABLED, SPEECH_OUTGOING_ENABLED, SPEECH_SCOPE
from minillama.agent_b.speech_io import SpeechPipelineConfig, SpeechTransport
from minillama.controller.dialog_manager import DialogManager
from minillama.controller.dialog_result import NullEventQueue
from minillama.controller.config import DEFAULT_MODEL_PARAM_KEY, SESSION_LOG_DIR
from minillama.evaluation.metrics import MetricComputer
from minillama.controller.session_logging import LOG_PROFILE_OFF, MonitoringEventQueue, SessionLogger
from minillama.test_cases.test_cases import TEST_CASES, get_test_case


class DropQueue:
    """Queue adapter for batch logging when no GUI consumes tuple events."""

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
    iteration: int


class ExperimentRunner:
    """Batch controller for running one condition or a full condition grid.
    """
    def __init__(
        self,
        model_adapter,
        num_turns,
        agent_b_plugin_key=AGENT_B_PLUGIN,
        speech_incoming_enabled=SPEECH_INCOMING_ENABLED,
        speech_outgoing_enabled=SPEECH_OUTGOING_ENABLED,
        speech_scope=SPEECH_SCOPE,
        speech_engine=SPEECH_ENGINE,
        speech_audio_dir=SPEECH_AUDIO_DIR,
        log_profile=LOG_PROFILE_OFF,
        log_dir=SESSION_LOG_DIR,
    ):
        """  init   method for this module's MVC responsibility.
        
        Args:
            model_adapter: Input value used by `__init__`; see the function signature and caller context for the expected type.
            num_turns: Input value used by `__init__`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        self.model_adapter = model_adapter
        self.num_turns = num_turns
        self.agent_b_plugin_key = agent_b_plugin_key
        self.speech_incoming_enabled = speech_incoming_enabled
        self.speech_outgoing_enabled = speech_outgoing_enabled
        self.speech_scope = speech_scope
        self.speech_engine = speech_engine
        self.speech_audio_dir = speech_audio_dir
        self.log_profile = (log_profile or LOG_PROFILE_OFF).lower()
        self.log_dir = log_dir
        self.metric_computer = MetricComputer()

    def run_condition(self, condition: ExperimentCondition):
        """Run condition method for this module's MVC responsibility.
        
        Args:
            condition: Input value used by `run_condition`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        base_case = get_test_case(condition.test_case_key)
        test_case = base_case.with_persona(condition.persona_key)
        model_adapter = self._model_adapter_for(condition)
        speech_transport = SpeechTransport(
            config=SpeechPipelineConfig(
                incoming_enabled=self.speech_incoming_enabled,
                outgoing_enabled=self.speech_outgoing_enabled,
                scope=self.speech_scope,
                pattern_key=condition.speech_pattern_key,
                engine=self.speech_engine,
                audio_dir=self.speech_audio_dir,
            )
        )
        agent_b_plugin = create_agent_b_plugin(self.agent_b_plugin_key, model_adapter)
        manager = DialogManager(
            test_case,
            agent_b_plugin,
            self.num_turns,
            speech_transport=speech_transport,
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
            ):
                result = manager.run(event_queue)
        finally:
            if hasattr(event_queue, "close"):
                event_queue.close()
        condition_runtime_sec = time.perf_counter() - started_perf
        result.condition_id = condition.condition_id
        result.speech_pattern_key = condition.speech_pattern_key
        result.extra["model_param_key"] = condition.model_param_key
        result.extra["iteration"] = condition.iteration
        result.extra["condition_runtime_sec"] = round(condition_runtime_sec, 6)
        model_parameters = getattr(model_adapter, "model_parameters", None)
        if model_parameters is not None:
            result.extra["model_parameters"] = asdict(model_parameters)
        return result, self.metric_computer.compute(result, test_case.scenario)

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

    def _model_adapter_for(self, condition: ExperimentCondition):
        """ model adapter for method for this module's MVC responsibility.
        
        Args:
            condition: Input value used by `_model_adapter_for`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        if hasattr(self.model_adapter, "with_model_params"):
            return self.model_adapter.with_model_params(condition.model_param_key)
        return self.model_adapter

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
        return (results if results is not None else []), metrics


def build_condition_grid(
    test_case_keys=None,
    persona_keys=None,
    speech_pattern_keys=None,
    model_param_keys=None,
    iterations=1,
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

    test_case_cache = {}
    for test_case_key, persona_key, speech_pattern_key, model_param_key, iteration in product(
        test_case_keys,
        persona_keys,
        speech_pattern_keys,
        model_param_keys,
        range(iterations),
    ):
        condition_id = (
            f"{test_case_key}__{persona_key}__{speech_pattern_key}__{model_param_key}__{iteration}"
        )
        base_case = test_case_cache.get(test_case_key)
        if base_case is None:
            base_case = get_test_case(test_case_key)
            test_case_cache[test_case_key] = base_case
        yield ExperimentCondition(
            condition_id=condition_id,
            test_case_key=test_case_key,
            persona_key=persona_key,
            scenario_key=base_case.scenario_key,
            speech_pattern_key=speech_pattern_key,
            model_param_key=model_param_key,
            iteration=iteration,
        )


def write_metrics_csv(metrics, path):
    """Write metrics csv function for this module's MVC responsibility.
    
    Args:
        metrics: Input value used by `write_metrics_csv`; see the function signature and caller context for the expected type.
        path: Input value used by `write_metrics_csv`; see the function signature and caller context for the expected type.
    
    Returns:
        The computed value or side effect documented by the implementation.
    """
    iterator = iter(metrics)
    try:
        first_metric = next(iterator)
    except StopIteration:
        return

    first_row = first_metric.as_dict()
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(first_row))
        writer.writeheader()
        writer.writerow(first_row)
        for metric in iterator:
            writer.writerow(metric.as_dict())
