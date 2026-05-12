"""Batch experiment controller that executes condition grids and serializes metric records.
"""
import csv
from dataclasses import dataclass
from itertools import product

from minillama.dialog_manager import DialogManager
from minillama.agent_b_plugins import create_agent_b_plugin
from minillama.config import AGENT_B_PLUGIN
from minillama.metrics import MetricComputer
from minillama.dialog_result import NullEventQueue
from minillama.speech_io import LoopbackTextToSpeech, PatternedSpeechToText, SpeechTransport
from minillama.test_cases import TEST_CASES, get_test_case


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
    def __init__(self, model_adapter, num_turns):
        """  init   method for this module's MVC responsibility.
        
        Args:
            model_adapter: Input value used by `__init__`; see the function signature and caller context for the expected type.
            num_turns: Input value used by `__init__`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        self.model_adapter = model_adapter
        self.num_turns = num_turns
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
            LoopbackTextToSpeech(),
            PatternedSpeechToText(
                condition.speech_pattern_key,
                seed=condition.iteration,
            ),
        )
        agent_b_plugin = create_agent_b_plugin(AGENT_B_PLUGIN, model_adapter)
        manager = DialogManager(
            test_case,
            agent_b_plugin,
            self.num_turns,
            speech_transport=speech_transport,
        )

        result = manager.run(NullEventQueue())
        result.condition_id = condition.condition_id
        result.speech_pattern_key = condition.speech_pattern_key
        result.extra["model_param_key"] = condition.model_param_key
        result.extra["iteration"] = condition.iteration
        return result, self.metric_computer.compute(result, test_case.scenario)

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

    def run_grid(self, conditions):
        """Run grid method for this module's MVC responsibility.
        
        Args:
            conditions: Input value used by `run_grid`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        results = []
        metrics = []
        for condition in conditions:
            result, metric = self.run_condition(condition)
            results.append(result)
            metrics.append(metric)
        return results, metrics


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
    persona_keys = persona_keys or ["focused_commuter"]
    speech_pattern_keys = speech_pattern_keys or ["clean"]
    model_param_keys = model_param_keys or ["greedy"]

    conditions = []
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
        base_case = get_test_case(test_case_key)
        conditions.append(
            ExperimentCondition(
                condition_id=condition_id,
                test_case_key=test_case_key,
                persona_key=persona_key,
                scenario_key=base_case.scenario_key,
                speech_pattern_key=speech_pattern_key,
                model_param_key=model_param_key,
                iteration=iteration,
            )
        )

    return conditions


def write_metrics_csv(metrics, path):
    """Write metrics csv function for this module's MVC responsibility.
    
    Args:
        metrics: Input value used by `write_metrics_csv`; see the function signature and caller context for the expected type.
        path: Input value used by `write_metrics_csv`; see the function signature and caller context for the expected type.
    
    Returns:
        The computed value or side effect documented by the implementation.
    """
    if not metrics:
        return

    rows = [metric.as_dict() for metric in metrics]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
