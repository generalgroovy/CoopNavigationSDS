import csv
from dataclasses import dataclass
from itertools import product

from dialog_manager import DialogManager
from dialog_result import NullEventQueue
from metrics import MetricComputer
from pipeline import VerbalTransformationPipeline
from speech_io import LoopbackTextToSpeech, PatternedSpeechToText, SpeechTransport
from test_cases import TEST_CASES, get_test_case


@dataclass(frozen=True)
class ExperimentCondition:
    condition_id: str
    test_case_key: str
    persona_key: str
    scenario_key: str
    speech_pattern_key: str
    model_param_key: str
    iteration: int


class ExperimentRunner:
    def __init__(self, model_adapter, num_turns):
        self.model_adapter = model_adapter
        self.num_turns = num_turns
        self.metric_computer = MetricComputer()

    def run_condition(self, condition: ExperimentCondition):
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
        pipeline = VerbalTransformationPipeline(model_adapter)
        manager = DialogManager(
            test_case,
            pipeline,
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
        if hasattr(self.model_adapter, "with_model_params"):
            return self.model_adapter.with_model_params(condition.model_param_key)
        return self.model_adapter

    def run_grid(self, conditions):
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
    if not metrics:
        return

    rows = [metric.as_dict() for metric in metrics]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
