"""Experiment metric model. It converts dialog results into route quality, duration, focus, and runtime metrics.
"""
from dataclasses import asdict, dataclass
import re

from minillama.config import METRIC_QUALITY_BASE_WEIGHT, METRIC_QUALITY_DURATION_WEIGHT
from minillama.metro_data import STATION_POS
from minillama.route_planner import optimal_time_route, route_duration_breakdown


TASK_TERMS = {
    "station",
    "route",
    "line",
    "transfer",
    "change",
    "wait",
    "waiting",
    "duration",
    "time",
    "minutes",
    "arrive",
    "destination",
}


COMPARISON_TERMS = {
    "alternative",
    "better",
    "best",
    "compare",
    "faster",
    "improve",
    "option",
    "slower",
}

COOPERATION_TERMS = {
    "check",
    "confirm",
    "current",
    "candidate",
    "build",
    "revise",
    "together",
    "step",
}


@dataclass
class MetricRecord:
    """Data model for one row of experiment metrics.
    """
    condition_id: str
    test_case_key: str
    persona_key: str
    scenario_key: str
    speech_pattern_key: str
    model_name: str
    model_param_key: str
    success: bool
    route_valid: bool
    route_reaches_goal: bool
    route_duration_min: int | None
    reference_duration_min: int | None
    duration_excess_min: int | None
    travel_min: int
    wait_min: int
    transfer_min: int
    runtime_sec: float
    message_count: int
    word_count: int
    station_mentions: int
    task_focus_score: float
    comparison_terms: int
    cooperation_terms: int
    agent_a_question_count: int
    question_count: int
    avg_words_per_message: float
    candidate_route_count: int
    route_revision_count: int
    duration_score: float
    quality_score: float

    def as_dict(self):
        """As dict method for this module's MVC responsibility.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        return asdict(self)


class MetricComputer:
    """Metric model service that computes quality and dialog metrics from a completed dialog result.
    """
    def compute(self, result, scenario) -> MetricRecord:
        """Compute method for this module's MVC responsibility.
        
        Args:
            result: Input value used by `compute`; see the function signature and caller context for the expected type.
            scenario: Input value used by `compute`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        reference_arrival, _ = optimal_time_route(
            scenario["start_station"],
            scenario["destination_station"],
            scenario["start_time_min"],
            scenario["transfer_time_min"],
        )
        reference_duration = (
            reference_arrival - scenario["start_time_min"]
            if reference_arrival is not None
            else None
        )

        breakdown = route_duration_breakdown(result.route_steps) if result.route_steps else {
            "travel": 0,
            "wait": 0,
            "transfer": 0,
        }

        duration_excess = None
        if result.route_duration_min is not None and reference_duration is not None:
            duration_excess = result.route_duration_min - reference_duration

        conversation_text = " ".join(text for _, text in result.conversation)
        words = [word.strip(".,!?;:()").lower() for word in conversation_text.split()]
        words = [word for word in words if word]
        task_terms = sum(1 for word in words if word in TASK_TERMS)
        comparison_terms = sum(1 for word in words if word in COMPARISON_TERMS)
        cooperation_terms = sum(1 for word in words if word in COOPERATION_TERMS)
        station_mentions = 0
        for station in STATION_POS:
            station_mentions += len(
                re.findall(rf"\b{re.escape(station)}\b", conversation_text, flags=re.IGNORECASE)
            )
        question_count = sum(text.count("?") for _, text in result.conversation)
        agent_a_question_count = sum(
            text.count("?") for speaker, text in result.conversation if speaker == "Agent A"
        )
        message_count = result.extra.get("messages", len(result.conversation))
        avg_words_per_message = len(words) / message_count if message_count else 0.0

        task_focus_score = task_terms / len(words) if words else 0.0

        if result.route_duration_min is None or reference_duration is None:
            duration_score = 0.0
        else:
            duration_score = reference_duration / max(result.route_duration_min, reference_duration)

        quality_score = 0.0
        if result.route_correct:
            quality_score = METRIC_QUALITY_BASE_WEIGHT + METRIC_QUALITY_DURATION_WEIGHT * duration_score

        return MetricRecord(
            condition_id=result.condition_id,
            test_case_key=result.test_case_key,
            persona_key=result.persona_key,
            scenario_key=result.scenario_key,
            speech_pattern_key=result.speech_pattern_key,
            model_name=result.model_name,
            model_param_key=result.extra.get("model_param_key", "default"),
            success=result.route_correct,
            route_valid=result.route_valid,
            route_reaches_goal=result.route_reaches_goal,
            route_duration_min=result.route_duration_min,
            reference_duration_min=reference_duration,
            duration_excess_min=duration_excess,
            travel_min=breakdown["travel"],
            wait_min=breakdown["wait"],
            transfer_min=breakdown["transfer"],
            runtime_sec=result.runtime_sec,
            message_count=message_count,
            word_count=len(words),
            station_mentions=station_mentions,
            task_focus_score=round(task_focus_score, 4),
            comparison_terms=comparison_terms,
            cooperation_terms=cooperation_terms,
            agent_a_question_count=agent_a_question_count,
            question_count=question_count,
            avg_words_per_message=round(avg_words_per_message, 2),
            candidate_route_count=result.extra.get("candidate_routes", 0),
            route_revision_count=result.extra.get("route_revisions", 0),
            duration_score=round(duration_score, 4),
            quality_score=round(quality_score, 4),
        )
