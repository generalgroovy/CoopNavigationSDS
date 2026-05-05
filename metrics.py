from dataclasses import asdict, dataclass

from route_planner import optimal_time_route, route_duration_breakdown


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


@dataclass
class MetricRecord:
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
    duration_score: float
    quality_score: float

    def as_dict(self):
        return asdict(self)


class MetricComputer:
    def compute(self, result, scenario) -> MetricRecord:
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
        station_mentions = len(result.route)

        task_focus_score = task_terms / len(words) if words else 0.0

        if result.route_duration_min is None or reference_duration is None:
            duration_score = 0.0
        else:
            duration_score = reference_duration / max(result.route_duration_min, reference_duration)

        quality_score = 0.0
        if result.route_correct:
            quality_score = 0.70 + 0.30 * duration_score

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
            message_count=result.extra.get("messages", len(result.conversation)),
            word_count=len(words),
            station_mentions=station_mentions,
            task_focus_score=round(task_focus_score, 4),
            duration_score=round(duration_score, 4),
            quality_score=round(quality_score, 4),
        )
