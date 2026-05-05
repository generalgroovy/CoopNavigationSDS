from dataclasses import dataclass, field


@dataclass
class DialogResult:
    condition_id: str
    test_case_key: str
    persona_key: str
    scenario_key: str
    speech_pattern_key: str
    model_name: str
    conversation: list
    route: list
    route_steps: list
    route_valid: bool
    route_reaches_goal: bool
    route_correct: bool
    route_duration_min: int | None
    runtime_sec: float
    metrics_text: str = ""
    extra: dict = field(default_factory=dict)


class NullEventQueue:
    def put(self, event):
        pass
