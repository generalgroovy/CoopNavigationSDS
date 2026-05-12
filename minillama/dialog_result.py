"""Data model for the completed dialog result and a null event queue used by non-GUI experiment controllers.
"""
from dataclasses import dataclass, field


@dataclass
class DialogResult:
    """Data model containing the completed conversation, inferred route, route quality flags, timing, and extra run metadata.
    """
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
    """Controller helper that accepts events without displaying them for batch experiment runs.
    """
    def put(self, event):
        """Put method for this module's MVC responsibility.
        
        Args:
            event: Input value used by `put`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        pass
