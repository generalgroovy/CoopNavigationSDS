"""Standardized test-case model for scenario/persona combinations and opening utterances.
"""
from dataclasses import dataclass
from dataclasses import replace

from minillama.personas import DEFAULT_PERSONA, get_persona, preference_text
from minillama.route_planner import fmt_time
from minillama.scenarios import DEFAULT_SCENARIO, get_scenario


@dataclass(frozen=True)
class StandardizedTestCase:
    """Experiment test-case model binding scenario, persona, and opening utterance behavior.
    """
    key: str
    name: str
    persona_key: str
    scenario_key: str

    @property
    def persona(self):
        """Persona method for this module's MVC responsibility.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        persona = dict(get_persona(self.persona_key))
        persona["key"] = self.persona_key
        return persona

    @property
    def scenario(self):
        """Scenario method for this module's MVC responsibility.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        return get_scenario(self.scenario_key)

    def opening_utterance(self) -> str:
        """Opening utterance method for this module's MVC responsibility.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        scenario = self.scenario
        return (
            f"I'm at {scenario['start_station']} at {fmt_time(scenario['start_time_min'])}, "
            f"and I need to get to {scenario['destination_station']}. "
            f"{preference_text(self.persona)} "
            "Can you compare the connected options and help me choose the best one?"
        )

    def with_persona(self, persona_key: str):
        """With persona method for this module's MVC responsibility.
        
        Args:
            persona_key: Input value used by `with_persona`; see the function signature and caller context for the expected type.
        
        Returns:
            The computed value or side effect documented by the implementation.
        """
        return replace(
            self,
            key=f"{self.key}:{persona_key}",
            name=f"{self.name} / {persona_key}",
            persona_key=persona_key,
        )


TEST_CASES = {
    "default": StandardizedTestCase(
        key="default",
        name="Default focused commuter route test",
        persona_key=DEFAULT_PERSONA,
        scenario_key=DEFAULT_SCENARIO,
    ),
    "short_cross": StandardizedTestCase(
        key="short_cross",
        name="Short cross-network route test",
        persona_key=DEFAULT_PERSONA,
        scenario_key="short_cross",
    ),
    "long_cross": StandardizedTestCase(
        key="long_cross",
        name="Long cross-network route test",
        persona_key=DEFAULT_PERSONA,
        scenario_key="long_cross",
    ),
}

DEFAULT_TEST_CASE = "default"


def get_test_case(test_case_key: str):
    """Get test case function for this module's MVC responsibility.
    
    Args:
        test_case_key: Input value used by `get_test_case`; see the function signature and caller context for the expected type.
    
    Returns:
        The computed value or side effect documented by the implementation.
    """
    return TEST_CASES.get(test_case_key, TEST_CASES[DEFAULT_TEST_CASE])
