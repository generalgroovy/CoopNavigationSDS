"""Standardized test-case model for scenario/persona combinations and opening utterances.
"""
from dataclasses import dataclass
from dataclasses import replace

from minillama.agent_a.personas import get_persona
from minillama.model.route_planner import fmt_time
from minillama.test_cases.config import DEFAULT_TEST_CASE, TEST_CASE_SPECS
from minillama.test_cases.scenarios import get_scenario


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
        destinations = scenario.get("destination_stations") or [scenario["destination_station"]]
        if len(destinations) > 1:
            later_destination_text = ", then ".join(destinations[1:])
            return (
                f"I'm at {scenario['start_station']} at {fmt_time(scenario['start_time_min'])}. "
                f"First to {scenario['destination_station']}, then {later_destination_text}. Which route should I take first?"
            )
        return (
            f"I'm at {scenario['start_station']} at {fmt_time(scenario['start_time_min'])}, "
            f"going to {scenario['destination_station']}. Which route should I take?"
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
    key: StandardizedTestCase(
        key=key,
        name=spec["name"],
        persona_key=spec["persona_key"],
        scenario_key=spec["scenario_key"],
    )
    for key, spec in TEST_CASE_SPECS.items()
}


def get_test_case(test_case_key: str):
    """Get test case function for this module's MVC responsibility.
    
    Args:
        test_case_key: Input value used by `get_test_case`; see the function signature and caller context for the expected type.
    
    Returns:
        The computed value or side effect documented by the implementation.
    """
    return TEST_CASES.get(test_case_key, TEST_CASES[DEFAULT_TEST_CASE])
