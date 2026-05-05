from dataclasses import dataclass
from dataclasses import replace

from personas import DEFAULT_PERSONA, get_persona
from route_planner import fmt_time
from scenarios import DEFAULT_SCENARIO, get_scenario


@dataclass(frozen=True)
class StandardizedTestCase:
    key: str
    name: str
    persona_key: str
    scenario_key: str

    @property
    def persona(self):
        persona = dict(get_persona(self.persona_key))
        persona["key"] = self.persona_key
        return persona

    @property
    def scenario(self):
        return get_scenario(self.scenario_key)

    def opening_utterance(self) -> str:
        scenario = self.scenario
        return (
            f"I am at {scenario['start_station']} at {fmt_time(scenario['start_time_min'])} "
            f"and need to get to {scenario['destination_station']}. "
            "Please help me compare the possible routes and build the best one step by step."
        )

    def with_persona(self, persona_key: str):
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
    return TEST_CASES.get(test_case_key, TEST_CASES[DEFAULT_TEST_CASE])
