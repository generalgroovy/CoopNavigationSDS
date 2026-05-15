import unittest

from minillama.agent_a.prompting import (
    build_agent_a_system,
    build_agent_b_phase_instruction,
    build_agent_b_system,
    generate_agent_a_template,
)


class PromptingTests(unittest.TestCase):
    def setUp(self):
        self.persona = {
            "key": "focused_commuter",
            "name": "Focused commuter",
            "description": "Direct, time-conscious, practical.",
            "preferences": {
                "priority": "fastest route first",
                "switching": "accepts line changes for a faster route",
                "fullness": "does not mind fuller trains",
            },
        }
        self.scenario = {
            "start_time_min": 480,
            "start_station": "Central",
            "destination_station": "Museum",
            "transfer_time_min": 4,
        }

    def test_agent_a_system_prompt_includes_persona_and_context(self):
        prompt = build_agent_a_system(self.persona, self.scenario)
        self.assertIn("Agent A", prompt)
        self.assertIn("Focused commuter", prompt)
        self.assertIn("Central", prompt)
        self.assertIn("Museum", prompt)

    def test_agent_b_phase_instruction_changes_by_turn(self):
        first = build_agent_b_phase_instruction(0, "Museum")
        later = build_agent_b_phase_instruction(5, "Museum")
        self.assertIn("Museum", first)
        self.assertIn("Confirm the best route", later)

    def test_agent_a_template_selects_persona_specific_text(self):
        text = generate_agent_a_template(0, self.persona, self.scenario)
        self.assertIn("Museum", text)
        self.assertIn("first set of lines", text)

    def test_agent_b_system_prompt_includes_route_context(self):
        prompt = build_agent_b_system(self.scenario, self.persona)
        self.assertIn("Agent B", prompt)
        self.assertIn("Transfer cost: 4 min", prompt)
