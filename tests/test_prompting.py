import unittest

from minillama.agent_a.prompting import (
    agent_a_route_reaction,
    agent_a_alternative_request,
    build_agent_a_system,
    build_agent_b_phase_instruction,
    build_agent_b_system,
    generate_agent_a_template,
)
from minillama.agent_a.agents import fallback_reply
from minillama.agent_a.config import PERSONAS
from minillama.agent_b.pipeline import DialogState, VerbalTransformationPipeline
from minillama.evaluation.route_interpreter import NaturalRouteInterpreter
from minillama.model.route_constraints import optimal_constraint_route
from minillama.test_cases import DEFAULT_TEST_CASE, get_test_case
from minillama.test_cases.scenarios import SCENARIOS


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
        self.real_scenario = get_test_case(DEFAULT_TEST_CASE).scenario

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
        self.assertIn("Central", text)
        self.assertIn("08:00", text)

    def test_agent_b_system_prompt_includes_route_context(self):
        prompt = build_agent_b_system(self.scenario, self.persona)
        self.assertIn("Agent B", prompt)
        self.assertIn("Transfer cost: 4 minutes", prompt)

    def test_agent_b_system_prioritizes_validity_over_constraints(self):
        prompt = build_agent_b_system(self.scenario, self.persona)
        self.assertIn("Validity comes first", prompt)
        self.assertIn("boarding stations", prompt)

    def test_agent_a_reacts_to_missing_route(self):
        text = agent_a_route_reaction(
            1,
            self.persona,
            self.scenario,
            [("Agent B", "I can help with that and keep it simple.")],
        )
        self.assertIn("actual route", text)
        self.assertIn("Museum", text)

    def test_agent_a_reacts_to_connected_route_with_comparison_request(self):
        text = agent_a_route_reaction(
            1,
            self.persona,
            self.real_scenario,
            [("Agent B", "Take Ring from Bravo to Alpha to Golf, then Diagonal-SE-6 to November to Uniform to Birch to Ivy, then Ring to Harbor.")],
        )
        self.assertIn("Valid:", text)
        self.assertIn("one faster valid route", text)
        self.assertIn("faster", text)

    def test_agent_a_requests_persona_specific_alternative_constraints(self):
        persona = {
            "preferences": {
                "priority": "simple route first",
                "switching": "prefers fewer line changes",
                "fullness": "prefers less crowded trains",
            }
        }

        request = agent_a_alternative_request(persona)

        self.assertIn("less full", request)
        self.assertIn("fewer line changes", request)

    def test_research_personas_and_scenarios_cover_delay_fullness_and_multi_destination(self):
        self.assertIn("delay_sensitive_traveler", PERSONAS)
        self.assertIn("crowd_averse_rider", PERSONAS)
        self.assertIn("multi_destination_errands", SCENARIOS)
        self.assertGreater(len(SCENARIOS["multi_destination_errands"]["destination_stations"]), 1)

        request = agent_a_alternative_request(PERSONAS["delay_sensitive_traveler"])

        self.assertIn("lower delay risk", request)

    def test_constraint_route_and_fallback_surface_delay_risk(self):
        test_case = get_test_case("airport_connection")
        constraint_route = optimal_constraint_route(test_case.scenario, test_case.persona)
        reply = fallback_reply("Agent B", test_case.scenario, route_index=0, persona=test_case.persona)

        self.assertIsNotNone(constraint_route)
        self.assertGreater(constraint_route.delay_probability, 0.0)
        self.assertIn("delay risk", reply)
        self.assertIn("full", reply)

    def test_fallback_agent_b_turns_are_concise_for_speech(self):
        test_case = get_test_case("airport_connection")

        reply = fallback_reply("Agent B", test_case.scenario, route_index=0, persona=test_case.persona)

        self.assertLessEqual(len(reply.split()), 28)
        self.assertIn("Boarding:", reply)
        self.assertIn("Total", reply)

    def test_agent_a_reaction_turns_are_concise_for_speech(self):
        text = agent_a_route_reaction(
            1,
            self.persona,
            self.real_scenario,
            [("Agent B", "Boarding: Bravo to Golf to Ivy to Harbor. Lines: Ring to Diagonal-SE-6 to Ring. Total 28 minutes, 2 line changes.")],
        )

        self.assertLessEqual(len(text.split()), 22)
        self.assertIn("Compare one", text)

    def test_agent_a_final_reaction_asks_for_confirmation(self):
        text = agent_a_route_reaction(
            3,
            self.persona,
            self.real_scenario,
            [("Agent B", "Take Ring from Bravo to Alpha to Golf, then Diagonal-SE-6 to November to Uniform to Birch to Ivy, then Ring to Harbor.")],
        )
        self.assertIn("confirm", text.lower())
        self.assertIn("total time", text)

    def test_agent_a_critiques_slower_alternative(self):
        text = agent_a_route_reaction(
            2,
            self.persona,
            self.real_scenario,
            [
                ("Agent B", "Take Ring from Bravo to Golf. Change at Golf to Diagonal-SE-6 to Ivy. Change at Ivy to Ring to Harbor. Boarding: Bravo to Golf to Ivy to Harbor. Total 28 minutes."),
                ("Agent A", "Now compare one faster valid route."),
                ("Agent B", "Take Ring from Bravo to Mike. Change at Mike to East-West-3 to November. Change at November to Diagonal-SE-6 to Ivy. Change at Ivy to Ring to Harbor. Boarding: Bravo to Mike to November to Ivy to Harbor. Total 38 minutes."),
            ],
        )

        self.assertIn("slower", text)
        self.assertIn("earlier 28-minute route", text)

    def test_agent_b_pipeline_rejects_partial_route_and_uses_valid_fallback(self):
        class PartialRouteModel:
            name = "partial-route-model"

            def generate_messages(self, messages):
                return "Take Ring from Bravo to Alpha."

        test_case = get_test_case(DEFAULT_TEST_CASE)
        state = DialogState(test_case=test_case, conversation=[], turn=0)
        reply = VerbalTransformationPipeline(PartialRouteModel()).run_agent_b(state)

        self.assertIn("Harbor", reply)
        self.assertIn("Boarding:", reply)
        self.assertLessEqual(len(reply.split()), 45)

    def test_interpreter_expands_boarding_route_mentions(self):
        interpreter = NaturalRouteInterpreter()
        text = "Take Ring from Bravo to Golf. Change at Golf to Diagonal-SE-6 from Golf to Ivy. Boarding: Bravo to Golf to Ivy to Harbor. Total 31 minutes."

        route = interpreter.interpret_reply(text, self.real_scenario)

        self.assertEqual(route[0], self.real_scenario["start_station"])
        self.assertEqual(route[-1], self.real_scenario["destination_station"])
        self.assertGreater(len(route), 4)
