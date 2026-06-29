import unittest

from coop_navigation_sds.NaturalLanguageGeneration.caller.prompting import (
    agent_a_route_reaction,
    agent_a_alternative_request,
    build_agent_a_system,
    build_agent_b_phase_instruction,
    build_agent_b_stage_instruction,
    build_agent_b_system,
    generate_agent_a_template,
)
from coop_navigation_sds.NaturalLanguageGeneration.caller.agents import fallback_reply
from coop_navigation_sds.NaturalLanguageGeneration.caller.config import PERSONAS
from coop_navigation_sds.NaturalLanguageGeneration.assistant.pipeline import DialogState, VerbalTransformationPipeline
from coop_navigation_sds.NaturalLanguageUnderstanding.interpreter import NaturalRouteInterpreter
from coop_navigation_sds.DialogManagement.stages import ConversationStage
from coop_navigation_sds.TransportNetwork.constraints import (
    acceptable_duration_limit,
    optimal_constraint_route,
    layered_optimal_routes,
    optimal_route_duration_min,
    route_allowed_modes,
    stage_route_options,
    stage_viability_report,
)
from coop_navigation_sds.TransportNetwork.routes import candidate_time_routes, route_text_from_steps
from coop_navigation_sds.TransportNetwork import DEFAULT_TEST_CASE, get_test_case
from coop_navigation_sds.NaturalLanguageGeneration.prompt_audit import PROMPT_POLICY_VERSION
from coop_navigation_sds.TransportNetwork.scenarios import SCENARIOS


class PromptingTests(unittest.TestCase):
    def test_agent_b_prompt_audit_records_exact_messages_and_acceptance(self):
        test_case = get_test_case(DEFAULT_TEST_CASE)
        valid_reply = fallback_reply(
            "Agent B",
            test_case.scenario,
            route_index=0,
            persona=test_case.persona,
        )

        class ValidRouteModel:
            def generate_messages(self, _messages):
                return valid_reply

        pipeline = VerbalTransformationPipeline(ValidRouteModel())
        state = DialogState(
            test_case,
            [("Agent A", test_case.opening_utterance())],
        )

        self.assertEqual(pipeline.run_agent_b(state), valid_reply)
        audits = pipeline.consume_prompt_audits()
        self.assertEqual(len(audits), 1)
        self.assertEqual(audits[0]["prompt_policy_version"], PROMPT_POLICY_VERSION)
        self.assertEqual(audits[0]["agent"], "Agent B")
        self.assertTrue(audits[0]["accepted"])
        self.assertEqual(audits[0]["decision"], "accepted")
        self.assertEqual(audits[0]["delivery_source"], "model")
        self.assertEqual(audits[0]["messages"][0]["role"], "system")
        self.assertEqual(len(audits[0]["prompt_sha256"]), 64)
        self.assertEqual(pipeline.consume_prompt_audits(), [])

    def test_agent_b_prompt_audit_preserves_rejected_model_drafts(self):
        test_case = get_test_case(DEFAULT_TEST_CASE)

        class InvalidRouteModel:
            def generate_messages(self, _messages):
                return "Take metro line M1 from Bravo to Alpha."

        pipeline = VerbalTransformationPipeline(InvalidRouteModel())
        pipeline.run_agent_b(
            DialogState(test_case, [("Agent A", test_case.opening_utterance())])
        )
        audits = pipeline.consume_prompt_audits()

        self.assertGreaterEqual(len(audits), 1)
        self.assertEqual(audits[0]["raw_output"], "Take metro line M1 from Bravo to Alpha.")
        self.assertFalse(audits[0]["accepted"])
        self.assertEqual(audits[0]["decision"], "incomplete_or_invalid_route")
        self.assertEqual(audits[-1]["delivery_source"], "deterministic_route_fallback")

    def setUp(self):
        self.persona = get_test_case(DEFAULT_TEST_CASE).persona
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
        self.assertIn("station and line names", prompt)
        self.assertIn("Known station names:", prompt)
        self.assertIn("Known line names:", prompt)
        self.assertIn("not which stations a line serves", prompt)
        self.assertIn("Focused commuter", prompt)
        self.assertIn("Central", prompt)
        self.assertIn("Museum", prompt)
        self.assertIn("Priority 1", prompt)
        self.assertIn("Never reveal more than one new constraint", prompt)
        self.assertIn("Private constraints", prompt)
        self.assertNotIn("Route candidates", prompt)
        self.assertNotIn("Transfer cost", prompt)

    def test_acceptable_duration_limit_defaults_under_fifty_percent_over_optimal(self):
        limit = acceptable_duration_limit(self.real_scenario, self.persona)
        optimal = optimal_route_duration_min(self.real_scenario, self.persona)

        self.assertGreaterEqual(limit, optimal)
        self.assertLess(limit, optimal * 1.5)

    def test_acceptable_duration_limit_ratio_is_configurable(self):
        scenario = dict(self.real_scenario)
        scenario["acceptable_duration_ratio"] = 1.2

        limit = acceptable_duration_limit(scenario, self.persona)
        optimal = optimal_route_duration_min(scenario, self.persona)

        self.assertGreaterEqual(limit, optimal)
        self.assertLess(limit, optimal * 1.2)

    def test_stage_viability_report_verifies_suboptimal_options_per_stage(self):
        default_report = stage_viability_report(get_test_case(DEFAULT_TEST_CASE).scenario, get_test_case(DEFAULT_TEST_CASE).persona)
        self.assertTrue(default_report["all_stage_requirements_satisfied"])

        test_case = get_test_case("midday_transfer").with_persona("distracted_multitasker")

        report = stage_viability_report(test_case.scenario, test_case.persona)

        self.assertTrue(report["all_stage_requirements_satisfied"])
        self.assertEqual(report["acceptable_duration_ratio"], 1.5)
        self.assertEqual(len(report["stages"]), 3)
        self.assertTrue(all(stage["suboptimal_option_count"] >= 1 for stage in report["stages"]))
        self.assertTrue(all(stage["constraint_changes_optimal_route"] for stage in report["stages"]))

    def test_optimal_routes_are_calculated_for_each_progressive_layer(self):
        layers = layered_optimal_routes(self.real_scenario, self.persona, max_constraints=3)

        self.assertEqual(
            [layer["layer"] for layer in layers],
            ["validity", "time", "constraint_1", "constraint_2", "constraint_3"],
        )
        self.assertEqual([len(layer["stated_constraints"]) for layer in layers], [0, 0, 1, 2, 3])
        constraint_paths = []
        for layer in layers:
            self.assertTrue(layer["available"])
            self.assertEqual(layer["route"][0], self.real_scenario["start_station"])
            self.assertEqual(layer["route"][-1], self.real_scenario["destination_station"])
            for step in layer["steps"]:
                if step.get("mode") != "walking":
                    self.assertIn(f"({step['line']}", layer["path_text"])
            if layer["layer"].startswith("constraint_"):
                constraint_paths.append(layer["path_text"])
        progressive_paths = [layer["path_text"] for layer in layers[1:]]
        for previous, current in zip(progressive_paths, progressive_paths[1:]):
            self.assertNotEqual(previous, current)
        self.assertEqual(len(constraint_paths), len(set(constraint_paths)))

    def test_all_standard_scenarios_require_constraint_driven_route_changes(self):
        from coop_navigation_sds.TransportNetwork.test_cases import TEST_CASES

        for key, test_case in TEST_CASES.items():
            with self.subTest(test_case=key):
                report = stage_viability_report(
                    test_case.scenario,
                    test_case.persona,
                    max_constraints=3,
                )
                self.assertTrue(report["require_constraint_route_changes"])
                self.assertTrue(report["all_stage_requirements_satisfied"])
                self.assertTrue(all(
                    stage["constraint_changes_optimal_route"]
                    for stage in report["stages"]
                ))

    def test_agent_b_phase_instruction_changes_by_turn(self):
        first = build_agent_b_phase_instruction(0, "Museum")
        later = build_agent_b_phase_instruction(5, "Museum")
        self.assertIn("Museum", first)
        self.assertIn("Confirm the best route", later)

    def test_agent_b_stage_instruction_is_explicit_and_contextual(self):
        comparison = build_agent_b_stage_instruction(ConversationStage.COMPARISON, "Museum")
        refinement = build_agent_b_stage_instruction(ConversationStage.REFINEMENT, "Museum")
        confirmation = build_agent_b_stage_instruction(ConversationStage.CONFIRMATION, "Museum")

        self.assertIn("latest comparison request", comparison)
        self.assertIn("remembered accepted details", refinement)
        self.assertIn("without introducing a new option", confirmation)

    def test_agent_a_template_selects_persona_specific_text(self):
        text = generate_agent_a_template(0, self.persona, self.scenario)
        self.assertIn("Museum", text)
        self.assertIn("Central", text)
        self.assertIn("08:00", text)

    def test_agent_b_system_prompt_includes_route_context(self):
        prompt = build_agent_b_system(self.scenario, self.persona)
        self.assertIn("Agent B", prompt)
        self.assertIn("Start: Central", prompt)
        self.assertIn("Destination: Museum", prompt)
        self.assertIn("verified route candidates", prompt)
        self.assertIn("Known station names:", prompt)
        self.assertIn("Known line names:", prompt)
        self.assertLess(len(prompt.split()), 350)

    def test_agent_b_system_prioritizes_validity_over_constraints(self):
        prompt = build_agent_b_system(self.scenario, self.persona)
        self.assertIn("valid route first", prompt)
        self.assertIn("line legs", prompt)

    def test_agent_a_reacts_to_missing_route(self):
        text = agent_a_route_reaction(
            1,
            self.persona,
            self.scenario,
            [("Agent B", "I can help with that and keep it simple.")],
        )
        self.assertIn("actual route", text)
        self.assertIn("Museum", text)

    def test_agent_a_answers_word_clarification_with_only_the_known_term(self):
        text = agent_a_route_reaction(
            1,
            self.persona,
            self.real_scenario,
            [("Agent B", "I heard 'Harbour' unclearly. Did you mean Harbor or Grove? Please repeat the start station, destination, and time.")],
        )

        self.assertEqual(text, "Harbor.")

    def test_agent_a_asks_about_possible_misheard_words_when_reply_makes_no_sense(self):
        text = agent_a_route_reaction(
            1,
            self.persona,
            self.real_scenario,
            [("Agent B", "Take wings of love had a harder; it takes 12 minutes.")],
        )

        self.assertIn("did you mean", text.lower())
        self.assertNotIn("actual route", text)
        self.assertLessEqual(len(text.split()), 8)

    def test_agent_a_reacts_to_connected_route_with_comparison_request(self):
        reply = route_text_from_steps(stage_route_options(self.real_scenario, self.persona)[0]["steps"])
        text = agent_a_route_reaction(
            1,
            self.persona,
            self.real_scenario,
            [("Agent B", reply)],
        )
        self.assertIn("route and timing work", text)
        self.assertIn("Can you make it", text)

    def test_agent_a_requests_persona_specific_alternative_constraints(self):
        persona = {
            "preferences": {
                "priority": "simple route first",
                "switching": "prefers fewer line changes",
                "fullness": "prefers less crowded trains",
            }
        }

        request = agent_a_alternative_request(persona)

        self.assertIn("avoids near-capacity trains", request)
        self.assertIn("fewer line changes", request)

    def test_research_personas_and_scenarios_cover_delay_fullness_and_multi_destination(self):
        self.assertIn("delay_sensitive_traveler", PERSONAS)
        self.assertIn("crowd_averse_rider", PERSONAS)
        self.assertIn("multi_destination_errands", SCENARIOS)
        self.assertGreater(len(SCENARIOS["multi_destination_errands"]["destination_stations"]), 1)

        request = agent_a_alternative_request(PERSONAS["delay_sensitive_traveler"])

        self.assertIn("lower delay risk", request)
        self.assertIn("safer transfers", request)

    def test_constraint_route_and_fallback_delays_secondary_constraints_until_asked(self):
        test_case = get_test_case("airport_connection")
        constraint_route = optimal_constraint_route(test_case.scenario, test_case.persona)
        reply = fallback_reply("Agent B", test_case.scenario, route_index=0, persona=test_case.persona)
        constrained_reply = fallback_reply(
            "Agent B",
            test_case.scenario,
            route_index=0,
            persona=test_case.persona,
            conversation=[("Agent A", "I need lower delay risk.")],
        )

        self.assertIsNotNone(constraint_route)
        self.assertGreater(constraint_route.delay_probability, 0.0)
        self.assertNotIn("delay risk", reply)
        self.assertNotIn("capacity", reply)
        self.assertIn("delay risk", constrained_reply.lower())
        self.assertNotIn("capacity", constrained_reply.lower())
        self.assertNotIn("percent", constrained_reply)

    def test_fallback_agent_b_turns_are_concise_for_speech(self):
        test_case = get_test_case("airport_connection")

        reply = fallback_reply("Agent B", test_case.scenario, route_index=0, persona=test_case.persona)

        self.assertLessEqual(len(reply.split()), 40)
        self.assertIn("take", reply.lower())
        self.assertIn("minutes", reply)

    def test_agent_a_reaction_turns_are_concise_for_speech(self):
        reply = route_text_from_steps(stage_route_options(self.real_scenario, self.persona)[0]["steps"])
        text = agent_a_route_reaction(
            1,
            self.persona,
            self.real_scenario,
            [("Agent B", reply)],
        )

        self.assertLessEqual(len(text.split()), 22)
        self.assertIn("Can you make it", text)

    def test_agent_a_final_reaction_closes_after_two_constraints(self):
        stage_one = stage_route_options(self.real_scenario, self.persona)[0]
        stage_two = stage_route_options(self.real_scenario, self.persona, ("transfer_miss",))[0]
        stage_three_options = stage_route_options(self.real_scenario, self.persona, ("transfer_miss", "tickets"))
        stage_three = next(
            (option for option in stage_three_options if option["route"] != stage_two["route"]),
            stage_three_options[0],
        )
        text = agent_a_route_reaction(
            3,
            self.persona,
            self.real_scenario,
            [
                ("Agent B", route_text_from_steps(stage_one["steps"])),
                ("Agent A", "Can you make it with safer transfer timing?"),
                ("Agent B", route_text_from_steps(stage_two["steps"])),
                ("Agent A", "Can you use only my metro and tram tickets? I cannot take bus."),
                ("Agent B", route_text_from_steps(stage_three["steps"])),
            ],
        )
        self.assertIn("Thanks", text)
        self.assertIn("take", text)

    def test_agent_a_critiques_slower_alternative(self):
        limit = acceptable_duration_limit(self.real_scenario, self.persona)
        slow = next(
            item for item in candidate_time_routes(
                self.real_scenario["start_station"],
                self.real_scenario["destination_station"],
                self.real_scenario["start_time_min"],
                self.real_scenario["transfer_time_min"],
                limit=80,
                max_extra_stops=8,
                max_paths=50000,
                allowed_modes=route_allowed_modes(self.real_scenario, self.persona),
            )
            if item[0] > limit
        )
        text = agent_a_route_reaction(
            2,
            self.persona,
            self.real_scenario,
            [
                ("Agent B", route_text_from_steps(stage_route_options(self.real_scenario, self.persona)[0]["steps"])),
                ("Agent A", "Now compare one faster valid route."),
                ("Agent B", route_text_from_steps(slow[2])),
            ],
        )

        self.assertIn("too long", text)
        self.assertNotIn("Now can you make it", text)

    def test_agent_b_pipeline_rejects_partial_route_and_uses_valid_fallback(self):
        class PartialRouteModel:
            name = "partial-route-model"

            def generate_messages(self, messages):
                return "Take metro line M1 from Bravo to Alpha."

        test_case = get_test_case(DEFAULT_TEST_CASE)
        state = DialogState(
            test_case=test_case,
            conversation=[("Agent A", test_case.opening_utterance())],
            turn=0,
        )
        reply = VerbalTransformationPipeline(PartialRouteModel()).run_agent_b(state)

        self.assertIn("Harbor", reply)
        self.assertIn("take", reply.lower())
        self.assertLessEqual(len(reply.split()), 45)

    def test_interpreter_expands_boarding_route_mentions(self):
        interpreter = NaturalRouteInterpreter()
        text = "Take Core Tram. Stations: Bravo to Golf to Mike to Sierra to Harbor. Boarding: Bravo to Harbor. Total 24 minutes."

        route = interpreter.interpret_reply(text, self.real_scenario)

        self.assertEqual(route[0], self.real_scenario["start_station"])
        self.assertEqual(route[-1], self.real_scenario["destination_station"])
        self.assertGreater(len(route), 4)

    def test_interpreter_ignores_duplicate_summary_mentions(self):
        interpreter = NaturalRouteInterpreter()
        option = stage_route_options(self.real_scenario, self.persona)[0]
        station_text = " to ".join(option["route"])
        text = f"Stations: {station_text}. Boarding: Bravo to Harbor. Total {option['duration_min']} minutes."

        route = interpreter.interpret_reply(text, self.real_scenario)

        self.assertEqual(route, option["route"])

    def test_interpreter_uses_spoken_line_for_compact_boarding_route(self):
        interpreter = NaturalRouteInterpreter()
        text = "Take metro line M1 from Bravo to Harbor. It takes 45 minutes, with no changes."

        route = interpreter.interpret_reply(text, self.real_scenario)

        self.assertEqual(route[0], "Bravo")
        self.assertEqual(route[-1], "Harbor")
        self.assertGreaterEqual(len(route), 3)
