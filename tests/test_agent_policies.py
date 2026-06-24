import unittest

from coop_navigation_sds.NaturalLanguageGeneration.assistant.plugin_registry import (
    AGENT_B_MODEL_SPECS,
    DiversePlannerAgentBPlugin,
    ParetoPlannerAgentBPlugin,
    RobustPlannerAgentBPlugin,
    agent_b_plugin_description,
    available_agent_b_plugin_keys,
    create_agent_b_plugin,
)
from coop_navigation_sds.NaturalLanguageGeneration.assistant.pipeline import DialogState
from coop_navigation_sds.NaturalLanguageGeneration.caller.responder import (
    LLMAgentAResponder,
    TemplateAgentAResponder,
    agent_a_uses_model,
    available_agent_a_types,
    normalize_agent_a_type,
)
from coop_navigation_sds.DialogManagement.stages import agent_memory_view
from coop_navigation_sds.TransportNetwork import DEFAULT_TEST_CASE, get_test_case


class AgentPolicyTests(unittest.TestCase):
    def test_opening_states_start_station_and_need_for_lines(self):
        opening = get_test_case(DEFAULT_TEST_CASE).opening_utterance()

        self.assertIn("starting at station", opening.lower())
        self.assertIn("transit lines", opening.lower())
        self.assertIn("Which lines", opening)

    def test_agent_a_types_are_explicit_and_legacy_compatible(self):
        self.assertEqual(available_agent_a_types(), ("staged", "tinyllama", "userlm"))
        self.assertEqual(normalize_agent_a_type(None, False), "staged")
        self.assertEqual(normalize_agent_a_type("minillama"), "staged")
        self.assertEqual(normalize_agent_a_type("tinyllama"), "tinyllama")
        self.assertEqual(normalize_agent_a_type(None, True), "userlm")
        self.assertTrue(agent_a_uses_model("tinyllama"))
        self.assertTrue(agent_a_uses_model("userlm"))
        self.assertFalse(agent_a_uses_model("staged"))
        self.assertIsInstance(TemplateAgentAResponder(), TemplateAgentAResponder)

    def test_research_agent_b_plugins_are_built_in_and_model_free(self):
        self.assertEqual(
            available_agent_b_plugin_keys(),
            ["llm", "simple", "pareto", "robust", "diverse"],
        )
        self.assertIsInstance(create_agent_b_plugin("pareto", None), ParetoPlannerAgentBPlugin)
        self.assertIsInstance(create_agent_b_plugin("robust", None), RobustPlannerAgentBPlugin)
        self.assertIsInstance(create_agent_b_plugin("diverse", None), DiversePlannerAgentBPlugin)

    def test_three_spoken_agent_b_models_have_distinct_public_profiles(self):
        profiles = [AGENT_B_MODEL_SPECS[key] for key in ("pareto", "robust", "diverse")]

        self.assertEqual({profile.style for profile in profiles}, {"balanced", "reassuring", "exploratory"})
        self.assertEqual(len({profile.description for profile in profiles}), 3)
        self.assertIn("reliability", agent_b_plugin_description("robust").lower())

    def test_spoken_agent_b_models_react_to_latest_caller_request(self):
        test_case = get_test_case("morning_peak_cross_city")
        conversation = [
            ("Agent A", test_case.opening_utterance()),
            ("Agent B", "Take metro line M1 from Bravo to Harbor. It takes 12 minutes."),
            ("Agent A", "Please compare a genuinely different alternative."),
        ]
        state = DialogState(test_case, conversation, turn=1)

        replies = {
            key: create_agent_b_plugin(key, None).run_agent_b(state)
            for key in ("pareto", "robust", "diverse")
        }

        self.assertEqual(len(set(replies.values())), 3)
        self.assertIn("balanced", replies["pareto"].lower())
        self.assertIn("reliable", replies["robust"].lower())
        self.assertIn("different", replies["diverse"].lower())

    def test_agent_memory_tracks_route_and_active_constraints(self):
        test_case = get_test_case("morning_peak_cross_city")
        conversation = [
            ("Agent A", test_case.opening_utterance()),
            ("Agent B", "Take metro line M1 from Bravo to Harbor. 12 minutes, no changes."),
            ("Agent A", "That route works. Can you avoid near-capacity trains?"),
        ]

        memory = agent_memory_view(
            "Agent A",
            conversation,
            scenario=test_case.scenario,
            persona=test_case.persona,
        )
        summary = memory.prompt_summary()

        self.assertIn("Bravo", summary)
        self.assertIn("Harbor", summary)
        self.assertIn("current route candidate", summary)
        self.assertIn("fullness", memory.active_constraints)
        self.assertIn("currently active caller constraints: fullness", summary)
        self.assertEqual(memory.task_variables["start_station"], test_case.scenario["start_station"])
        self.assertEqual(memory.task_variables["destination_station"], test_case.scenario["destination_station"])
        self.assertEqual(memory.task_variables["start_time_min"], test_case.scenario["start_time_min"])
        self.assertEqual(memory.missing_task_variables, ())
        self.assertEqual(memory.active_constraint_details["fullness"], "avoid near-capacity trains")

    def test_agent_memory_keeps_agent_b_task_facts_separate_from_hidden_scenario(self):
        test_case = get_test_case("morning_peak_cross_city")
        agent_b_heard = [
            ("Agent A", "I am going to Harbor at eight seven."),
        ]

        memory = agent_memory_view(
            "Agent B",
            agent_b_heard,
            scenario=test_case.scenario,
            persona=test_case.persona,
        )
        summary = memory.prompt_summary()

        self.assertIsNone(memory.task_variables["start_station"])
        self.assertEqual(memory.task_variables["destination_station"], "Harbor")
        self.assertEqual(memory.task_variables["start_time_min"], 8 * 60 + 7)
        self.assertIn("start_station", memory.missing_task_variables)
        self.assertIn("start=unknown", summary)
        self.assertNotIn(f"start={test_case.scenario['start_station']}", summary)

    def test_research_policies_choose_for_different_objectives(self):
        options = [
            {
                "duration_min": 10,
                "route": ["A", "B", "D"],
                "line_change_count": 1,
                "near_capacity_count": 1,
                "delay_risk_class": "high",
                "transfer_miss_risk_class": "medium",
            },
            {
                "duration_min": 13,
                "route": ["A", "C", "D"],
                "line_change_count": 0,
                "near_capacity_count": 0,
                "delay_risk_class": "low",
                "transfer_miss_risk_class": "low",
            },
            {
                "duration_min": 12,
                "route": ["A", "E", "F", "D"],
                "line_change_count": 1,
                "near_capacity_count": 0,
                "delay_risk_class": "medium",
                "transfer_miss_risk_class": "low",
            },
        ]
        prior = {("A", "C", "D")}
        robust = RobustPlannerAgentBPlugin().select_option(options, prior)
        diverse = DiversePlannerAgentBPlugin().select_option(options, prior)

        self.assertEqual(robust["route"], ["A", "C", "D"])
        self.assertNotEqual(diverse["route"], robust["route"])

        pareto_options = [
            {
                "duration_min": 11,
                "route": ["A", "B", "D"],
                "line_change_count": 0,
                "near_capacity_count": 0,
                "delay_risk_class": "medium",
                "transfer_miss_risk_class": "medium",
            },
            {
                "duration_min": 14,
                "route": ["A", "C", "E", "D"],
                "line_change_count": 2,
                "near_capacity_count": 0,
                "delay_risk_class": "low",
                "transfer_miss_risk_class": "low",
            },
        ]
        pareto = ParetoPlannerAgentBPlugin().select_option(pareto_options, set())
        robust = RobustPlannerAgentBPlugin().select_option(pareto_options, set())
        self.assertNotEqual(pareto["route"], robust["route"])

    def test_userlm_cannot_reveal_two_constraints_in_one_turn(self):
        class OverreachingModel:
            def generate_messages(self, _messages):
                return "Can you avoid crowded trains and also keep delay risk low?"

        test_case = get_test_case("morning_peak_cross_city").with_persona("distracted_multitasker")
        conversation = [
            ("Agent A", test_case.opening_utterance()),
            ("Agent B", "Take metro line M1 from Bravo to Harbor. It takes 12 minutes, with no changes."),
        ]
        reply = LLMAgentAResponder(OverreachingModel()).reply(
            0,
            test_case.persona,
            test_case.scenario,
            conversation,
        )

        self.assertFalse("crowded" in reply.lower() and "delay" in reply.lower())


if __name__ == "__main__":
    unittest.main()
