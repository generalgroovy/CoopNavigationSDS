from types import SimpleNamespace
import unittest

from coop_navigation_sds.NaturalLanguageGeneration.assistant.pipeline import (
    DialogState,
    VerbalTransformationPipeline,
    parse_heard_clock,
)
from coop_navigation_sds.NaturalLanguageGeneration.assistant.plugin_registry import SimplePlannerAgentBPlugin
from coop_navigation_sds.NaturalLanguageGeneration.caller.responder import LLMAgentAResponder
from coop_navigation_sds.TransportNetwork import get_test_case


class FailingModel:
    def generate_messages(self, _messages):
        raise AssertionError("model must not run until the heard request is complete")


class HeardStateTests(unittest.TestCase):
    def state(self, conversation):
        return DialogState(
            test_case=SimpleNamespace(
                scenario={
                    "start_station": "Bravo",
                    "destination_station": "Harbor",
                    "destination_stations": ["Harbor"],
                    "start_time_min": 487,
                    "transfer_time_min": 2,
                },
                persona={"preferences": {}},
            ),
            conversation=conversation,
        )

    def test_agent_b_does_not_use_hidden_scenario_when_asr_request_is_incomplete(self):
        state = self.state([
            ("Agent A", "But although it does event going to Harbour which should I take"),
        ])

        reply = VerbalTransformationPipeline(FailingModel()).run_agent_b(state)

        self.assertIn("repeat", reply.lower())
        self.assertIn("start", reply.lower())
        self.assertIn("did you mean", reply.lower())
        self.assertIn("Harbor", reply)
        self.assertNotIn("Bravo", reply)

    def test_agent_b_escalates_repeated_clarification_to_structured_fields(self):
        state = self.state([
            ("Agent A", "I need the rude to harder"),
            ("Agent B", "I heard that unclearly. Please repeat the start, destination, and time."),
            ("Agent A", "Still the rude there"),
        ])

        reply = VerbalTransformationPipeline(FailingModel()).run_agent_b(state)

        self.assertIn("each separately", reply.lower())
        self.assertIn("departure time", reply.lower())

    def test_agent_b_resets_after_configured_clarification_budget(self):
        state = self.state([
            ("Agent A", "I need the rude to harder"),
            ("Agent B", "Please repeat the start, destination, and time."),
            ("Agent A", "Still unclear"),
        ])
        state.scenario["clarification_max_attempts"] = 1

        reply = VerbalTransformationPipeline(FailingModel()).run_agent_b(state)

        self.assertIn("reset", reply.lower())
        self.assertIn("say only", reply.lower())

    def test_simple_agent_uses_only_journey_facts_it_heard(self):
        state = self.state([
            ("Agent A", "At 09:15 I am at Alpha and need to reach Echo."),
        ])

        heard = state.assistant_scenario
        reply = SimplePlannerAgentBPlugin().run_agent_b(state)

        self.assertEqual(heard["start_station"], "Alpha")
        self.assertEqual(heard["destination_station"], "Echo")
        self.assertEqual(heard["start_time_min"], 9 * 60 + 15)
        self.assertIn("Alpha", reply)
        self.assertIn("Echo", reply)

    def test_spoken_compact_clock_from_clean_speech_is_understood(self):
        self.assertEqual(parse_heard_clock("I am at Bravo at eight seven"), 8 * 60 + 7)
        state = self.state([
            ("Agent A", "I Emmett Bravo eight seven going to Harbor"),
        ])
        heard = state.assistant_scenario
        self.assertEqual(heard["start_station"], "Bravo")
        self.assertEqual(heard["destination_station"], "Harbor")
        self.assertEqual(heard["start_time_min"], 8 * 60 + 7)

    def test_agent_a_does_not_call_a_garbled_heard_route_valid(self):
        test_case = get_test_case("morning_peak_cross_city")
        responder = LLMAgentAResponder(FailingModel())

        reply = responder.reply(
            0,
            test_case.persona,
            test_case.scenario,
            [
                ("Agent A", test_case.opening_utterance()),
                (
                    "Agent B",
                    "Take wings of love had a harder it takes 12 minutes but no changes",
                ),
            ],
        )

        self.assertNotIn("valid route", reply.lower())
        self.assertTrue(
            "actual route" in reply.lower()
            or "restat" in reply.lower()
            or "missed" in reply.lower()
        )

    def test_agent_a_answers_structured_reset_with_trip_facts(self):
        test_case = get_test_case("morning_peak_cross_city")
        responder = LLMAgentAResponder(FailingModel())

        reply = responder.reply(
            1,
            test_case.persona,
            test_case.scenario,
            [
                ("Agent A", "The trip details were misunderstood."),
                (
                    "Agent B",
                    "Let's reset the trip details. Say only: starting station, destination station, and departure time.",
                ),
            ],
        )

        self.assertIn(test_case.scenario["start_station"], reply)
        self.assertIn(test_case.scenario["destination_station"], reply)
        self.assertIn("08:07", reply)
