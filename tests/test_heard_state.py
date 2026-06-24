from types import SimpleNamespace
import unittest

from coop_navigation_sds.NaturalLanguageGeneration.assistant.pipeline import (
    DialogState,
    VerbalTransformationPipeline,
    heard_trip_report,
    parse_heard_clock,
)
from coop_navigation_sds.NaturalLanguageGeneration.assistant.plugin_registry import SimplePlannerAgentBPlugin
from coop_navigation_sds.NaturalLanguageGeneration.caller.responder import LLMAgentAResponder
from coop_navigation_sds.NaturalLanguageGeneration.caller.prompting import agent_a_requested_trip_fact
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

        self.assertIn("starting station", reply.lower())
        self.assertNotIn("Bravo", reply)

    def test_agent_b_escalates_repeated_clarification_to_structured_fields(self):
        state = self.state([
            ("Agent A", "I need the rude to harder"),
            ("Agent B", "I heard that unclearly. Please repeat the start, destination, and time."),
            ("Agent A", "Still the rude there"),
        ])

        reply = VerbalTransformationPipeline(FailingModel()).run_agent_b(state)

        self.assertIn("starting station", reply.lower())
        self.assertIn("say", reply.lower())

    def test_agent_b_keeps_clarifying_after_configured_repair_budget(self):
        state = self.state([
            ("Agent A", "I need the rude to harder"),
            ("Agent B", "Please repeat the start, destination, and time."),
            ("Agent A", "Still unclear"),
        ])
        state.scenario["clarification_max_attempts"] = 1

        reply = VerbalTransformationPipeline(FailingModel()).run_agent_b(state)

        self.assertIn("reset the trip details", reply.lower())
        self.assertIn("starting station", reply.lower())
        self.assertNotIn("end this call", reply.lower())

    def test_agent_b_repair_budget_is_specific_to_missing_slot(self):
        state = self.state([
            ("Agent A", "I am going to Harbor at eight seven."),
            ("Agent B", "I missed the starting station. Please say only that."),
            ("Agent A", "Still unclear."),
        ])
        state.scenario["clarification_max_attempts"] = 1

        reply = VerbalTransformationPipeline(FailingModel()).run_agent_b(state)

        self.assertIn("starting station", reply.lower())
        self.assertIn("please say only", reply.lower())
        self.assertNotIn("end this call", reply.lower())

        state = self.state([
            ("Agent A", "Bravo."),
            ("Agent B", "I missed the starting station. Please say only that."),
            ("Agent A", "Bravo going to Harbor."),
        ])
        state.scenario["clarification_max_attempts"] = 1

        reply = VerbalTransformationPipeline(FailingModel()).run_agent_b(state)

        self.assertIn("departure time", reply.lower())
        self.assertNotIn("end this call", reply.lower())

    def test_slot_repair_does_not_reverse_start_and_destination(self):
        state = self.state([
            ("Agent A", "I am going to Harbor at eight seven."),
            ("Agent B", "I missed the starting station. Please say only that."),
            ("Agent A", "Bravo."),
        ])

        heard = state.assistant_scenario

        self.assertEqual(heard["start_station"], "Bravo")
        self.assertEqual(heard["destination_station"], "Harbor")
        self.assertEqual(heard["start_time_min"], 8 * 60 + 7)

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
        self.assertEqual(parse_heard_clock("I am at Bravo at eight oh seven"), 8 * 60 + 7)
        self.assertEqual(parse_heard_clock("I am at Bravo at eight o seven"), 8 * 60 + 7)
        self.assertEqual(parse_heard_clock("I am at Bravo at eight 0 7"), 8 * 60 + 7)
        self.assertEqual(parse_heard_clock("I am at Bravo at 8 oh 7"), 8 * 60 + 7)
        self.assertEqual(parse_heard_clock("Shore, I said Bravo at 8-7, going to Harbor"), 8 * 60 + 7)
        self.assertEqual(parse_heard_clock("8, 7."), 8 * 60 + 7)
        self.assertEqual(parse_heard_clock("8 7"), 8 * 60 + 7)
        self.assertEqual(parse_heard_clock("departure time 807"), 8 * 60 + 7)
        self.assertEqual(parse_heard_clock("departure time eight to seven"), 8 * 60 + 7)
        self.assertEqual(parse_heard_clock("departure time eight oh seventy"), 8 * 60 + 7)
        state = self.state([
            ("Agent A", "I Emmett Bravo eight seven going to Harbor"),
        ])
        heard = state.assistant_scenario
        self.assertEqual(heard["start_station"], "Bravo")
        self.assertEqual(heard["destination_station"], "Harbor")
        self.assertEqual(heard["start_time_min"], 8 * 60 + 7)

    def test_departure_time_repair_accepts_asr_digit_pair(self):
        state = self.state([
            ("Agent A", "Shore I said Bravo at 8-7 going to Harbor please use those"),
            ("Agent B", "I missed the departure time. Please say only that."),
            ("Agent A", "8, 7."),
        ])

        heard = state.assistant_scenario

        self.assertEqual(heard["start_station"], "Bravo")
        self.assertEqual(heard["destination_station"], "Harbor")
        self.assertEqual(heard["start_time_min"], 8 * 60 + 7)
        self.assertEqual(state.missing_trip_slots, ())

    def test_focused_departure_time_repair_is_self_identifying(self):
        test_case = get_test_case("morning_peak_cross_city")

        reply = agent_a_requested_trip_fact(
            "I missed the departure time. Please say only that.",
            test_case.scenario,
        )
        state = self.state([
            ("Agent A", "I am at Bravo going to Harbor."),
            ("Agent B", "I missed the departure time. Please say only that."),
            ("Agent A", "Departure time 807."),
        ])

        self.assertEqual(reply, "Departure time: 08:07.")
        self.assertEqual(state.assistant_scenario["start_time_min"], 8 * 60 + 7)
        self.assertNotIn("departure time", SimplePlannerAgentBPlugin().run_agent_b(state).lower())

    def test_route_numbers_do_not_become_departure_time_without_time_context(self):
        self.assertIsNone(parse_heard_clock("Take line B12 from Bravo to Hotel, then M4 to Harbor.", allow_contextless=False))
        state = self.state([
            ("Agent A", "Take line B12 from Bravo to Hotel, then M4 to Harbor."),
        ])

        report = heard_trip_report(state.conversation)

        self.assertEqual(report["facts"]["start_station"], "Bravo")
        self.assertEqual(report["facts"]["destination_station"], "Harbor")
        self.assertIsNone(report["facts"]["start_time_min"])
        self.assertIn("start_time_min", report["missing_slots"])

    def test_heard_trip_state_contains_metric_evidence(self):
        state = self.state([
            ("Agent A", "I Emmett Bravo eight seven going to Harbor"),
        ])

        report = state.heard_trip_state

        self.assertEqual(report["facts"]["start_station"], "Bravo")
        self.assertEqual(report["facts"]["destination_station"], "Harbor")
        self.assertEqual(report["facts"]["start_time_min"], 8 * 60 + 7)
        self.assertEqual(report["missing_slots"], ())
        self.assertEqual(report["completeness"], 1.0)
        self.assertEqual(report["evidence"]["start_time_min"]["method"], "clock_expression")

    def test_full_time_repetition_after_departure_repair_resumes_route_dialogue(self):
        state = self.state([
            (
                "Agent A",
                "I'm starting at station Bravo at eight oh seven. "
                "I need to take transit lines to Harbor. Which lines should I take?",
            ),
            ("Agent B", "I missed the departure time. Please say only that."),
            (
                "Agent A",
                "Sure: I said Bravo at eight oh seven, going to Harbor. Please use those.",
            ),
        ])

        heard = state.assistant_scenario
        reply = SimplePlannerAgentBPlugin().run_agent_b(state)

        self.assertEqual(heard["start_station"], "Bravo")
        self.assertEqual(heard["destination_station"], "Harbor")
        self.assertEqual(heard["start_time_min"], 8 * 60 + 7)
        self.assertNotIn("missed the departure time", reply.lower())
        self.assertIn("Harbor", reply)

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
        self.assertIn("did you mean", reply.lower())
        self.assertNotIn("12 minutes", reply.lower())

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
