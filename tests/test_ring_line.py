import unittest

from minillama.model.metro_data import LINES
from minillama.model.route_constraints import optimal_constraint_route, route_constraint_gap
from minillama.model.route_planner import (
    estimate_route_time,
    line_direction_sequences,
    route_duration_breakdown,
    route_text_from_steps,
)


class RingLineTests(unittest.TestCase):
    def test_ring_line_uses_one_canonical_sequence(self):
        ring_name = next((name for name, data in LINES.items() if data.get("kind") == "Ring"), None)
        self.assertIsNotNone(ring_name)

        sequences = line_direction_sequences(ring_name)
        self.assertEqual(len(sequences), 2)
        self.assertGreaterEqual(len(sequences[0]), 3)
        self.assertEqual(sequences[0][0], sequences[0][-1])
        self.assertEqual(sequences[1][0], sequences[1][-1])

    def test_same_line_through_station_has_no_intermediate_wait_or_transfer(self):
        estimate = estimate_route_time(["Alpha", "Bravo", "Charlie"], 480, 4)
        self.assertIsNotNone(estimate)
        _, steps = estimate

        self.assertEqual([step["line"] for step in steps], ["Ring", "Ring"])
        self.assertEqual(steps[1]["depart"], steps[0]["arrive"])
        self.assertEqual(steps[1]["wait"], 0)
        self.assertEqual(steps[1]["transfer"], 0)
        self.assertEqual(route_duration_breakdown(steps)["transfer"], 0)

        proposal = route_text_from_steps(steps)
        self.assertEqual(proposal.count("Take Ring"), 1)
        self.assertIn("Take Ring from Alpha", proposal)
        self.assertIn("to Charlie", proposal)
        self.assertIn("Boarding: Alpha to Charlie", proposal)

    def test_startup_constraint_route_is_available_for_proposal_comparison(self):
        scenario = {
            "start_station": "Alpha",
            "destination_station": "Echo",
            "start_time_min": 480,
            "transfer_time_min": 4,
        }
        persona = {
            "preferences": {
                "priority": "fast route",
                "switching": "avoid unnecessary line changes",
                "fullness": "dislikes very full train cars",
            }
        }

        target = optimal_constraint_route(scenario, persona)
        self.assertIsNotNone(target)
        self.assertEqual(target.route[0], "Alpha")
        self.assertEqual(target.route[-1], "Echo")

        estimate = estimate_route_time(target.route, 480, 4)
        self.assertIsNotNone(estimate)
        arrival, steps = estimate
        gap = route_constraint_gap(steps, arrival - 480, target)
        self.assertEqual(gap["duration_gap_min"], 0)
        self.assertEqual(gap["line_change_gap"], 0)
